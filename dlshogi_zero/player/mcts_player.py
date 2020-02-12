import tensorflow as tf
from tensorflow.keras.backend import set_session
from tensorflow.keras.models import load_model
import numpy as np
from cshogi import *
from dlshogi_zero.features import *
from dlshogi_zero.database import *
from dlshogi_zero.uct.uct_node import *
from dlshogi_zero.player.base_player import BasePlayer

from collections import defaultdict
import time
import logging
import os
import math

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

# UCBの定数
C_BASE = 19652
C_INIT = 1.25
# ノイズの定数
ALPHA = 0.15
EPSILON = 0.0
# 投了する勝率の閾値
RESIGN_THRESHOLD = 0.01
# 温度パラメータ
TEMPERATURE = 1.0
# 最大手数
MAX_MOVE_COUNT = 512
# キューに追加されたときの戻り値（数値に意味はない）
QUEUING = 2.0
# 探索を破棄するときの戻り値（数値に意味はない）
DISCARDED = 3.0
# Virtual Loss
VIRTUAL_LOSS = 1

CONST_PLAYOUT = 1000

def softmax_temperature_with_normalize(logits, temperature):
    # 温度パラメータを適用
    logits /= temperature

    # 確率を計算(オーバーフローを防止するため最大値で引く)
    max_logit = max(logits)
    probabilities = np.exp(logits - max_logit)

    # 合計が1になるように正規化
    sum_probabilities = sum(probabilities)
    probabilities /= sum_probabilities

    return probabilities

def update_result(current_node, next_index, result):
    current_node.win += result
    current_node.move_count += 1 - VIRTUAL_LOSS
    current_node.child_win[next_index] += result
    current_node.child_move_count[next_index] += 1 - VIRTUAL_LOSS

class PlayoutInfo:
    def __init__(self):
        self.halt = 0 # 探索を打ち切る回数
        self.count = 0 # 現在の探索回数
        
class MCTSPlayer(BasePlayer):
    def __init__(self):
        # モデルファイルのパス
        self.modelfile = r'H:\model.h5'
        self.model = None # モデル

        self.board = Board()
        self.moves = []
        
        # 同一局面の繰り返し数
        self.repetitions = [1]
        self.repetition_hash = defaultdict(int)

        # プレイアウト回数管理
        self.po_info = PlayoutInfo()
        self.playout = CONST_PLAYOUT

        # キュー
        self.policy_value_batch_maxsize = 32
        self.features = None
        self.policy_value_node = None

    def usi(self):
        print('id name dlshogi-zero')
        print('option name modelfile type string default ' + self.modelfile)
        print('option name playout type spin default ' + str(self.playout) + ' min 100 max 10000')
        print('option name batchsize type spin default ' + str(self.policy_value_batch_maxsize) + ' min 1 max 256')
        print('option name temperature type spin default ' + str(int(TEMPERATURE * 100)) + ' min 10 max 1000')
        print('option name noise type spin default ' + str(int(EPSILON * 100)) + ' min 0 max 100')
        print('usiok')

    def setoption(self, option):
        if option[1] == 'modelfile':
            self.modelfile = option[3]
        elif option[1] == 'playout':
            self.playout = int(option[3])
        elif option[1] == 'batchsize':
            self.policy_value_batch_maxsize = int(option[3])
        elif option[1] == 'temperature':
            global TEMPERATURE
            TEMPERATURE = int(option[3]) / 100
        elif option[1] == 'noise':
            global EPSILON
            EPSILON = int(option[3]) / 100

    def isready(self):
        # モデルをロード
        if self.model is None:
            self.model = load_model(self.modelfile)

        # キューを初期化
        if self.features is None or len(self.features) != self.policy_value_batch_maxsize:
            self.features = np.empty((self.policy_value_batch_maxsize, MAX_FEATURES, 81), dtype=np.float32)
            self.policy_value_node = [None for _ in range(self.policy_value_batch_maxsize)]
        self.current_policy_value_batch_index = 0

        # 盤面情報をクリア
        self.board.reset()
        self.moves.clear()
        self.repetitions.clear()
        self.repetitions.append(1)
        self.repetition_hash.clear()

        # 1手目を速くするためモデルをキャッシュする
        current_node = self.expand_node()
        self.queuing_node(self.board, self.moves, self.repetitions, current_node)
        self.eval_node()

        print('readyok')
    
    def position(self, moves):
        self.moves.clear()
        self.repetitions.clear()
        self.repetitions.append(1)
        self.repetition_hash.clear()

        if moves[0] == 'startpos':
            self.board.reset()
            for move_usi in moves[2:]:
                self.do_move(self.board.move_from_usi(move_usi))

        elif moves[0] == 'sfen':
            self.board.set_sfen(' '.join(moves[1:]))

        # for debug
        if __debug__:
            print(self.board)
    
    def go(self):
        if self.board.is_game_over():
            print('bestmove resign')
            return

        # 入玉宣言勝ち
        if self.board.is_nyugyoku():
            print('bestmove win')
            return

        # 探索情報をクリア
        self.po_info.count = 0

        # 探索開始時刻の記録
        begin_time = time.time()

        # 探索回数の閾値を設定
        self.po_info.halt = self.playout

        # ルートノードの展開
        self.current_root_node = self.expand_node()
        current_node = self.current_root_node

        # 候補手が1つの場合は、その手を返す
        child_num = current_node.child_num
        child_move = current_node.child_move
        if child_num == 1:
            print('bestmove', move_to_usi(child_move[0]).decode())
            return

        # ルートノードを評価
        self.current_policy_value_batch_index = 0
        self.queuing_node(self.board, self.moves, self.repetitions, current_node)
        self.eval_node()

        # 探索経路のバッチ
        trajectories_batch = []

        # 探索回数が閾値を超える、または探索が打ち切られたらループを抜ける
        while self.po_info.count < self.po_info.halt:
            trajectories_batch.clear()
            self.current_policy_value_batch_index = 0

            # すべての探索についてシミュレーションを行う
            for _ in range(self.policy_value_batch_maxsize):
                trajectories_batch.append([])
                result = self.uct_search(current_node, 0, trajectories_batch[-1])

                if result != DISCARDED:
                    # 探索回数を1回増やす
                    self.po_info.count += 1

                # 評価中の葉ノードに達した、もしくはバックアップ済みため破棄する
                if result == DISCARDED or result != QUEUING:
                    trajectories_batch.pop()
        
            # 評価
            if self.current_policy_value_batch_index > 0:
                self.eval_node()

            # バックアップ
            for trajectories in trajectories_batch:
                for i in reversed(range(len(trajectories))):
                    (current_node, next_index) = trajectories[i]
                    if i == len(trajectories) - 1:
                        # 葉ノード
                        child_node = current_node.child_nodes[next_index]
                        result = -child_node.value_win
                    update_result(current_node, next_index, result)
                    result = -result

            # 探索を打ち切るか確認
            if self.interruption_check():
                break

        # 探索にかかった時間を求める
        finish_time = time.time() - begin_time

        child_move_count = current_node.child_move_count
        # 訪問回数最大の手を選択する
        selected_index = np.argmax(child_move_count)

        child_win = current_node.child_win

        # for debug
        if __debug__:
            for i in range(child_num):
                print('{:3}:{:5} move_count:{:4} nn_rate:{:.5f} win_rate:{:.5f}'.format(
                    i, move_to_usi(child_move[i]), child_move_count[i],
                    current_node.nnrate[i],
                    child_win[i] / child_move_count[i] if child_move_count[i] > 0 else 0))

        # 選択した着手の勝率の算出（valueの範囲は[-1,1]なので[0,1]にする）
        best_wp = ((child_win[selected_index] / child_move_count[selected_index]) + 1) / 2

        # 閾値未満の場合投了
        if best_wp < RESIGN_THRESHOLD:
            print('bestmove resign')
            return

        bestmove = child_move[selected_index]

        # 勝率を評価値に変換
        if best_wp == 1.0:
            cp = 30000
        else:
            cp = int(-math.log(1.0 / best_wp - 1.0) * 600)

        print('info nps {} time {} nodes {} score cp {} pv {}'.format(
            int(current_node.move_count / finish_time),
            int(finish_time * 1000),
            current_node.move_count,
            cp, move_to_usi(bestmove)))

        print('bestmove', move_to_usi(bestmove))

    # 局面の評価
    def eval_node(self):
        # predict
        logits, values = self.model.predict(self.features[0:self.current_policy_value_batch_index].reshape((self.current_policy_value_batch_index, MAX_FEATURES, 9, 9)))

        for i, (logit, value) in enumerate(zip(logits, values)):
            current_node = self.policy_value_node[i]

            child_num = current_node.child_num
            child_move = current_node.child_move

            # 合法手一覧
            legal_move_probabilities = np.empty(child_num, dtype=np.float32)
            for j in range(child_num):
                move = child_move[j]
                move_label = make_action_label(move)
                legal_move_probabilities[j] = logit[move_label]

            # Boltzmann分布
            probabilities = softmax_temperature_with_normalize(legal_move_probabilities, TEMPERATURE)

            # ノードの値を更新
            current_node.nnrate = probabilities
            current_node.value_win = float(value)
            current_node.evaled = True

    # ノードをキューに追加
    def queuing_node(self, board, moves, repetitions, node):
        assert len(moves) == len(repetitions) - 1

        # set all zero
        self.features[self.current_policy_value_batch_index].fill(0)

        # 入力特徴量に現局面を設定
        make_position_features(board, repetitions[-1], self.features[self.current_policy_value_batch_index], 0)

        # 入力特徴量に履歴局面を設定
        for i, (move, repetition) in enumerate(zip(moves[-1:-MAX_HISTORY:-1], repetitions[-2:-MAX_HISTORY-1:-1])):
            board.pop(move)
            make_position_features(board, repetition, self.features[self.current_policy_value_batch_index], i + 1)

        # 局面を戻す
        for move in moves[-MAX_HISTORY+1::]:
            board.push(move)

        # 入力特徴量に手番と手数を設定
        make_color_totalmovecout_features(board.turn, board.move_number, self.features[self.current_policy_value_batch_index])

        self.policy_value_node[self.current_policy_value_batch_index] = node
        self.current_policy_value_batch_index += 1


    # 着手
    def do_move(self, move):
        self.moves.append(move)
        self.board.push(move)
        key = self.board.zobrist_hash()
        self.repetition_hash[key] += 1
        self.repetitions.append(self.repetition_hash[key])

    def undo_move(self):
        self.repetition_hash[self.board.zobrist_hash()] -= 1
        self.repetitions.pop()
        self.board.pop(self.moves[-1])
        self.moves.pop()

    # UCB値が最大の手を求める
    def select_max_ucb_child(self, current_node, depth):
        child_num = current_node.child_num
        child_win = current_node.child_win
        child_move_count = current_node.child_move_count

        q = np.divide(child_win, child_move_count, out=np.repeat(np.float32(-1), child_num), where=child_move_count != 0)
        c = np.log((np.float32(current_node.move_count) + C_BASE + 1.0) / C_BASE) + C_INIT
        if current_node.move_count == 0:
            u = 1.0
        else:
            u = np.sqrt(np.float32(current_node.move_count)) / (1 + child_move_count)
        if depth == 0:
            # Dirichlet noise
            eta = np.random.dirichlet([ALPHA] * len(current_node.nnrate))
            p = (1 - EPSILON) * current_node.nnrate + EPSILON * eta
        else:
            p = current_node.nnrate
        ucb = q + c * u * p

        return np.argmax(ucb)

    # ノードの展開
    def expand_node(self):
        # ノードを作成する
        current_node = UctNode()

        # 候補手の展開
        current_node.child_move = [move for move in self.board.legal_moves]
        child_num = len(current_node.child_move)
        current_node.child_nodes = [None for _ in range(child_num)]
        current_node.child_move_count = np.zeros(child_num, dtype=dtypeVisit)
        current_node.child_win = np.zeros(child_num, dtype=np.float32)

        # 子ノードの個数を設定
        current_node.child_num = child_num

        return current_node

    # 探索を打ち切るか確認
    def interruption_check(self):
        child_num = self.current_root_node.child_num
        child_move_count = self.current_root_node.child_move_count
        rest = self.po_info.halt - self.po_info.count

        # 探索回数が最も多い手と次に多い手を求める
        second, first = child_move_count[np.argpartition(child_move_count, -2)[-2:]]

        # 残りの探索を全て次善手に費やしても最善手を超えられない場合は探索を打ち切る
        if first - second > rest:
            return True
        else:
            return False

    # UCT探索
    def uct_search(self, current_node, depth, trajectories):
        # 詰みのチェック
        if current_node.child_num == 0:
            return 1.0 # 反転して値を返すため1を返す

        # 千日手チェック
        if self.repetitions[-1] == 4:
            draw = self.board.is_draw()
            if draw == REPETITION_WIN:
                # 連続王手の千日手
                return -1.0
            elif draw == REPETITION_LOSE:
                # 連続王手の千日手
                return 1.0
            else:
                # 千日手
                return 0.0

        # 他の探索がpolicy計算中のため破棄する
        if current_node.evaled == False:
            return DISCARDED

        child_move = current_node.child_move
        child_move_count = current_node.child_move_count
        child_nodes = current_node.child_nodes

        # UCB値が最大の手を求める
        next_index = self.select_max_ucb_child(current_node, depth)
        # 選んだ手を着手
        self.do_move(child_move[next_index])

        # 経路を記録
        trajectories.append((current_node, next_index))

        # Virtual Lossを加算
        current_node.move_count += VIRTUAL_LOSS
        child_move_count[next_index] += VIRTUAL_LOSS

        # ノードの展開の確認
        if child_nodes[next_index] == None:
            # ノードの展開
            child_node = self.expand_node()
            child_nodes[next_index] = child_node

            if child_node.child_num == 0:
                # 詰み
                result = 1.0 # 反転して値を返すため1を設定
            elif self.repetitions[-1] == 4:
                draw = self.board.is_draw()
                if draw == REPETITION_WIN:
                    # 連続王手の千日手
                    result = -1.0
                elif draw == REPETITION_LOSE:
                    # 連続王手の千日手
                    result = 1.0
                else:
                    # 千日手
                    result = 0.0
            else:
                # ノードをキューに追加
                self.queuing_node(self.board, self.moves, self.repetitions, child_node)
                result = QUEUING
        else:
            # 手番を入れ替えて1手深く読む
            result = self.uct_search(child_nodes[next_index], depth + 1, trajectories)

        # 手を戻す
        self.undo_move()

        if result == QUEUING:
            return QUEUING
        elif result == DISCARDED:
            # Virtual Lossを戻す
            current_node.move_count -= VIRTUAL_LOSS
            child_move_count[next_index] -= VIRTUAL_LOSS
            return DISCARDED

        # 探索結果の反映
        update_result(current_node, next_index, result)

        return -result
