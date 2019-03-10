import tensorflow as tf
import numpy as np
from cshogi import *
from dlshogi_zero.nn.resnet import ResNet
from dlshogi_zero.features import *
from dlshogi_zero.database import *
from dlshogi_zero.uct.uct_node import *

from threading import Thread, Lock
from array import array
from collections import defaultdict

RES_BLOCKS = 10
FILTERS = 192

# UCBの定数
C_BASE = 19652
C_INIT = 1.25
# 投了する勝率の閾値
RESIGN_THRESHOLD = 0.01
# 温度パラメータ
TEMPERATURE = 1.0
# 最大手数
MAX_MOVE_COUNT = 512
# キューに追加されたときの戻り値（数値に意味はない）
QUEUING = 2.0
# ハッシュサイズ
UCT_HASH_SIZE = 65536

database = None
nodes = 0
written_nodes = 0
limit_games = 10000
games = 0
stopflg = False

def init_database(filepath):
    global database
    database = TrainingDataBase(filepath)

def term_database():
    database.close()

def stop():
    global stop
    stop = True

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

class PlayoutInfo:
    def __init__(self):
        self.halt = 0 # 探索を打ち切る回数
        self.count = 0 # 現在の探索回数

# 探索経路のノード
class TrajectorEntry:
    def __init__(self, uct_node, current, next_index):
        self.uct_node = uct_node
        self.current = current
        self.next_index = next_index

class SelfPlayAgentGroup:
    def __init__(self, id, policy_value_batch_maxsize, parent):
        self.node_hash = NodeHash(UCT_HASH_SIZE)
        self.uct_node = [UctNode() for _ in range(UCT_HASH_SIZE)]

        self.parent = parent
        self.thread_selfplay = None
        self.running = False
        self.group_id = id

        # キュー
        self.policy_value_batch_maxsize = policy_value_batch_maxsize
        self.features = np.empty((policy_value_batch_maxsize, MAX_FEATURES, 9, 9), dtype=np.float32)
        self.policy_value_hash_index = array('i', [0] * policy_value_batch_maxsize)
        self.current_policy_value_batch_index = 0

        # 自己対局エージェント
        self.agents = [SelfPlayAgent(self, self.node_hash, self.uct_node, i) for i in range(policy_value_batch_maxsize)]

    def selfplay(self):
        # 探索経路のバッチ
        trajectories_batch = [[]] * self.policy_value_batch_maxsize

        # 全スレッドが生成したゲーム数が上限ゲーム数以上になったら終了
        while games < limit_games and not stopflg:
            # すべての対局についてシミュレーションを行う
            for agent, trajectories in zip(self.agents, trajectories_batch):
                trajectories.clear()
                agent.playout(trajectories)
        
            # 評価
            eval_node()

            # バックアップ
            result = 0.0
            for trajectories in trajectories_batch:
                for i in range(len(trajectories) - 1, 0, -1):
                    current_next = trajectories[i]
                    uct_node = current_next.uct_node
                    current = current_next.current
                    next_index = current_next.next_index
                    uct_child = uct_node[current].child
                    if i == len(trajectories) - 1:
                        child_index = uct_child[next_index].index
                        result = -uct_node[child_index].value_win
                    self.update_result(uct_node, uct_child[next_index], result, current)
                    result = -result

            # 次のシミュレーションへ
            for agent in self.agents:
                agent.next_step()

        self.running = False

    # 局面の評価
    def eval_node(self):
        policy_value_batch_size = self.current_policy_value_batch_index
        
        # predict
        logits, values = parent.model_forward(policy_value_batch_size, self.features)

        for i, (logit, value) in enumerate(zip(logits, values)):
            index = self.policy_value_hash_index[i]

            # グループ内の対局は1スレッドで行うためロックは不要
            child_num = self.uct_node[index].child_num
            uct_child = self.uct_node[index].child
            color = self.node_hash[index].color

            # 合法手一覧
            legal_move_probabilities = np.empty(child_num, dtype=np.float32)
            for j in range(child_num):
                move = uct_child[j].move
                move_label = make_action_label(move)
                legal_move_probabilities[j] = logit[move_label]

            # Boltzmann distribution
            softmax_tempature_with_normalize(legal_move_probabilities, TEMPERATURE)

            for j in range(child_num):
                uct_child[j].nnrate = legal_move_probabilities[j]

            uct_node[index].value_win = value
            uct_node[index].evaled = True

    # ノードをキューに追加
    def queuing_node(self, board, moves, repetitions, index):
        # set all zero
        self.features[self.current_policy_value_batch_index].fill(0)

        # 入力特徴量に履歴局面を設定
        for i, (move, repetition) in enumerate(zip(moves[-1:-MAX_HISTORY-1:-1], repetitions[-1:-MAX_HISTORY-1:-1])):
            make_position_features(board, repetition, self.features[current_policy_value_batch_index], i)
            board.pop(move)

        # 局面を戻す
        for move in moves[-MAX_HISTORY::]:
            board.push(move)

        # 入力特徴量に手番と手数を設定
        make_color_totalmovecout_features(board.turn, board.ply, self.features[current_policy_value_batch_index])

        self.policy_value_hash_index[current_policy_value_batch_index] = index
        self.current_policy_value_batch_index += 1

    # スレッド開始
    def run(self):
        # 自己対局用スレッド
        self.running = True
        self.thread_selfplay = Thread(target=self.selfplay)
        self.thread_selfplay.start()

    # スレッド終了待機
    def join(self):
        # 自己対局用スレッド
        self.thread_selfplay.join()
        print('thread_selfplay.join()')

class SelfPlayAgentGroupPair:
    def __init__(self, model_path, policy_value_batch_maxsize):
        self.model = ResNet(res_blocks=RES_BLOCKS, filters=FILTERS)
        self.groups = [SelfPlayAgentGroup(i, policy_value_batch_maxsize, self) for i in range(1)]
        self.gpu_lock = Lock()

    def model_forward(self, x):
        with gpu_lock:
            return self.model.predict(x)

    def running(self):
        for group in self.groups:
            if group.running:
                return True
        return False

    def run(self):
        for group in self.groups:
            group.run()

    def join(self):
        for group in self.groups:
            group.join()
            print('group.join()')

class SelfPlayAgent:
    def __init__(self, grp, node_hash, uct_node, id):
        self.grp = grp
        self.node_hash = node_hash
        self.uct_node = uct_node
        self.id = id

        self.playouts = 0
        self.board = Board()
        self.moves = []
        
        # 1ゲーム分の訓練データ
        self.chunk = []
        # 履歴局面
        self.hcprs = np.empty(MAX_MOVE_COUNT, dtype=HcpAndRepetition)
        # 同一局面の繰り返し数
        self.repetitions = []
        self.repetition_hash = defaultdict(int)

    # 着手
    def do_move(self, move):
        self.moves.append(move)
        self.board.push(move)
        key = self.board.zobrist_hash()
        if self.board.is_draw() == REPETITION_DRAW:
            self.repetition_hash[key] += 1
        self.repetitions.append(self.repetition_hash[key])

    def undo_move(self):
        if self.repetitions[-1] > 0:
            self.repetition_hash[self.board.zobrist_hash()] -= 1
        self.repetitions.pop()
        self.board.pop(self.moves[-1])
        self.moves.pop()

    # 教師局面をチャンクに追加
    def add_teacher(self, current_root_node):
        i = self.board.ply
        # hcp
        self.board.to_hcp(self.hcprs[i]['hcp'])
        # repetition
        self.hcprs[i]['repetition'] = self.repetitions[-1]
        # legalMoves
        legal_moves = current_root_node.child_move[0:current_root_node.child_num]
        # visits
        visits = current_root_node.child_move_count[0:current_root_node.child_num]

        # history
        hcprs = np.zeros(MAX_HISTORY, dtype=HcpAndRepetition)
        if i < MAX_HISTORY:
            hist = i + 1
        else:
            hist = MAX_HISTORY
        for prev in range(hist):
            hcprs[prev] = self.hcprs[i - prev]

        self.chunk.append((hcprs.data, self.board.ply, legal_moves.data, visits.data))
        nodes += 1

    # シミュレーションを1回行う
    def playout(self, trajectories):
        while True:
            # 手番開始
            if self.playouts == 0:
                # ハッシュクリア
                self.node_hash.clear_hash(id)

                # ルートノード展開
                self.current_root = self.expand_node()

                # 詰みのチェック
                if self.uct_node[self.current_root].child_num == 0:
                    if self.board.turn == BLACK:
                        game_result = WHITE_WIN
                    else:
                        game_result = BLACK_WIN

                    self.next_game(game_result)
                    continue
                elif self.uct_node[self.current_root].child_num == 1:
                    # 1手しかないときは、その手を指して次の手番へ
                    self.do_move(uct_node[current_root].child[0].move)
                    self.playouts = 0
                    continue
                elif self.board.is_nyugyoku():
                    # 入玉宣言勝ち
                    if self.board.turn == BLACK:
                        game_result = BLACK_WIN
                    else:
                        game_result = WHITE_WIN

                    self.next_game(game_result)
                    continue

                # ルート局面をキューに追加
                self.grp.queuing_node(self.board, self.moves, self.current_root)

                return

            # プレイアウト
            result = self.uct_search(self.current_root, trajectories)
            if result != QUEUING:
                self.next_step()
                continue

            return

    # 次の手に進める
    def next_step(self):
        # プレイアウト回数加算
        self.playouts += 1
    
        # 探索終了判定
        if self.interruption_check():
    
            # 探索回数最大の手を見つける
            current_root_node = self.uct_node[self.current_root]
            child_move_count = current_root_node.child_move_count
            select_index = np.argmax(child_move_count)
    
            # 選択した着手の勝率の算出
            best_wp = current_root_node.child_win[select_index] / child_move_count[select_index]
            best_move = current_root_node.child_move[select_index]
    
            # 勝率が閾値を超えた場合、ゲーム終了
            if WINRATE_THRESHOLD < abs(best_wp):
                if self.board.turn == BLACK:
                    game_result = WHITE_WIN if best_wp < 0 else BLACK_WIN
                else:
                    game_result = BLACK_WIN if best_wp < 0 else WHITE_WIN
    
                self.next_game(game_result)
                return
    
            # 局面追加
            self.add_teacher(current_root_node)
    
            # 一定の手数以上で引き分け
            if self.board.ply >= MAX_MOVE_COUNT:
                self.next_game(DRAW)
                return
    
            # 着手
            self.do_move(best_move)
            self.playouts = 0

    def next_game(self, game_result):
        # 局面出力
        if len(self.chunk) > 0:
            # 勝敗を1局全てに付ける
            for i in range(len(self.chunk)):
                self.chunk[i] += (game_result,)
            database.write_chunk(self.chunk)
            made_teacher_nodes += len(self.chunk)
            games += 1
    
        # 新しいゲーム
        self.playouts = 0
        self.board.reset()
        self.chunk.clear()
        self.repetitions.clear()

    # UCB値が最大の手を求める
    def select_max_ucb_child(self, current_node):
        child_num = current_node.child_num
        child_win = current_node.child_win
        child_move_count = current_node.child_move_count

        q = np.divide(child_win, child_move_count, out=np.repeat(np.float32(0), child_num), where=child_move_count != 0)
        c = np.log((np.float32(current_node.move_count) + C_BASE + 1.0) / C_BASE) + C_INIT
        u = np.sqrt(np.float32(current_node.move_count)) / (1 + child_move_count)
        ucb = q + c * u * current_node.nnrate

        return np.argmax(ucb)

    # ノードの展開
    def expand_node(self):
        index = self.node_hash.find_same_hash_index(self.board.zobrist_hash(), self.board.turn, self.board.move_number, self.id)

        # 合流先が検知できれば, それを返す
        if not index == UCT_HASH_SIZE:
            return index
    
        # 空のインデックスを探す
        index = self.node_hash.search_empty_index(self.board.zobrist_hash(), self.board.turn, self.board.move_number, self.id)

        # 現在のノードの初期化
        current_node = self.uct_node[index]
        current_node.move_count = 0
        current_node.win = 0.0
        current_node.child_num = 0
        current_node.evaled = False
        current_node.value_win = 0.0

        # 候補手の展開
        current_node.child_move = [move for move in self.board.legal_moves]
        child_num = len(current_node.child_move)
        current_node.child_index = [NOT_EXPANDED for _ in range(child_num)]
        current_node.child_move_count = np.zeros(child_num, dtype=np.int32)
        current_node.child_win = np.zeros(child_num, dtype=np.float32)

        # 子ノードの個数を設定
        current_node.child_num = child_num

        return index

    # 探索を打ち切るか確認
    def interruption_check(self):
        child_num = self.uct_node[self.current_root].child_num
        child_move_count = self.uct_node[self.current_root].child_move_count
        rest = self.po_info.halt - self.po_info.count

        # 探索回数が最も多い手と次に多い手を求める
        second, first = child_move_count[np.argpartition(child_move_count, -2)[-2:]]

        # 残りの探索を全て次善手に費やしても最善手を超えられない場合は探索を打ち切る
        if first - second > rest:
            return True
        else:
            return False

    # UCT探索
    def uct_search(self, current, trajectories):
        current_node = self.uct_node[current]

        # 詰みのチェック
        if current_node.child_num == 0:
            return 1.0 # 反転して値を返すため1を返す

        child_move = current_node.child_move
        child_move_count = current_node.child_move_count
        child_index = current_node.child_index

        # UCB値が最大の手を求める
        next_index = self.select_max_ucb_child(current_node)
        # 選んだ手を着手
        self.do_move(child_move[next_index])

        # 経路を記録
        trajectories.append((current_node, next_index))

        # ノードの展開の確認
        if child_index[next_index] == NOT_EXPANDED:
            # ノードの展開
            index = self.expand_node()
            child_index[next_index] = index
            child_node = self.uct_node[index]

            if child_node.evaled:
                # 合流
                # valueを報酬として返す
                result = -child_node.value_win
            elif child_node.child_num == 0:
                # 詰み
                child_node.value_win = -1.0
                child_node.evaled = True
                result = 1.0
            else:
                # ノードをキューに追加
                self.grp.queuing_node(self.board, self.moves, self.repetitions, index)
                result = QUEUING
        else:
            # 手番を入れ替えて1手深く読む
            result = self.uct_search(child_index[next_index], trajectories)

        # 手を戻す
        self.undo_move()

        if result == QUEUING:
            return QUEUING

        # 探索結果の反映
        current_node.win += result
        current_node.move_count += 1
        current_node.child_win[next_index] += result
        current_node.child_move_count[next_index] += 1

        return -result
