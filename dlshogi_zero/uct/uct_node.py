import shogi

UCT_HASH_SIZE = 65536 # 2のn乗であること
UCT_HASH_LIMIT = UCT_HASH_SIZE * 9 / 10

# 未展開のノードのインデックス
NOT_EXPANDED = -1

# 未使用を表す
NOT_USE = -1

def hash_to_index(hash):
    return ((hash & 0xffffffff) ^ ((hash >> 32) & 0xffffffff)) & (UCT_HASH_SIZE - 1)

class NodeHashEntry:
    def __init__(self):
        self.hash = 0     # ゾブリストハッシュの値
        self.color = 0    # 手番
        self.moves = 0    # ゲーム開始からの手数
        self.id = NOT_USE # 使用中のエージェントのID

class NodeHash:
    def __init__(self, hash_size):
        self.used = 0
        self.enough_size = True
        self.node_hash = None
        self.node_hash = [NodeHashEntry()] * hash_size

    # 配列の添え字でノードを取得する
    def __getitem__(self, i):
        return self.node_hash[i]

    # ハッシュのクリア
    def clear_hash(self, id):
        self.enough_size = True

        for node in self.node_hash:
            if node.id == id:
                node.id = NOT_USE
                used -= 1

    # 未使用のインデックスを探して返す
    def search_empty_index(self, hash, color, moves, id):
        key = hash_to_index(hash)
        i = key

        while True:
            if self.node_hash[i].id == NOT_USE:
                self.node_hash[i].hash = hash
                self.node_hash[i].color = color
                self.node_hash[i].moves = moves
                self.node_hash[i].id = id
                self.used += 1
                if self.used > UCT_HASH_LIMIT:
                    self.enough_size = False
                return i
            i += 1
            if i >= UCT_HASH_SIZE:
                i = 0
            if i == key:
                return UCT_HASH_SIZE

    # ハッシュ値に対応するインデックスを返す
    def find_same_hash_index(self, hash, color, moves, id):
        key = hash_to_index(hash)
        i = key

        while True:
            if self.node_hash[i].id == NOT_USE:
                return UCT_HASH_SIZE
            elif self.node_hash[i].hash == hash and self.node_hash[i].moves == moves and self.node_hash[i].id == id:
                return i
            i += 1
            if i >= UCT_HASH_SIZE:
                i = 0
            if i == key:
                return UCT_HASH_SIZE

class UctNode:
    def __init__(self):
        self.move_count = 0          # ノードの訪問回数
        self.win = 0.0               # 勝率の合計
        self.child_num = 0           # 子ノードの数
        self.child_move = None       # 子ノードの指し手
        self.child_index = None      # 子ノードのインデックス
        self.child_move_count = None # 子ノードの訪問回数
        self.child_win = None        # 子ノードの勝率の合計
        self.nnrate = None           # 方策ネットワークの予測確率
        self.value_win = 0.0         # 価値ネットワークの予測勝率
        self.evaled = False          # 評価済みフラグ
