import shogi

UCT_HASH_SIZE = 4096 # 2のn乗であること
UCT_HASH_LIMIT = UCT_HASH_SIZE * 9 / 10

# 未展開のノードのインデックス
NOT_EXPANDED = -1

def hash_to_index(hash):
    return ((hash & 0xffffffff) ^ ((hash >> 32) & 0xffffffff)) & (UCT_HASH_SIZE - 1)

class NodeHashEntry:
    def __init__(self):
        self.hash = 0     # ゾブリストハッシュの値
        self.color = 0    # 手番
        self.moves = 0    # ゲーム開始からの手数
        self.flag = False # 使用中か識別するフラグ

class NodeHash:
    def __init__(self):
        self.used = 0
        self.enough_size = True
        self.node_hash = None

    # UCTノードのハッシュの初期化
    def initialize(self):
        self.used = 0
        self.enough_size = True

        if self.node_hash is None:
            self.node_hash = [NodeHashEntry() for _ in range(UCT_HASH_SIZE)]
        else:
            for i in range(UCT_HASH_SIZE):
                self.node_hash[i].hash = 0
                self.node_hash[i].color = 0
                self.node_hash[i].moves = 0
                self.node_hash[i].flag = False


    # 配列の添え字でノードを取得する
    def __getitem__(self, i):
        return self.node_hash[i]

    # 未使用のインデックスを探して返す
    def search_empty_index(self, hash, color, moves):
        key = hash_to_index(hash)
        i = key

        while True:
            if not self.node_hash[i].flag:
                self.node_hash[i].hash = hash
                self.node_hash[i].color = color
                self.node_hash[i].moves = moves
                self.node_hash[i].flag = True
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
    def find_same_hash_index(self, hash, moves):
        key = hash_to_index(hash)
        i = key

        while True:
            if not self.node_hash[i].flag:
                return UCT_HASH_SIZE
            elif self.node_hash[i].hash == hash and self.node_hash[i].moves == moves:
                return i
            i += 1
            if i >= UCT_HASH_SIZE:
                i = 0
            if i == key:
                return UCT_HASH_SIZE

    # 古いハッシュを削除
    def delete_old_hash(self, moves):
        self.used = 0
        self.enough_size = True

        for i in range(UCT_HASH_SIZE):
            if self.node_hash[i].flag and self.node_hash[i].moves < moves:
                self.node_hash[i].flag = False
            else:
                self.used += 1

    def get_usage_rate(self):
        return self.used / UCT_HASH_SIZE
