class UctNode:
    def __init__(self):
        self.move_count = 0          # ノードの訪問回数
        self.win = 0.0               # 勝率の合計
        self.child_num = 0           # 子ノードの数
        self.child_move = None       # 子ノードの指し手
        self.child_nodes = None      # 子ノード
        self.child_move_count = None # 子ノードの訪問回数
        self.child_win = None        # 子ノードの勝率の合計
        self.nnrate = None           # 方策ネットワークの予測確率
        self.value_win = 0.0         # 価値ネットワークの予測勝率
        self.evaled = False          # 評価済みフラグ
