import numpy as np

HISTORY = 8

TrainingData = np.dtype([
    ('hcps', ((np.uint8, 32), HISTORY)), # 局面(現局面と履歴局面)
    ('repetition', np.uint8),            # 繰り返し数
    ('totalMoveCount', np.uint16),       # 手数
    ('moves', np.uint16, 256),           # 合法手(上限256手)
    ('visits', np.uint16, 256),          # 合法手に対応した訪問数
    ('value', np.float32),               # 推定勝率
    ('gameResult', np.uint8),            # ゲーム結果
    ])