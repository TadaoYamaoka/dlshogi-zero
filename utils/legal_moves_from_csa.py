import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cshogi import *
import sys
import glob

legal_moves = []

def read_kifu(kifu_list):
    positions = []
    parser = Parser()
    for filepath in kifu_list:
        parser.parse_csa_file(filepath)
        board = Board()
        for move in parser.moves:
            legal_moves.append(len(board.leagal_move_list()))
            board.push(move)

kifu_list = glob.glob(sys.argv[1] + r"\*")

read_kifu(kifu_list)

plt.hist(legal_moves)
plt.show()

print(pd.Series(legal_moves).describe())
print(np.percentile(legal_moves, 99))
print(np.percentile(legal_moves, 99.9))