import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cshogi import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('hcpe')
args = parser.parse_args()


hcpes = np.fromfile(args.hcpe, dtype=HuffmanCodedPosAndEval)
legal_moves = np.zeros(len(hcpes), dtype=np.uint16)

board = Board()

for i, hcpe in enumerate(hcpes):
    board.set_hcp(hcpe['hcp'])
    legal_moves[i] = len(board.leagal_move_list())

plt.hist(legal_moves)
plt.show()

print(pd.Series(legal_moves).describe())
print(np.percentile(legal_moves, 99))
print(np.percentile(legal_moves, 99.9))