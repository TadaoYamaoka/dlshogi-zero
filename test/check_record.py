from dlshogi_zero.database import *
from cshogi import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('database', default='record.db')
args = parser.parse_args()

db = TrainingDataBase(args.database)

records = db.cur.execute('SELECT hcprs, total_move_count, legal_moves, visits, game_result FROM training_data').fetchall()

board = Board()

for i, data in enumerate(records):
    (hcprs, total_move_count, legal_moves, visits, game_result) = (
                np.frombuffer(data[0], dtype=HcpAndRepetition),
                data[1],
                np.frombuffer(data[2], dtype=dtypeMove16),
                np.frombuffer(data[3], dtype=dtypeVisit),
                data[4]
                )

    index = np.argmax(visits)
    move = legal_moves[index]
    print(i, move_to_usi(move))
    for hcpr in hcprs:
        board.set_hcp(hcpr['hcp'])
        print(board.sfen())

