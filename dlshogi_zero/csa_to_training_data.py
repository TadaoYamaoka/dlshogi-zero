import numpy as np
from cshogi import *
from dlshogi_zero.database import *
from dlshogi_zero.encoder import *
import argparse
import glob
import os.path
import math
import logging

parser = argparse.ArgumentParser()
parser.add_argument('csa_dir', help='directory stored CSA file')
parser.add_argument('training_database', help='training database file')
parser.add_argument('--filter_moves', type=int, default=50, help='filter by move count')
parser.add_argument('--clear', action='store_true', help='clear database before processing')

args = parser.parse_args()

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

MAX_MOVE_COUNT = 256

# process csa
def process_csa(database, csa_file_list):
    board = Board()
    parser = Parser()
    for filepath in csa_file_list:
        parser.parse_csa_file(filepath.encode('utf-8'))
        if parser.endgame not in (b'%TORYO', b'%SENNICHITE', b'%KACHI', b'%HIKIWAKE') or len(parser.moves) < args.filter_moves:
            continue
        board.set_sfen(parser.sfen)
        chunk = []
        game_hcps = np.empty(MAX_MOVE_COUNT, dtype=dtypeHcp)
        # totalMoveCount
        total_move_count = len(parser.moves)
        # gameResult
        game_result = parser.win
        for i, move in enumerate(parser.moves):
            # hcp
            board.to_hcp(game_hcps[i])
            hcps = np.zeros(MAX_HISTORY, dtype=dtypeHcp)
            if i < MAX_HISTORY:
                hist = i + 1
            else:
                hist = MAX_HISTORY
            for prev in range(hist):
                hcps[prev] = game_hcps[i - prev]
            # repetition
            repetition = 1 if board.is_draw() == REPETITION_DRAW else 0
            # legalMoves
            legal_moves = np.empty(1, dtypeMove)
            legal_moves[0] = move16(move) # 指し手のみ
            # visits
            visits = np.empty(1, dtypeVisit)
            visits[0] = 1 # 指し手のみ

            board.push(move)

            chunk.append((hcps.data, repetition, total_move_count, legal_moves.data, visits.data, game_result))

        # write data
        database.write_chunk(chunk)

csa_file_list = glob.glob(os.path.join(args.csa_dir, '*'))

training_database = TrainingDataBase(args.training_database, args.clear)
training_database.set_model_ver(0)
logging.info('start process csa')
process_csa(training_database, csa_file_list)
training_database.commit()
training_database.close()
logging.info('done')