import numpy as np
from cshogi import *
from dlshogi_zero.database import *
from dlshogi_zero.features import *
import glob
import os.path
import math
from collections import defaultdict

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
        repetitions = defaultdict(int)
        game_hcprs = np.empty(MAX_MOVE_COUNT, dtype=HcpAndRepetition)
        # gameResult
        game_result = parser.win
        for i, move in enumerate(parser.moves):
            # hcp
            board.to_hcp(game_hcprs[i]['hcp'])
            # repetition
            if board.is_draw() == REPETITION_DRAW:
                repetitions[board.zobrist_hash()] += 1
            game_hcprs[i]['repetition'] = repetitions[board.zobrist_hash()]
            # legalMoves
            legal_moves = np.empty(1, dtypeMove16)
            legal_moves[0] = move16(move) # 指し手のみ
            # totalMoveCount
            total_move_count = board.move_number
            # visits
            visits = np.empty(1, dtypeVisit)
            visits[0] = 1 # 指し手のみ

            # history
            hcprs = np.zeros(MAX_HISTORY, dtype=HcpAndRepetition)
            if i < MAX_HISTORY:
                hist = i + 1
            else:
                hist = MAX_HISTORY
            for prev in range(hist):
                hcprs[prev] = game_hcprs[i - prev]

            board.push(move)

            chunk.append((hcprs.data, total_move_count, legal_moves.data, visits.data, game_result))

        # write data
        database.write_chunk(chunk)

if __name__ == '__main__':
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    parser.add_argument('csa_dir', help='directory stored CSA file')
    parser.add_argument('training_database', help='training database file')
    parser.add_argument('--filter_moves', type=int, default=50, help='filter by move count')
    parser.add_argument('--clear', action='store_true', help='clear database before processing')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

    csa_file_list = glob.glob(os.path.join(args.csa_dir, '*'))

    training_database = TrainingDataBase(args.training_database, args.clear)
    training_database.set_model_ver(0)
    logging.info('start process csa')
    process_csa(training_database, csa_file_list)
    training_database.close()
    logging.info('done')
