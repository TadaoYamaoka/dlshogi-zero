from dlshogi_zero.agent import selfplay_agent
from dlshogi_zero.agent.selfplay_agent import *
from threading import Thread
import logging
import os

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG if __debug__ else logging.INFO)

def run(record_filepath, model_path, batch_size):
    init_database(record_filepath)

    agent_group = SelfPlayAgentGroup(model_path, batch_size)

    # 自己対局開始
    agent_group.selfplay()

    term_database()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('database', default='record.db')
    parser.add_argument('model', default='model.h5')
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--limit_games', type=int, default=10000)
    parser.add_argument('--num_playouts', type=int, default=800)

    args = parser.parse_args()

    logging.info('batchsize : {}'.format(args.batchsize))
    logging.info('limit_games : {}'.format(args.limit_games))
    logging.info('num_playouts : {}'.format(args.num_playouts))

    selfplay_agent.limit_games = args.limit_games
    selfplay_agent.num_playouts = args.num_playouts

    run(args.database,
        args.model,
        args.batchsize
        )