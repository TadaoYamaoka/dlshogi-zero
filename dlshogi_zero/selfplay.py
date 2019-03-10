from dlshogi_zero.agent import selfplay_agent
from dlshogi_zero.agent.selfplay_agent import *
from threading import Thread
import time
import logging
import os

def progress_func(agent_group_pair):
    start_time = time.time()

    while agent_group_pair.running() != 0:
        if os.path.exists('stop'):
            selfplay_agent.stopflg = True
            break

        time.sleep(10)

        ply_per_game = selfplay_agent.written_nodes / selfplay_agent.games if selfplay_agent.games > 0 else 150
        limit_nodes = ply_per_game * selfplay_agent.limit_games
        progress = selfplay_agent.written_nodes / limit_nodes
        elapsed_time = time.time() - start_time
        nodes_per_sec = selfplay_agent.nodes / elapsed_time

        logging.info("progress:{:.2f}%, nodes:{}, nodes/sec:{:.2f}, games:{}, ply/game:{:.2f}, elapsed:{}[s]".format(
            progress, selfplay_agent.nodes, nodes_per_sec, selfplay_agent.games, ply_per_game, elapsed_time
            ))

def run(record_filepath, model_path, batch_size):
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

    init_database(record_filepath)

    agent_group_pair = SelfPlayAgentGroupPair(model_path, batch_size)

    # 探索スレッド開始
    agent_group_pair.run()

    # 進捗状況表示
    th_progress = Thread(target=progress_func, args=(agent_group_pair,))
    th_progress.start()

    # 探索スレッド終了待機
    agent_group_pair.join()
    th_progress.join()

    term_database()

    # 結果表示
    logging.info("made {} games. nodes:{}, ply/game:{:.2f}".format(
        selfplay_agent.games, selfplay_agent.nodes, selfplay_agent.written_nodes / selfplay_agent.games
        ))

if __name__ == '__main__':
    import argparse
    import signal

    parser = argparse.ArgumentParser()
    parser.add_argument('database', default='record.db')
    parser.add_argument('model', default='model.h5')
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--limit_games', type=int, default=10000)

    args = parser.parse_args()

    selfplay_agent.limit_games = args.limit_games

    run(args.database,
        args.model,
        args.batchsize
        )