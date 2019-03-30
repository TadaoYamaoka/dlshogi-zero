from dlshogi_zero.agent.selfplay_agent import SelfPlayAgentGroup
from dlshogi_zero.train import compile, train
from dlshogi_zero import database
from tensorflow.keras.models import load_model
from threading import Thread
import logging
import os
import re

def run(training_database_path, test_database_path, model_path, agents, checkpoint, playouts, limit_cycles, batchsize, lr, steps, test_steps, window_size, weight_decay):
    # モデルパスからモデルバージョン取得
    m = re.search('^(.*?)(\d+)(\..*)$', model_path)
    if m is None:
        raise RuntimeError('the file name must include the version number suffix(e.g. model-001.h5)')
    model_ver = int(m.group(2))

    # モデル
    model = load_model(model_path)
    compile(model, lr, weight_decay)

    # データベース
    training_database = database.TrainingDataBase(training_database_path)
    training_database.set_model_ver(model_ver)
    test_database = database.TrainingDataBase(test_database_path)

    # エージェント
    agent_group = SelfPlayAgentGroup(model, training_database, agents, checkpoint, playouts)

    cycles = 0
    while True:
        # 自己対局開始
        ret = agent_group.selfplay()

        if ret == False:
            break

        # 訓練
        train(training_database, test_database, model, batchsize, steps, test_steps, window_size)

        cycles += 1

        # 終了判定
        if limit_cycles is not None and cycles >= limit_cycles:
            break

        # モデル保存
        model_ver += 1
        model.save('{}{:03}{}'.format(m.group(1), model_ver, m.group(3)))

        # モデルバージョン設定
        training_database.set_model_ver(model_ver)

    training_database.close()
    test_database.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('training_database', default='selfplay.db')
    parser.add_argument('test_database')
    parser.add_argument('model', default='model-000.h5')
    parser.add_argument('--agents', type=int, default=64)
    parser.add_argument('--checkpoint', type=int, default=1000)
    parser.add_argument('--playouts', type=int, default=800)
    parser.add_argument('--limit_cycles', type=int)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=int, default=1e-4)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--test_steps', type=int, default=100)
    parser.add_argument('--window_size', type=int, default=10000)
    parser.add_argument('--log')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', filename=args.log, level=logging.DEBUG if args.debug else logging.INFO)

    logging.info('agents : {}'.format(args.agents))
    logging.info('checkpoint : {}'.format(args.checkpoint))
    logging.info('playouts : {}'.format(args.playouts))
    logging.info('batchsize : {}'.format(args.batchsize))
    logging.info('lr : {}'.format(args.lr))
    logging.info('steps : {}'.format(args.steps))
    logging.info('test_steps : {}'.format(args.test_steps))
    logging.info('window_size : {}'.format(args.window_size))
    logging.info('weight_decay : {}'.format(args.weight_decay))

    run(args.training_database,
        args.test_database,
        args.model,
        args.agents,
        args.checkpoint,
        args.playouts,
        args.limit_cycles,
        args.batchsize,
        args.lr,
        args.steps,
        args.test_steps,
        args.window_size,
        args.weight_decay
        )