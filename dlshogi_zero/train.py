import tensorflow as tf
from tensorflow.keras.backend import set_session
from tensorflow.keras.models import load_model
import numpy as np
from cshogi import *
from dlshogi_zero.nn.resnet import ResNet
from dlshogi_zero.features import *
from dlshogi_zero.database import *
import os

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

def mini_batch(database, batch_size):
    board = Board()
    features = np.zeros((batch_size, MAX_FEATURES, 81), dtype=np.float32)
    action_probabilities = np.zeros((batch_size, MAX_ACTION_LABELS), dtype=np.float32)
    game_outcomes = np.empty(batch_size, dtype=np.float32)

    for i, (hcprs, total_move_count, legal_moves, visits, game_result) in enumerate(database.get_training_batch(batch_size)):
        # input features
        for j, hcpr in enumerate(hcprs):
            board.set_hcp(hcpr['hcp'])
            make_position_features(board, hcpr['repetition'], features[i], j)
            if j == 0:
                color = board.turn
        make_color_totalmovecout_features(color, total_move_count, features[i])

        # action probabilities
        sum = np.sum(visits)
        for move, visit in zip(legal_moves, visits):
            action_probabilities[i][make_action_label(move)] = visit / sum

        # game outcome
        game_outcomes[i] = make_outcome(color, game_result)

    return (features.reshape((batch_size, MAX_FEATURES, 9, 9)), { 'policy': action_probabilities, 'value': game_outcomes })

def datagen(database, window_size, batchsize):
    database.prepare_training(window_size)
    while True:
        yield mini_batch(database, batchsize)

def categorical_crossentropy(y_true, y_pred):
    return tf.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

def categorical_accuracy(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true, tf.nn.softmax(y_pred))

def binary_accuracy(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(tf.keras.backend.round((y_true + 1) / 2), y_pred, threshold=0)

def compile(model, lr, weight_decay):

    # add weight decay
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(tf.keras.regularizers.l2(weight_decay)(layer.kernel))

    model.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9),
                  loss={'policy': categorical_crossentropy, 'value': 'mse'},
                  metrics={'policy': categorical_accuracy, 'value': binary_accuracy})

def train(training_database, test_database, model, batchsize, steps, test_steps, window_size):
    model.fit_generator(datagen(training_database, window_size, batchsize), steps,
                        validation_data=datagen(test_database, window_size, batchsize), validation_steps=test_steps)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('training_database')
    parser.add_argument('test_database')
    parser.add_argument('model', default='model.h5')
    parser.add_argument('--resume', '-r')
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--test_steps', type=int, default=1000)
    parser.add_argument('--window_size', type=int, default=1000000)
    parser.add_argument('--weight_decay', type=int, default=1e-4)

    args = parser.parse_args()

    if args.resume is not None:
        model = load_model(args.resume)
    else:
        model = ResNet()

    compile(model, args.lr, args.weight_decay)

    training_database = TrainingDataBase(args.training_database)
    test_database = TrainingDataBase(args.test_database)

    train(training_database,
          test_database,
          model,
          args.batchsize,
          args.steps,
          args.test_steps,
          args.window_size
          )

    model.save(args.model)
