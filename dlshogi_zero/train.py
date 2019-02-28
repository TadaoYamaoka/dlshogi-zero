import tensorflow as tf
import numpy as np
from cshogi import *
from dlshogi_zero.nn.resnet import ResNet
from dlshogi_zero.encoder import *
from dlshogi_zero.database import *
import os

def mini_batch(database, window_size, batch_size):
    board = Board()
    features = np.zeros((batch_size, MAX_FEATURES, 81), dtype=np.float32)
    action_probabilities = np.zeros((batch_size, MAX_ACTION_LABELS), dtype=np.float32)
    game_outcomes = np.empty(batch_size, dtype=np.float32)

    for i, (hcprs, total_move_count, legal_moves, visits, game_result) in enumerate(database.get_training_batch(window_size=window_size, batch_size=batch_size)):
        # input features
        for j, hcpr in enumerate(hcprs):
            board.set_hcp(hcpr['hcp'])
            encode_position(board, hcpr['repetition'], features[i], j)
            if j == 0:
                color = board.turn
        encode_color_totalmovecout(color, total_move_count, features[i])

        # action probabilities
        sum = np.sum(visits)
        for move, visit in zip(legal_moves, visits):
            action_probabilities[i][encode_action(move)] = visit / sum

        # game outcome
        game_outcomes[i] = encode_outcome(color, game_result)

    return (features.reshape((batch_size, MAX_FEATURES, 9, 9)), { 'policy': action_probabilities, 'value': game_outcomes })

def datagen(database_path, window_size, batchsize):
    training_database = TrainingDataBase(database_path)
    while True:
        yield mini_batch(training_database, window_size, batchsize)

def categorical_crossentropy(y_true, y_pred):
    return tf.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

def categorical_accuracy(y_true, y_pred):
    return tf.keras.metrics.categorical_accuracy(y_true, tf.nn.softmax(y_pred))

def binary_accuracy(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(tf.keras.backend.round((y_true + 1) / 2), y_pred, threshold=0)

def train(training_database_path, test_database_path, model_path, batchsize, steps, test_steps, window_size, weight_decay, use_tpu):
    model = ResNet(res_blocks=10, filters=192)

    # add weight decay
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Dense):
            layer.add_loss(tf.keras.regularizers.l2(weight_decay)(layer.kernel))

    model.compile(optimizer=tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9),
                  loss={'policy': categorical_crossentropy, 'value': 'mse'},
                  metrics={'policy': categorical_accuracy, 'value': binary_accuracy})

    # TPU
    if use_tpu:
        model = tf.contrib.tpu.keras_to_tpu_model(
            model,
            strategy=tf.contrib.tpu.TPUDistributionStrategy(
                tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
            )
        )

    model.fit_generator(datagen(training_database_path, window_size, batchsize), steps,
                        validation_data=datagen(test_database_path, window_size, batchsize), validation_steps=test_steps)

    model.save(model_path)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('training_database')
    parser.add_argument('test_database')
    parser.add_argument('model', default='model.h5')
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--test_steps', type=int, default=100)
    parser.add_argument('--window_size', type=int, default=1000000)
    parser.add_argument('--weight_decay', type=int, default=1e-4)
    parser.add_argument('--use_tpu', action='store_true')

    args = parser.parse_args()

    train(args.training_database,
          args.test_database,
          args.model,
          args.batchsize,
          args.steps,
          args.test_steps,
          args.window_size,
          args.weight_decay,
          args.use_tpu
          )