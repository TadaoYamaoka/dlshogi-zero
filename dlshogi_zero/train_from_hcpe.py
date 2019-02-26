import tensorflow as tf
import numpy as np
from cshogi import *
from dlshogi_zero.nn.resnet import ResNet
from dlshogi_zero.encoder.shogi_encoder import *
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('train_hcpe')
parser.add_argument('test_hcpe')
parser.add_argument('--batchsize', type=int, default=256)
parser.add_argument('--epochs', default=1)
parser.add_argument('--use_tpu', action='store_true')

args = parser.parse_args()

train_hcpe_path = args.train_hcpe
test_hcpe_path = args.test_hcpe
batchsize = args.batchsize
epochs = args.epochs
weight_decay = 1e-4
use_tpu = args.use_tpu

model = ResNet(res_blocks=10, filters=192)

train_hcpes = np.fromfile(train_hcpe_path, dtype=HuffmanCodedPosAndEval)
test_hcpes = np.fromfile(test_hcpe_path, dtype=HuffmanCodedPosAndEval)

board = Board()
def mini_batch(hcpes):
    features = np.zeros((len(hcpes), 45, 81), dtype=np.float32)
    action_labels = np.empty(len(hcpes), dtype=np.int)
    game_outcomes = np.empty(len(hcpes), dtype=np.float32)

    for i, hcpe in enumerate(hcpes):
        # input features
        board.set_hcp(hcpe['hcp'])
        encode_position(board, features[i])
        encode_color_totalmovecout(board.turn, 0, features[i])

        # action label
        move = hcpe['bestMove16']
        action_labels[i] = encode_action(move)

        # game outcome
        game_result = hcpe['gameResult']
        game_outcomes[i] = encode_outcome(board.turn, game_result)

    return (features.reshape((len(hcpes), 45, 9, 9)), { 'policy': action_labels, 'value': game_outcomes })

def datagen(hcpes, batchsize):
    while True:
        np.random.shuffle(hcpes)
        for i in range(0, len(hcpes) - batchsize, batchsize):
            yield mini_batch(hcpes[i:i+batchsize])

def categorical_crossentropy(y_true, y_pred):
    return tf.keras.backend.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)

def categorical_accuracy(y_true, y_pred):
    return tf.keras.metrics.sparse_categorical_accuracy(y_true, tf.nn.softmax(y_pred))

def binary_accuracy(y_true, y_pred):
    return tf.keras.metrics.binary_accuracy(tf.keras.backend.round((y_true + 1) / 2), y_pred, threshold=0)

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

model.fit_generator(datagen(train_hcpes, batchsize), len(train_hcpes) // batchsize,
                    epochs=epochs,
                    validation_data=datagen(test_hcpes, batchsize), validation_steps=len(test_hcpes) // batchsize)
