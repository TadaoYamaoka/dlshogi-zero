import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, BatchNormalization, Activation, Flatten, Add
from dlshogi_zero.encoder import MAX_FEATURES

class Bias(Layer):
    def __init__(self, **kwargs):
        super(Bias, self).__init__(**kwargs)

    def build(self, input_shape):
        self.b = self.add_weight(name='b',
                                 shape=(input_shape[1:]),
                                 initializer='zeros',
                                 trainable=True)
        super(Bias, self).build(input_shape)

    def call(self, x):
        return x + self.b

def conv_layer(inputs,
               filters,
               activation='relu',
               use_bias=True):

    x =  Conv2D(filters,
                kernel_size=3,
                strides=1,
                padding='same',
                data_format='channels_first',
                kernel_initializer='he_normal',
                use_bias=use_bias)(inputs)
    x = BatchNormalization(axis=1)(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x

def ResNet(input_planes=MAX_FEATURES,
           res_blocks=20,
           filters=256,
           fcl_units=256,
           policy_planes=139):

    inputs = Input(shape=(input_planes, 9, 9))
    x = conv_layer(inputs, filters=filters, use_bias=False)

    for res_block in range(res_blocks):
        # bottleneck residual unit
        y = conv_layer(x, filters=filters, use_bias=False)
        y = conv_layer(y, filters=filters, use_bias=False, activation=None)
        x = Add()([x, y])
        x = Activation('relu')(x)

    # Add policy output
    policy_x = Conv2D(policy_planes,
                      kernel_size=1,
                      strides=1,
                      padding='same',
                      data_format='channels_first',
                      kernel_initializer='he_normal',
                      use_bias=False)(x)
    policy_x = Flatten()(policy_x)
    policy_y = Bias(name='policy')(policy_x)

    # Add value output
    value_x = conv_layer(x, filters=1)
    value_x = Flatten()(value_x)
    value_x = Dense(fcl_units,
                    activation='relu',
                    kernel_initializer='he_normal')(value_x)
    value_y = Dense(1,
                    activation='tanh',
                    kernel_initializer='he_normal',
                    name='value')(value_x)

    # Instantiate model
    return Model(inputs=inputs, outputs=[policy_y, value_y])
