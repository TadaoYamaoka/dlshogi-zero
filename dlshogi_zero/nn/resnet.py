import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, Dense, Conv2D, BatchNormalization, Activation, Flatten, Add
from dlshogi_zero.features import MAX_FEATURES, MAX_ACTION_PLANES

RES_BLOCKS = 5
FILTERS = 192
FCL_UNITS = 192

def conv_layer(inputs,
               filters,
               kernel_size=3,
               activation='relu',
               use_bias=True):

    x =  Conv2D(filters,
                kernel_size=kernel_size,
                padding='same',
                data_format='channels_first',
                kernel_initializer='he_normal',
                use_bias=use_bias)(inputs)
    x = BatchNormalization(axis=1)(x)
    if activation is not None:
        x = Activation(activation)(x)
    return x

def ResNet(input_planes=MAX_FEATURES,
           res_blocks=RES_BLOCKS,
           filters=FILTERS,
           fcl_units=FCL_UNITS,
           policy_planes=MAX_ACTION_PLANES):

    inputs = Input(shape=(input_planes, 9, 9))
    x = conv_layer(inputs, filters=filters, use_bias=False)

    for res_block in range(res_blocks):
        # bottleneck residual unit
        y = conv_layer(x, filters=filters, use_bias=False)
        y = conv_layer(y, filters=filters, use_bias=False, activation=None)
        x = Add()([x, y])
        x = Activation('relu')(x)

    # Add policy head
    policy_x = conv_layer(x, filters=filters)
    policy_x = Conv2D(policy_planes,
                      kernel_size=3,
                      padding='same',
                      data_format='channels_first',
                      kernel_initializer='he_normal')(policy_x)
    policy_y = Flatten(name='policy')(policy_x)

    # Add value head
    value_x = conv_layer(x, filters=1, kernel_size=1)
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
