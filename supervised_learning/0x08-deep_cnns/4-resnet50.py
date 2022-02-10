#!/usr/bin/env python3
"""ResNet-50"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Function that builds the ResNet-50 architecture

    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the blocks should be followed by batch
    normalization along the channels axis and a rectified linear activation
    (ReLU), respectively.
    All weights should use he normal initialization
    You may use:
        identity_block = __import__('2-identity_block').identity_block
        projection_block = __import__('3-projection_block').projection_block

    Returns: the keras model
    """

    init = K.initializers.HeNormal()
    input_layer = K.Input(shape=(224, 224, 3))

    conv1 = K.layers.Conv2D(64, 7, strides=2,
                            padding='same',
                            kernel_initializer=init)(input_layer)
    batch_norm = K.layers.BatchNormalization()(conv1)

    activation = K.layers.Activation('relu')(batch_norm)

    max_pooling = K.layers.MaxPooling2D(pool_size=(3, 3), strides=2,
                                        padding='same')(activation)

    conv2_proj1 = projection_block(max_pooling, [64, 64, 256], s=1)
    con2_inde2 = identity_block(conv2_proj1, [64, 64, 256])
    conv2_inde3 = identity_block(con2_inde2, [64, 64, 256])

    conv3_proj1 = projection_block(conv2_inde3, [128, 128, 512])

    con3_inde2 = identity_block(conv3_proj1, [128, 128, 512])
    con3_inde3 = identity_block(con3_inde2, [128, 128, 512])
    con3_inde4 = identity_block(con3_inde3, [128, 128, 512])

    conv4_proj1 = projection_block(con3_inde4, [256, 256, 1024])
    con4_inde2 = identity_block(conv4_proj1, [256, 256, 1024])
    con4_inde3 = identity_block(con4_inde2, [256, 256, 1024])
    con4_inde4 = identity_block(con4_inde3, [256, 256, 1024])
    con4_inde5 = identity_block(con4_inde4, [256, 256, 1024])
    con4_inde6 = identity_block(con4_inde5, [256, 256, 1024])

    conv5_proj1 = projection_block(con4_inde6, [512, 512, 2048])
    con5_inde2 = identity_block(conv5_proj1, [512, 512, 2048])
    con5_inde3 = identity_block(con5_inde2, [512, 512, 2048])

    avg_pooling = K.layers.AveragePooling2D(pool_size=(7, 7)
                                            )(con5_inde3)

    output_layer = K.layers.Dense(1000,
                                  kernel_initializer=init,
                                  activation='softmax')(avg_pooling)

    model = K.Model(input_layer, output_layer)

    return model
