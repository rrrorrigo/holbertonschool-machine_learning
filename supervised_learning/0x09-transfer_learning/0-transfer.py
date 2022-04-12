#!/usr/bin/env python3
"""Transfer learning"""


import tensorflow.keras as K


def preprocess_data(X, Y):
    """Function that pre-processes the data for your model"""
    X_p = K.applications.resnet50.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)
    return X_p, Y_p

if __name__ == '__main__':
    # import dataset
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    (x_train, y_train) = preprocess_data(x_train, y_train)
    (x_test, y_test) = preprocess_data(x_test, y_test)

    model = K.models.Sequential()
    # import resnet50 model and freeze its weights
    rn50 = K.applications.resnet50.ResNet50(include_top=False,
                                            weights='imagenet')
    for layer in rn50.layers:
        layer.trainable = False

    resize = K.layers.Lambda(lambda image: K.backend.resize_images(image, 7, 7,
                             data_format="channels_last",
                             interpolation='bilinear'))

    # implementation of resnet50 to my new model
    model.add(resize)
    model.add(rn50)
    model.add(K.layers.Flatten())
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(256, activation='relu',
                             kernel_initializer='he_uniform'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(64, activation='relu'))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(10, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy', optimizer=K.optimizers.Adam(lr=0.001),
        metrics=['accuracy'])

    # train the model and for each epoch show validation accuracy
    model.fit(
        x_train, y_train, batch_size=50, epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[K.callbacks.ModelCheckpoint(filepath='cifar10.h5',
                   save_best_only=True)]
    )
