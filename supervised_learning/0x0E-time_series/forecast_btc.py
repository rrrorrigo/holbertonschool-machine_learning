#!/usr/bin/env python3
"""Script that create, train and validate keras model"""


import keras.api._v2.keras as K
import tensorflow as tf
import matplotlib.pyplot as plt


class Forecasting:
    """Keras model that create, trains and validate"""

    def __init__(self, train_X, val_X, train_Y, val_Y, batch_size=256):
        """Class constructor"""
        self.train_X = train_X
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_X, train_Y)).shuffle(train_X.shape[0])\
            .batch(batch_size).repeat()
        self.val_dataset = tf.data.Dataset.from_tensor_slices(
            (val_X, val_Y)).batch(batch_size).repeat()
        self.batch_size = batch_size

    def create(self):
        """Function that creates Forecasting model"""
        model = K.models.Sequential()
        model.add(K.layers.Bidirectional(
                  K.layers.LSTM(64, activation='relu',
                                input_shape=self.train_X.shape[1:])))
        model.add(K.layers.Dense(1))
        model.compile(loss='mse',
                      optimizer='adam')
        return model

    def train(self, steps=800, epochs=10,
              model=K.models.Sequential(), val_steps=80):
        """Function that trains Forecasting model"""
        history = model.fit(self.train_dataset, epochs=epochs,
                            steps_per_epoch=steps,
                            validation_data=self.val_dataset,
                            validation_steps=val_steps,
                            )
        model.summary()
        return model, history

    def plot_0(self, df, title):
        """function that plots Price at Close vs. Timestamp"""

        plt.figure(figsize=(8, 6))
        plt.plot(df)
        plt.title(title)
        plt.xlabel('Timestamp')
        plt.ylabel('Price at Close (USD)')

        plt.show()

    def plot_1(self, history, title):
        """function that plots the loss results of the model"""

        plt.figure(figsize=(8, 6))
        plt.plot(history.history['loss'], 'o-', mfc='none',
                 markersize=10, label='Train')
        plt.plot(history.history['val_loss'], 'o-', mfc='none',
                 markersize=10, label='Valid')
        plt.title('LSTM Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

    def plot_2(self, data_24h, single_label, single_prediction, title):
        """function that plots a single-step price prediction following
        24h of data"""

        time_steps = list(range(24))
        next_step = 24

        plt.figure(figsize=(8, 6))
        plt.plot(time_steps, data_24h, 'o-', markersize=8, label='data_24h')
        plt.plot(next_step, single_label, 'b+', mfc='none',
                 markersize=12, label='Label')
        plt.plot(next_step, single_prediction, 'ro', mfc='none',
                 markersize=12, label='Prediction')
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Price at Close (Standardized Data)')
        plt.legend()

        plt.show

    def plot_3(self, future, prediction, title):
        """function that plots predictions over "batch_size" x 24h
        timeframes"""

        days = list(range(1, future.shape[0] + 1))
        plt.figure(figsize=(12, 6))
        plt.plot(days, future, 'o-', markersize=5, mfc='none', label='Labels')
        plt.plot(days, prediction, 'o-', markersize=5,
                 mfc='none', label='Predictions')
        plt.title(title)
        plt.xlim([days[0], days[-1]])
        plt.xlabel('24h Steps')
        plt.ylabel('Price at Close (Standardized Data)')
        plt.legend()

        plt.show
