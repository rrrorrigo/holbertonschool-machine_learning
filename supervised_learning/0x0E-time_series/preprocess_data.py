#!/usr/bin/env python3
"""Preprocess data"""


import pandas as pd
import numpy as np
import math


def clean_data(data=pd.DataFrame()):
    """Function that clean data

    data: DataFrame that contain transaction

    Return: Same DataFrame without unnecesary data"""
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
    data = data.set_index('Timestamp')
    # format the dataset to one row per hour
    data = data.resample("H").agg({
        "Open": "mean",
        "High": "mean",
        "Close": "mean",
        "Low": "mean",
        "Volume_(BTC)": "sum",
        "Volume_(Currency)": "sum",
        "Weighted_Price": "mean"
    })
    data = data.drop(['Low', 'High', 'Volume_(BTC)', 'Weighted_Price'],
                     axis=1).dropna()
    data = data.reindex(columns=['Open', 'Close', 'Volume_(Currency)'])
    print(data.shape)
    return data.to_numpy()


def save_data(data=np.ndarray):
    """Function that saves preprocessed data

    data: Data preprocessed to save"""
    df = pd.DataFrame(data, columns=['Open',
                                     'High',
                                     'Volume_(BTC)',
                                     'Volume_(Currency)'])
    df.to_csv('data/preprocessed_data.csv')


def X_Y_data(data=np.ndarray, steps=24):
    """Function that generate input and output from data"""
    X, Y = [], []
    for i in range(data.shape[0]):
        if (i + steps) >= data.shape[0]:
            break
        # Divide data between data (input) and labels (output)
        seq_X, seq_Y = data[i: i + steps], data[i + steps, -2]
        X.append(seq_X)
        Y.append(seq_Y)
    X, Y = np.array(X), np.array(Y)
    print(X.shape, Y.shape)
    return X, Y


def preprocess_data(data=np.ndarray):
    """Function that preprocess data before training

    data: np.ndarray that represent dataframe

    Return: a tuple that represent preprocessed data
        train_data: 70% of preprocessed data
        validate_data: 20% of preprocessed data
        test_data: 10% of preprocessed data"""
    # 70% for train data, 30% for validation data
    data_len = data.__len__()
    train_idx = math.ceil(data_len * 0.7)
    val_idx = int(data_len * 0.2) + train_idx

    data = (data - data.min()) / (data.max() - data.min())

    X_train_data = data[:train_idx]
    X_validate_data = data[train_idx:val_idx]
    X_test_data = data[val_idx:]

    X_train_data, Y_train_data = X_Y_data(X_train_data, 24)
    X_validate_data, Y_validate_data = X_Y_data(X_validate_data, 24)
    X_test_data, Y_test_data = X_Y_data(X_test_data, 24)

    return X_train_data, X_validate_data, X_test_data,\
           Y_train_data, Y_validate_data, Y_test_data
