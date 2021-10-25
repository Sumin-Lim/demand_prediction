'''
Author: Sumin Lim @ KAIST
Date: 2021-10-25
Usage: python base.py
Description:
    * Train baseline models - CNN, LSTM, ConvLSTM
    * TODO: add linear regression, XGBOOST, prophet, ARIMA
'''
from tqdm import tqdm
from datetime import datetime
from typing import Tuple
import pickle as pkl
import numpy as np
import pandas as pd
import h3

import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow import keras
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from load_data import main as main_data
from keras.activations import sigmoid

def smape(a, b):
    a = np.reshape(a, (-1, ))
    b = np.reshape(b, (-1, ))
    return np.mean(2.0*np.abs(a-b) / (np.abs(a)+np.abs(b)+1)).item()*100

def get_data(sliding=True):
    sliding_dates_week, tensor, _ = main_data()
    n_weeks_sliding_train = (len(sliding_dates_week) * 2) // 3
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for idx in range(len(sliding_dates_week)):
        week = sliding_dates_week[idx]
        if sliding:
            X_data = np.array([np.expand_dims(tensor[x],axis=-1) for x in week[:6]])
            y_data = np.array([np.expand_dims(tensor[week[-1]],axis=-1)])
            if idx < n_weeks_sliding_train:
                X_train.append(X_data)
                y_train.append(y_data)
            else:
                X_test.append(X_data)
                y_test.append(y_data)
        else:
            return np.array(list(tensor.values()))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_train = np.swapaxes(y_train, 1, -1)
    y_test = np.swapaxes(y_test, 1, -1)

    return X_train, X_test, y_train, y_test

def cnn_model(rows: int, cols: int, x_shape: Tuple[int]):
    inp = layers.Input(shape=x_shape[1:])
    x = layers.Conv3D(
            filters=64,
            kernel_size=3,
            activation='relu',
            padding='same')(inp)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(
            filters=32,
            kernel_size=3,
            activation='relu',
            padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(
            filters=1,
            kernel_size=3,
            activation='relu',
            padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(rows*cols, activation='relu')(x)
    out = layers.Reshape(target_shape=(1, rows, cols, 1))(x)

    model = keras.Model(inp, out)
    return model

def lstm_model(rows: int, cols:int, x_shape: Tuple[int]):
    inp = layers.Input(shape=x_shape[1:])
    x = layers.Reshape(target_shape=(6, x_shape[2]*x_shape[3]*x_shape[4]))(inp)

    x = layers.LSTM(
            units=32,
            return_sequences=True
            )(x)
    x = layers.BatchNormalization()(x)

    x = layers.LSTM(
            units=32,
            return_sequences=True
            )(x)
    x = layers.BatchNormalization()(x)

    x = layers.LSTM(
            units=32,
            return_sequences=True
            )(x)
    x = layers.BatchNormalization()(x)

    x = layers.Flatten()(x)
    x = layers.Dense(rows*cols, activation='relu')(x)
    out = layers.Reshape(target_shape=(1, rows, cols, 1))(x)

    model = keras.Model(inp, out)
    return model

def convlstm_model(rows: int, cols: int, x_shape: Tuple[int]):
    inp = layers.Input(shape=x_shape[1:])
    x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='relu')(inp)
    x = layers.BatchNormalization()(x)

    x = layers.ConvLSTM2D(
            filters=32,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='relu')(inp)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(
            filters=1,
            kernel_size=(3, 3, 3),
            padding='same',
            activation='sigmoid')(inp)

    x = layers.Flatten()(x)
    x = layers.Dense(rows*cols, activation='relu')(x)
    out = layers.Reshape(target_shape=(1, rows, cols, 1))(x)

    model = keras.Model(inp, out)
    return model

def main(X_train, X_test, Y_train, Y_test):
    rmse = tf.keras.metrics.RootMeanSquaredError()

    allzeros = np.zeros(Y_test.shape)
    print('baseline allzero rmse:', rmse(Y_test, allzeros).numpy())
    print('baseline allzero smape:', smape(Y_test, allzeros))

    es = EarlyStopping(monitor='val_loss', mode='min', patience=20)

    cnn = cnn_model(55, 36, X_train.shape)
    cnn.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())
    cnn.fit(
            X_train,
            Y_train,
            epochs=500,
            callbacks=[es],
            validation_split=0.2)
    predict_cnn = cnn.predict(X_test)
    smape_cnn = smape(Y_test, predict_cnn)

    lstm = lstm_model(55, 36, X_train.shape)
    lstm.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())
    lstm.fit(
            X_train,
            Y_train,
            epochs=500,
            callbacks=[es],
            validation_split=0.2)
    predict_lstm = lstm.predict(X_test)
    smape_lstm = smape(Y_test, predict_lstm)

    convlstm = convlstm_model(55, 36, X_train.shape)
    convlstm.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())
    convlstm.fit(
            X_train,
            Y_train,
            epochs=500,
            callbacks=[es],
            validation_split=0.2)
    predict_convlstm = convlstm.predict(X_test)
    smape_convlstm = smape(Y_test, predict_convlstm)

    res = {}
    res['smape_allzero'] = smape(Y_test, allzeros)
    res['smape_cnn'] = smape_cnn
    res['smape_lstm'] = smape_lstm
    res['smape_convlstm'] = smape_convlstm
    res['rmse_allzero'] = rmse(Y_test, allzeros).numpy()
    res['rmse_cnn'] = rmse(Y_test, predict_cnn).numpy()
    res['rmse_lstm'] = rmse(Y_test, predict_lstm).numpy()
    res['rmse_convlstm'] = rmse(Y_test, predict_convlstm).numpy()

    print()
    print('=============================================================')
    print('cnn smape:', smape_cnn)
    print('cnn rmse:', rmse(Y_test, predict_cnn).numpy())
    print('=============================================================')
    print()
    print('=============================================================')
    print('lstm smape:', smape_lstm)
    print('lstm rmse:', rmse(Y_test, predict_lstm).numpy())
    print('=============================================================')
    print()
    print('=============================================================')
    print('convlstm smape:', smape_convlstm)
    print('convlstm rmse:', rmse(Y_test, predict_convlstm).numpy())
    print('=============================================================')
    return res

if __name__ == '__main__':

    X_train, X_test, Y_train, Y_test = get_data()

    result = []
    for _ in tqdm(range(10)):
        result.append(main(X_train, X_test, Y_train, Y_test))

    df_res = pd.DataFrame(result)
    df_res.to_csv('result/result_base.csv')

