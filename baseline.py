'''
Author: Sumin Lim @ KAIST
Date: 2021-10-25
Usage: python base.py
Description:
    * Train baseline models - CNN, LSTM, ConvLSTM
    * TODO: add linear regression, XGBOOST, prophet, ARIMA
'''
from tqdm import tqdm
from time import time
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
from keras.activations import sigmoid

from load_data import main as main_data

def smape(a, b):
    a = np.reshape(a, (-1, ))
    b = np.reshape(b, (-1, ))
    return np.mean(2.0*np.abs(a-b) / (np.abs(a)+np.abs(b)+1)).item()*100

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

def train(X_train, X_test, Y_train, Y_test, interval, n_sliding, n_experiment):
    rmse = tf.keras.metrics.RootMeanSquaredError()

    allzeros = np.zeros(Y_test.shape)
    print('baseline allzero rmse:', rmse(Y_test, allzeros).numpy())
    print('baseline allzero smape:', smape(Y_test, allzeros))

    es = EarlyStopping(monitor='val_loss', mode='min', patience=20)

    cnn = cnn_model(55, 36, X_train.shape)
    cnn.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())
    time_cnn = time.time()
    cnn.fit(
            X_train,
            Y_train,
            epochs=500,
            callbacks=[es],
            validation_split=0.2)
    training_cnn = time.time() - time_cnn
    time_cnn = time.time()
    predict_cnn = cnn.predict(X_test)
    predicting_cnn = time.time() - time_cnn
    pkl.dump(predict_cnn,
            open(f'output/{interval}_{n_sliding}/cnn_{n_experiment}.pkl', 'wb'))
    smape_cnn = smape(Y_test, predict_cnn)

    lstm = lstm_model(55, 36, X_train.shape)
    lstm.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())
    time_lstm = time.time()
    lstm.fit(
            X_train,
            Y_train,
            epochs=500,
            callbacks=[es],
            validation_split=0.2)
    training_lstm = time.time() - time_lstm
    time_lstm = time.time()
    predict_lstm = lstm.predict(X_test)
    predicting_lstm = time.time() - time_lstm
    pkl.dump(predict_lstm,
            open(f'output/{interval}_{n_sliding}/lstm_{n_experiment}.pkl', 'wb'))
    smape_lstm = smape(Y_test, predict_lstm)

    convlstm = convlstm_model(55, 36, X_train.shape)
    convlstm.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())
    tiem_convlstm = time.time()
    convlstm.fit(
            X_train,
            Y_train,
            epochs=500,
            callbacks=[es],
            validation_split=0.2)
    training_convlstm = time.time() - time_convlstm
    time_convlstm = time.time()
    predict_convlstm = convlstm.predict(X_test)
    predicting_convlstm = time.time() - time_convlstm
    pkl.dump(predict_convlstm,
            open(f'output/{interval}_{n_sliding}/convlstm_{n_experiment}.pkl',
                'wb'))
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
    res['training_cnn'] = training_cnn
    res['training_lstm'] = training_lstm
    res['training_convlstm'] = training_convlstm
    res['predicting_cnn'] = predicting_cnn
    res['predicting_lstm'] = predicting_lstm
    res['predicting_convlstm'] = predicting_convlstm

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

    with open('result/cnn_summary.txt', 'w') as f:
        cnn.summary(print_fn=lambda x: f.write(x+'\n'))
    with open('result/lstm_summary.txt', 'w') as f:
        lstm.summary(print_fn=lambda x: f.write(x+'\n'))
    with open('result/convlstm_summary.txt', 'w') as f:
        convlstm.summary(print_fn=lambda x: f.write(x+'\n'))

    return res

def main():
    X_train, X_test, Y_train, Y_test = main_data('30min', 4)

    result = []
    for _ in tqdm(range(20)):
        result.append(train(X_train, X_test, Y_train, Y_test))

    df_res = pd.DataFrame(result)
    df_res.to_csv('result/result_base.csv')

if __name__ == '__main__':
    main()
