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

def dae_model(rows: int, cols: int, conv=True):
    inp = layers.Input(shape=(rows, cols, 1))
    print('input:', inp.shape)
    flatten1 = layers.Flatten()(inp)
    dense1 = layers.Dense(800, activation='relu')(flatten1)
    print('dense:', dense1.shape)
    reshape1 = layers.Reshape(target_shape=(40, 20, 1))(dense1)
    print('reshape1:', reshape1.shape)
    if conv:
        conv1 = layers.Conv2D(
                filters=32,
                kernel_size=(5,3),
                padding='same',
                activation='relu')(reshape1)
        print('conv1:', conv1.shape)
        max1 = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)
        print('max1:', max1.shape)
        conv2 = layers.Conv2D(
                filters=16,
                kernel_size=(5,3),
                padding='same',
                activation='relu')(max1)
        print('conv2:', conv2.shape)
        max2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
        print('max2:', max2.shape)
        conv3 = layers.Conv2D(
                filters=4,
                kernel_size=(5,3),
                padding='same',
                activation='relu')(max2)
        print('conv3:', conv3.shape)
        print()

        deconv1 = layers.Conv2DTranspose(
                filters=4,
                kernel_size=(5, 3),
                padding='same',
                activation='relu')(conv3)
        print('deconv1:', deconv1.shape)
        up1 = layers.UpSampling2D(size=(2, 2))(deconv1)
        print('up1:', up1.shape)
        deconv2 = layers.Conv2DTranspose(
                filters=16,
                kernel_size=(5, 3),
                padding='same',
                activation='relu')(up1)
        print('deconv2:', deconv2.shape)
        up2 = layers.UpSampling2D(size=(2, 2))(deconv2)
        print('up2:', up2.shape)
        deconv3 = layers.Conv2DTranspose(
                filters=32,
                kernel_size=(5, 3),
                padding='same',
                activation='relu')(up2)
        print('deconv3:', deconv3.shape)

    else:
        pass


    flatten2 = layers.Flatten()(deconv3)
    dense2 = layers.Dense(rows*cols, activation='relu')(flatten2)
    print('dense2:', dense2.shape)
    out = layers.Reshape(target_shape=(rows, cols, 1))(dense2)
    print('reshape2:', out.shape)

    model_train = keras.Model(inp, out)
    model_intermediate = keras.Model(inp, conv3)
    return model_train, model_intermediate

def get_hidden(unet_feature, data):
    X_feature = []
    for seq in tqdm(data):
        temp = []
        for batch in seq:
            batch = batch.reshape((1, 55, 36, 1))
            hidden = unet_feature.predict(batch)
            temp.append(hidden)
        temp = np.concatenate(temp)
        X_feature.append(temp)

    X_feature = np.array(X_feature)
    return X_feature

def cnn_model(rows: int, cols: int, x_shape: Tuple[int]):
    inp = layers.Input(shape=x_shape[1:])
    x = layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation='relu',
            padding='same')(inp)
    x = layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation='relu',
            padding='same')(x)
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
    x = layers.LSTM(
            units=32,
            return_sequences=True
            )(x)
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
    x = layers.ConvLSTM2D(
            filters=64,
            kernel_size=(3, 3),
            padding='same',
            return_sequences=True,
            activation='relu')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(rows*cols, activation='relu')(x)
    out = layers.Reshape(target_shape=(1, rows, cols, 1))(x)

    model = keras.Model(inp, out)
    return model

def main(X_train, X_test, Y_train, Y_test):
    rmse = tf.keras.metrics.RootMeanSquaredError()

    tensor = get_data(sliding=False)
    dae, dae_feature = dae_model(X_train.shape[2], X_train.shape[3])
    dae.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())

    es = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    dae.fit(tensor, tensor, epochs=500, callbacks=[es], validation_split=0.2)

    X_train_feature = get_hidden(dae_feature, X_train)
    X_test_feature = get_hidden(dae_feature, X_test)


    cnn = cnn_model(55, 36, X_train_feature.shape)
    cnn.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())
    cnn.fit(
            X_train_feature,
            Y_train,
            epochs=500,
            callbacks=[es],
            validation_split=0.2)
    predict_cnn = cnn.predict(X_test_feature)
    smape_cnn = smape(Y_test, predict_cnn)

    lstm= lstm_model(55, 36, X_train_feature.shape)
    lstm.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())
    lstm.fit(
            X_train_feature,
            Y_train,
            epochs=500,
            callbacks=[es],
            validation_split=0.2)
    predict_lstm = lstm.predict(X_test_feature)
    smape_lstm = smape(Y_test, predict_lstm)

    convlstm = convlstm_model(55, 36, X_train_feature.shape)
    convlstm.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())
    convlstm.fit(
            X_train_feature,
            Y_train,
            epochs=500,
            callbacks=[es],
            validation_split=0.2)
    predict_convlstm = convlstm.predict(X_test_feature)
    smape_convlstm = smape(Y_test, predict_convlstm)

    res = {}
    res['smape_cnn'] = smape_cnn
    res['smape_lstm'] = smape_lstm
    res['smape_convlstm'] = smape_convlstm
    res['rmse_cnn'] = rmse(Y_test, predict_cnn).numpy()
    res['rmse_lstm'] = rmse(Y_test, predict_lstm).numpy()
    res['rmse_convlstm'] = rmse(Y_test, predict_convlstm).numpy()
    res['rmse_unet'] = rmse(tensor, tensor).numpy()

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
    print()

    print('=============================================================')
    print('CNN model')
    print(cnn.summary())
    print('LSTM model')
    print(lstm.summary())
    print('ConvLSTM model')
    print(convlstm.summary())
    return res

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = get_data()

    result = []
    for _ in tqdm(range(10)):
        result.append(main(X_train, X_test, Y_train, Y_test))

    df_res = pd.DataFrame(result)
    df_res.to_csv('result/result_unet.csv')