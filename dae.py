'''
Author: Sumin Lim @ KAIST BTM
Date: 2021-10-25
Usage: python dae.py
Description:
    * Run DAE-CNN, DAE-LSTM, DAE-ConvLSTM
'''
from typing import Tuple, Dict
from tqdm import tqdm
from datetime import datetime
from typing import Tuple
from pathlib import Path
import time
import argparse
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
from baseline import train as baseline

def smape(a: np.array, b: np.array) -> float:
    a = np.reshape(a, (-1, ))
    b = np.reshape(b, (-1, ))
    return np.mean(2.0*np.abs(a-b) / (np.abs(a)+np.abs(b)+1)).item()*100

def dae_model(rows: int, cols: int, conv=True) -> keras.Model:
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

def get_hidden(unet_feature: keras.Model, data: np.array) -> np.array:
    X_feature = []
    print('-----------------------------------------------------------------')
    print('Get hidden features as numpy array')
    print('-----------------------------------------------------------------')
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

def cnn_model(rows: int, cols: int, x_shape: Tuple[int]) -> keras.Model:
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

def lstm_model(rows: int, cols:int, x_shape: Tuple[int], n_sliding: int) -> keras.Model:
    inp = layers.Input(shape=x_shape[1:])
    x = layers.Reshape(target_shape=(n_sliding, x_shape[2]*x_shape[3]*x_shape[4]))(inp)
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

def convlstm_model(rows: int, cols: int, x_shape: Tuple[int]) -> keras.Model:
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

def main(X_train: np.array,
         X_test: np.array,
         Y_train: np.array,
         Y_test: np.array,
         tensor: Tuple[np.array],
         num_experiment: int,
         interval: str,
         n_sliding: int) -> Dict:
    Path(f'output/{interval}_{n_sliding}').mkdir(exist_ok=True)

    rmse = tf.keras.metrics.RootMeanSquaredError()

    dae, dae_feature = dae_model(X_train.shape[2], X_train.shape[3])
    dae.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())

    es = EarlyStopping(monitor='val_loss', mode='min', patience=5)

    time_dae = time.time()
    dae.fit(tensor, tensor, epochs=500, callbacks=[es], validation_split=0.2)
    training_dae = time.time() - time_dae

    predict_dae = dae.predict(tensor)

    X_train_feature = get_hidden(dae_feature, X_train)
    X_test_feature = get_hidden(dae_feature, X_test)

    print('Hidden features shape - X_train:', X_train_feature.shape)
    print('Hidden features shape - X_test:', X_test_feature.shape)
    cnn = cnn_model(55, 36, X_train_feature.shape)
    cnn.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())
    time_cnn = time.time()
    cnn.fit(
            X_train_feature,
            Y_train,
            epochs=500,
            callbacks=[es],
            validation_split=0.2)
    training_cnn = time.time() - time_cnn
    time_cnn = time.time()
    predict_cnn = cnn.predict(X_test_feature)
    predicting_cnn = time.time() - time_cnn
    pkl.dump(predict_cnn,
            open(f'output/{interval}_{n_sliding}/dae_cnn_{num_experiment}.pkl',
                'wb'))

    smape_cnn = smape(Y_test, predict_cnn)

    lstm = lstm_model(55, 36, X_train_feature.shape, n_sliding)
    lstm.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())
    time_lstm = time.time()
    lstm.fit(
            X_train_feature,
            Y_train,
            epochs=500,
            callbacks=[es],
            validation_split=0.2)
    training_lstm = time.time() - time_lstm
    time_lstm = time.time()
    predict_lstm = lstm.predict(X_test_feature)
    predicting_lstm = time.time() - time_lstm
    pkl.dump(predict_lstm,
            open(f'output/{interval}_{n_sliding}/dae_lstm_{num_experiment}.pkl',
                'wb'))

    smape_lstm = smape(Y_test, predict_lstm)

    convlstm = convlstm_model(55, 36, X_train_feature.shape)
    convlstm.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())
    time_convlstm = time.time()
    convlstm.fit(
            X_train_feature,
            Y_train,
            epochs=500,
            callbacks=[es],
            validation_split=0.2)
    training_convlstm = time.time() - time_convlstm
    time_convlstm = time.time()
    predict_convlstm = convlstm.predict(X_test_feature)
    predicting_convlstm = time.time() - time_convlstm

    pkl.dump(predict_convlstm,
            open(f'output/{interval}_{n_sliding}/dae_convlstm_{num_experiment}.pkl',
                'wb'))

    smape_convlstm = smape(Y_test, predict_convlstm)

    res = {}
    res['smape_dae_cnn'] = smape_cnn
    res['smape_dae_lstm'] = smape_lstm
    res['smape_dae_convlstm'] = smape_convlstm
    res['rmse_dae_cnn'] = rmse(Y_test, predict_cnn).numpy()
    res['rmse_dae_lstm'] = rmse(Y_test, predict_lstm).numpy()
    res['rmse_dae_convlstm'] = rmse(Y_test, predict_convlstm).numpy()
    res['rmse_dae'] = rmse(tensor, predict_dae).numpy()
    res['training_dae_cnn'] = training_cnn
    res['training_dae_lstm'] = training_lstm
    res['training_dae_convlstm'] = training_convlstm
    res['predicting_dae_cnn'] = predicting_cnn
    res['predicting_dae_lstm'] = predicting_lstm
    res['predicting_dae_convlstm'] = predicting_convlstm

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

    #with open('result/dae_cnn_summary.txt', 'w') as f:
    #    cnn.summary(print_fn=lambda x: f.write(x+'\n'))

    #with open('result/dae_lstm_summary.txt', 'w') as f:
    #    lstm.summary(print_fn=lambda x: f.write(x+'\n'))

    #with open('result/dae_convlstm_summary.txt', 'w') as f:
    #    convlstm.summary(print_fn=lambda x: f.write(x+'\n'))

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set the number of' \
            'experiments')
    parser.add_argument('-e', '--experiments',
            metavar='E',
            type=int,
            default=20,
            help='number of experiments')
    parser.add_argument('-i', '--interval',
            metavar='I',
            type=str)
    parser.add_argument('-n', '--n_sliding',
            metavar='N',
            type=int,
            help='Number of sliding windows to use for training')
    args = parser.parse_args()

    print()
    print('=================================================================')
    print('Current interval:', args.interval)
    print('Current n_sliding:', args.n_sliding)
    print('=================================================================')
    print()
    print('Make X_train, X_test, Y_train, Y_test and tensor ...')
    X_train, X_test, Y_train, Y_test = main_data(args.interval, args.n_sliding)
    tensor = main_data(args.interval, args.n_sliding, sliding=False)
    print('Data shape, X_train:', X_train.shape)
    print('Data shape, X_test:', X_test.shape)
    print('Data shape, Y_train:', Y_train.shape)
    print('Data shape, Y_test:', Y_test.shape)

    result = []
    print('\nExperiments...')
    for expr in tqdm(range(args.experiments)):
        res_dae = main(X_train,
                X_test,
                Y_train,
                Y_test,
                tensor,
                expr,
                args.interval,
                args.n_sliding)
        res_base = baseline(X_train,
                X_test,
                Y_train,
                Y_test,
                args.interval,
                args.n_sliding,
                expr)

        res = res_dae.update(res_base)
        result.append(res)



    df_res = pd.DataFrame(result)
    df_res.to_csv(f'result/result_{args.interval}_{args.n_sliding}.csv')
    keras.backend.clear_session()
