'''
Author: Sumin Lim @ KAIST BTM
Date: 2021-10-25
Usage: python deep_tobit.py
Description:
    * Run Dee-tobit network
Reference: Zhang, Jiaming, Zhanfeng Li, Xinyuan Song, and Hanwen Ning.
    "Deep Tobit networks: A novel machine learning approach to microeconometrics."
    Neural Networks 144 (2021): 279-296.
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
from keras.activations import sigmoid, relu

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

def dtn1_model(rows: int, cols: int, x_shape: Tuple[int]):
    inp = layers.Input(shape=x_shape[1:])
    x = layers.Flatten()(inp)
    x = layers.Dense(rows*cols*8, activation='sigmoid')(x)
    x = layers.Dense(rows*cols*4, activation='sigmoid')(x)
    x = layers.Dense(rows*cols*2, activation='sigmoid')(x)
    x = layers.Dense(rows*cols, activation='sigmoid')(x)
    x = layers.ReLU(threshold=1)(x)
    out = layers.Reshape(target_shape=(1, rows, cols, 1))(x)

    model = keras.Model(inp, out)
    return model

def dtn2():
    inp = layers.Input()

def main():
    rmse = tf.keras.metrics.RootMeanSquaredError()

    X_train, X_test, y_train, y_test = get_data()
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)

    smape_allzero = smape(y_test, np.zeros(y_test.shape))
    print('SMAPE allzero:', smape_allzero)
    print('RMSE allzero:', rmse(y_test, np.zeros(y_test.shape)).numpy())

    # dtn1 loss: MSE, optimizer: SGD/Adam/AdaGrad/RMSProp, Adam is used in the paper
    dtn1 = dtn1_model(55, 36, X_train.shape)
    dtn1.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam())
    es = EarlyStopping(monitor='val_loss', mode='min', patience=20)
    dtn1.fit(
            X_train,
            y_train,
            epochs=500,
            callbacks=[es],
            validation_split=0.2)
    predict_dtn1 = dtn1.predict(X_test)
    smape_dtn1 = smape(y_test, predict_dtn1)
    print('SMAPE dtn1:', smape_dtn1)
    print('RMSE dtn1:', rmse(y_test, predict_dtn1).numpy())
    print(predict_dtn1[np.where(predict_dtn1!=0)].max())
    print(predict_dtn1[np.where(predict_dtn1!=0)].min())

if __name__ == '__main__':
    main()
