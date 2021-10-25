from tqdm import tqdm
from datetime import datetime
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

seed_value = 42
tf.random.set_seed(seed_value)

def get_data(binary=True):
    if binary:
        sliding_dates_week, _, tensor = main_data()
    else:
        sliding_dates_week, tensor, _ = main_data()

    n_weeks_sliding_train = (len(sliding_dates_week) * 2) // 3

    days_train = []
    days_test = []
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for idx in range(len(sliding_dates_week)):
        week = sliding_dates_week[idx]
        X_data = np.array([np.expand_dims(tensor[x],axis=-1) for x in week[:6]])
        y_data = np.array([np.expand_dims(tensor[week[-1]],axis=-1)])
        if idx < n_weeks_sliding_train:
            days_train.extend(week)
            X_train.append(X_data)
            y_train.append(y_data)
        else:
            days_test.extend(week)
            X_test.append(X_data)
            y_test.append(y_data)

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    y_train = np.swapaxes(y_train, 1, -1)
    y_test = np.swapaxes(y_test, 1, -1)

    return X_train, X_test, y_train, y_test

def cnnlstm(rows, cols):
    inp = layers.Input(shape=(6, rows, cols, 1))
    model = models.Sequential()
    model.add(layers.Input(shape=(6, rows, cols, 1)))
    model.add(layers.ConvLSTM2D(
        filters=64,
        kernel_size=(5, 5),
        padding="same",
        return_sequences=True,
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed_value),
        activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.ConvLSTM2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        return_sequences=True,
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed_value),
        activation="relu"))
    model.add(layers.BatchNormalization())
    model.add(layers.ConvLSTM2D(
        filters=64,
        kernel_size=(1, 1),
        padding="same",
        return_sequences=True,
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed_value),
        activation="relu"))
    model.add(layers.Conv3D(
        filters=1, kernel_size=(3, 3, 3), activation="sigmoid", padding="same",
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed_value)))
    model.add(layers.Flatten())
    model.add(layers.Dense(rows*cols,# 55*36,
        kernel_initializer=keras.initializers.glorot_uniform(seed=seed_value)))
    model.add(layers.Reshape((1, rows, cols, 1)))
    return model

def cnn(rows, cols):
    model = models.Sequential()
    model.add(layers.Input(shape=(6, rows, cols, 1)))
    model.add(layers.Conv2D(
        filters=64,
        kernel_size=3,
        activation='relu',
        padding='same',
        input_shape=(None, rows, cols, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=16, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(55*36, kernel_initializer=keras.initializers.glorot_uniform(seed=42)))
    model.add(layers.Reshape((1, rows, cols, 1)))
    model.add(layers.Dense(1))
    return model

def train():
    X_train, X_test, y_train, y_test = get_data(binary=True)
    model1 = cnnlstm(55, 36)
    model2 = cnn(55, 36)
    # Next, we will build the complete model and compile it.
    model1.compile(
        #loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(),
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']#, Recall()]
    )

    print(model1.summary())
    print()

    batch_size = 5

    es = EarlyStopping(monitor='val_loss', mode='min', patience=5)

    model1.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=200,
        validation_split=0.2,
        callbacks=[es]
    )

    y_train1 = model1.predict(X_train)
    y_train1 = sigmoid(y_train1)
    y_train1 = y_train1 > 0.5
    pkl.dump(y_train1, open('result/binary_train_cnnlstm.pkl', 'wb'))

    y_hat_cnnLstm = model1.predict(X_test)
    y_hat_cnnLstm = sigmoid(y_hat_cnnLstm)
    y_hat1 = np.squeeze(y_hat_cnnLstm, axis=(1, -1))
    y_hat1 = y_hat1 > 0.5
    y_hat1 = y_hat1.astype(int)
    pkl.dump(y_hat1, open('result/binary_cnnlstm.pkl', 'wb'))

    y_test_sq = np.squeeze(y_test, axis=(1, -1))

    print(y_hat_cnnLstm)
    #rmse = tf.keras.metrics.RootMeanSquaredError()
    print('========================================================')
    print('CNN-LSTM with Sequence Model Performance')
    #print('RMSE:', rmse(y_test, y_hat_cnnLstm).numpy())
    print('accuracy:', np.mean(y_hat1 == y_test_sq))
    print('========================================================')

    model2.compile(
        #loss=keras.losses.mean_squared_error, optimizer=keras.optimizers.Adam(),
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy']#, Recall()]
    )

    print(model2.summary())
    print()

    batch_size = 5

    model2.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=200,
        validation_split=0.2,
        callbacks=[es]
    )

    y_hat_cnn = model2.predict(X_test)
    y_hat_cnn = sigmoid(y_hat_cnn)
    y_hat2 = np.squeeze(y_hat_cnn, axis=(1, -1))
    y_hat2 = y_hat2 > 0.5
    y_hat2 = y_hat2.astype(int)

    print('========================================================')
    print('CNN with Sequence Model Performance')
    print('accuracy:', np.mean(y_hat2 == y_test_sq))
    print('========================================================')

    y_train2 = model2.predict(X_train)
    y_train2 = sigmoid(y_train2)
    y_train2 = y_train2 > 0.5

    pkl.dump(y_train2, open('result/binary_train_cnn.pkl', 'wb'))
    pkl.dump(y_hat2, open('result/binary_cnn.pkl', 'wb'))
    pkl.dump(y_test_sq, open('result/binary_y.pkl', 'wb'))

if __name__=='__main__':
    train()
