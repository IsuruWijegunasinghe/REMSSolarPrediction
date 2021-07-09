import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
if tf.config.list_physical_devices('GPU'):
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
np.random.seed(42)

import logging
logging.getLogger('tensorflow').disabled = True

df = pd.read_csv('C:/Users/Isuru Wijegunasinghe/Desktop/NN Model/HourTempWind.csv')
dataset = df.values
X = dataset[:, 0:3]
Y = dataset[:, 3]
print("input shape : {}".format(X.shape))
print("output shape : {}".format(Y.shape))
print(X[1])
print(X[1].shape)

inputs = Input(shape=(X.shape[1],))
x = Dense(32, activation='relu', name='dense1')(inputs)
x = Dense(32, activation='relu', name='dense2')(x)
outputs = Dense(1, name='dense3')(x)

model = Model(
    inputs=inputs,
    outputs=outputs)

model.compile(
    loss='mse',
    optimizer='adam',
    metrics=[tf.keras.metrics.RootMeanSquaredError()])

history = model.fit(
    X,
    Y,
    batch_size=64,
    epochs=200,
    validation_split=0.2,
    verbose=1)

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('Model accuracy')
plt.ylabel('RootMeanSquaredError')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

y_pred = model.predict(X[7488:7512], batch_size=64, verbose=1, steps=None, callbacks=None, max_queue_size=10,
    workers=1, use_multiprocessing=False)
print(X[7488:7512])
print(y_pred)