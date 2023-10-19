from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from keras import layers
import keras
import numpy as np


def get_lstm_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2], 1))

    L1 = LSTM(8, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00),
              stateful=True)(inputs)
    
    L2 = LSTM(4, activation='relu', return_sequences=False,
              stateful=True)(L1)
    
    L3 = RepeatVector(X.shape[1])(L2)

    L4 = LSTM(4, activation='relu', return_sequences=True,
              stateful=True)(L3)
    
    L5 = LSTM(8, activation='relu', return_sequences=True,
              stateful=True)(L4)
    
    output = TimeDistributed(Dense(X.shape[2]))(L5)  

    model = Model(inputs=inputs, outputs=output)
    return model

def get_conv_model(X):
    print('######################', X.shape)
    model = keras.Sequential(
        [
            layers.Input(shape=(X.shape[1], X.shape[2])),#, X.shape[3])),
            layers.BatchNormalization(),
            layers.Conv1D(
                filters=32, kernel_size=3, padding="same", strides=1, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=3, padding="same", strides=1, activation="relu"
            ),

            layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=8, kernel_size=3, padding="same", strides=1, activation="relu"
            ),

            layers.Conv1DTranspose(
                filters=8, kernel_size=3, padding="same", strides=1, activation="relu"
            ),

            layers.Conv1DTranspose(
                filters=16, kernel_size=3, padding="same", strides=1, activation="relu"
            ),
            layers.Dropout(rate=0.2),
            layers.Conv1DTranspose(
                filters=32, kernel_size=3, padding="same", strides=1, activation="relu"
            ),
            layers.Conv1DTranspose(
                filters=1, kernel_size=3, padding="same"),

            #layers.Dense(X.shape[2])
        ]
        )
    
    return model