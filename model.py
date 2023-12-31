from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from keras import layers
import keras
import numpy as np


def get_lstm_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))

    L1 = LSTM(100, activation='relu')(inputs)
    
    #L2 = LSTM(16, activation='relu', return_sequences=False)(L1)
    
    L3 = RepeatVector(X.shape[1])(L1)

    #L4 = LSTM(16, activation='relu', return_sequences=True)(L3)
    
    L5 = LSTM(100, activation='relu', return_sequences=True)(L3)
    
    output = TimeDistributed(Dense(X.shape[2]))(L5)  

    model = Model(inputs=inputs, outputs=output)
    return model

def get_conv_model(X):
    print('######################', X.shape)
    model = keras.Sequential(
        [
            layers.Input(shape=(X.shape[1], X.shape[2])),#, X.shape[3])),
            #layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=32, kernel_size=7, padding="same", strides=1, activation="relu"
            ),

            # layers.Dropout(rate=0.2),
            layers.Conv1D(
                filters=16, kernel_size=7, padding="same", strides=1, activation="relu"
            ),

            layers.Conv1DTranspose(
                filters=16, kernel_size=7, padding="same", strides=1, activation="relu"
            ),

            layers.Conv1DTranspose(
                filters=32, kernel_size=7, padding="same", strides=1, activation="relu"
            ),

            layers.Conv1DTranspose(
                filters=1, kernel_size=7, padding="same"),

            #layers.Dense(X.shape[2])
        ]
        )

    #model.load_weights('./checkpoints/my_checkpoint_PCA')


    return model