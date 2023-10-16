import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
#from sklearn.externals import joblib
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt

from numpy.random import seed
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

def load_dataset(dataset_name, cid, n_clients):
# load, average and merge sensor samples

    if dataset_name == 'bearing':
        data_dir = 'data/bearing_data'
        merged_data = pd.DataFrame()
        for filename in os.listdir(data_dir):
            dataset = pd.read_csv(os.path.join(data_dir, filename), sep='\t')
            dataset_mean_abs = np.array(dataset.abs().mean())
            dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,4))
            dataset_mean_abs.index = [filename]
            #merged_data = merged_data.append(dataset_mean_abs)
            merged_data = pd.concat([merged_data, dataset_mean_abs], axis = 0)
        merged_data.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']
        # transform data file index to datetime and sort in chronological order
        merged_data.index = pd.to_datetime(merged_data.index, format='%Y.%m.%d.%H.%M.%S')
        merged_data = merged_data.sort_index()

        train = merged_data['2004-02-12 10:52:39': '2004-02-15 12:52:39']
        test = merged_data['2004-02-15 12:52:39':]

        train = pd.DataFrame(train[f'Bearing {cid+1}'])
        test  = pd.DataFrame( test[f'Bearing {cid+1}'])

        # normalize the data
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(train)
        X_test = scaler.transform(test)

        # reshape inputs for LSTM [samples, timesteps, features]
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])


    if dataset_name == 'SKAB': 

        data = pd.read_csv('data/SKAB/anomaly-free/anomaly-free.csv', sep = ';')

        data.index = pd.to_datetime(data['datetime'])
        data.drop('datetime', axis = 1, inplace=True)


        size = len(data)
        int_cid = int(cid)

        train = data.reset_index().iloc[int(int_cid*size/(n_clients+1)) : int((int_cid+1)*size/(n_clients+1))].set_index('datetime').values
        test =  data.reset_index().iloc[int((int_cid+1)*size/(n_clients+1)) : ].set_index('datetime').values

        #train = data.head(9000).values
        #test = data.tail(1400).values

        #for anomaly:
        data = pd.read_csv('/home/gabrieltalasso/IoT_Anomaly_Detection/data/SKAB/valve2/0.csv', sep = ';')
        data.index = pd.to_datetime(data['datetime'])
        data.drop('datetime', axis = 1, inplace=True)
        test = data[data['anomaly'] == 0]
        test.drop(['anomaly', 'changepoint'], axis = 1, inplace = True)
        test = test.values
        
        X_train = train.reshape(train.shape[0], 1, train.shape[1])
        X_test = test.reshape(test.shape[0], 1, test.shape[1])

    return X_train, X_test