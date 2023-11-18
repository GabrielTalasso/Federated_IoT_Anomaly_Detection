import flwr as fl
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from model import get_lstm_model, get_conv_model
from load_dataset import load_dataset


def make_logs(filename, config, cid, loss):

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    anomaly = 0
    anomaly2 = 0
    anomaly3 = 0
    diff = 0

    th = 2
    r = config['server_round']

    if r>3:

        df = pd.read_csv(filename, names = ['cid', 'round', 'loss','diff', 'anomaly', 'anomaly2', 'anomaly3'])
        df['anomaly'] = 0
        df['anomaly2'] = 0
        df['anomaly3'] = 0
        df['anomaly12'] = 0

        try:
            diff = loss - df['loss'].tail(1).values[0]

            diff1 = df[(df['cid'] == cid) & (df['round'] == r-1)]['diff'].values[0]
            diff2 = df[(df['cid'] == cid) & (df['round'] == r-2)]['diff'].values[0]
            diff3 = df[(df['cid'] == cid) & (df['round'] == r-3)]['diff'].values[0]
            mean_diff = np.mean([diff1, diff2, diff3])

            anomaly = 0
            if diff >= th*mean_diff: #se cresceu mais que o esperado, é anomalia
                anomaly = 1

            diff = abs(loss - df['loss'].tail(1).values[0])

            diff1 = abs(df[(df['cid'] == cid) & (df['round'] == r-1)]['diff'].values[0])
            diff2 = abs(df[(df['cid'] == cid) & (df['round'] == r-2)]['diff'].values[0])
            diff3 = abs(df[(df['cid'] == cid) & (df['round'] == r-3)]['diff'].values[0])
            mean_diff = np.mean([diff1, diff2, diff3])

            anomaly3 = 0
            if diff <= th*mean_diff: #se cresceu mais que o esperado, é anomalia
                anomaly3 = 1

            anomaly2 = 0
            loss =  df[(df['cid'] == cid) & (df['round'] == r)]['loss'].values[0]
            last_losses1 = df[(df['cid'] == cid) & (df['round'] == r-1)]['loss'].values[0]
            last_losses2 = df[(df['cid'] == cid) & (df['round'] == r-2)]['loss'].values[0]

            if (loss - last_losses1) > 0:
                if (last_losses1 - last_losses2) > 0:
                    anomaly2 = 1 

        except IndexError:
            print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAaa', cid)


    with open(filename, 'a') as arquivo:
        arquivo.write(f"{cid}, {config['server_round']}, {loss}, {diff}, {anomaly}, {anomaly2}, {anomaly3}\n")

    return anomaly3 + anomaly2

#def make_logs(filename, config, cid, loss):
#
#    os.makedirs(os.path.dirname(filename), exist_ok=True)
#
#    anomaly = 0
#    anomaly2 = 0
#    diff = 0
#    if config['server_round'] > 3:
#        log_data = pd.read_csv(filename, names = ['cid', 'round', 'loss','diff', 'anomaly', 'anomaly2'])
#        log_data = log_data[log_data['cid'] == cid]
#        last_loss = log_data['loss'].tail(1).values[0]
#        diff = loss - last_loss #o quanto a loss cresce no inicio de um round
#
#        mean_diff = log_data['diff'].tail(3).mean() #media do crescimento nos ultimos 3 rounds
#        anomaly = 0
#        if diff >= 1.5*mean_diff: #se cresceu mais que o esperado, é a nomalia
#            anomaly = 1
#
#        anomaly2 = 0
#        last_losses = log_data['loss'].tail(3).values
#        if (loss - last_losses[-1]) > 0:
#            if (last_losses[-1] - last_losses[-2]) > 0:
#                #if (last_losses[-2] - last_losses[-3]) > 0:
#                anomaly2 = 1 #se a loss tem subido nos ultimos 2 rounds
#
#
#    with open(filename, 'a') as arquivo:
#        arquivo.write(f"{cid}, {config['server_round']}, {loss}, {diff}, {anomaly}, {anomaly2}\n")