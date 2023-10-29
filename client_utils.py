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
    diff = 0
    if config['server_round'] > 3:
        log_data = pd.read_csv(filename, names = ['cid', 'round', 'loss','diff', 'anomaly', 'anomaly2'])
        log_data = log_data[log_data['cid'] == cid]
        last_loss = log_data['loss'].tail(1).values[0]
        diff = loss - last_loss #o quanto a loss cresce no inicio de um round

        mean_diff = log_data['diff'].tail(3).mean() #media do crescimento nos ultimos 3 rounds
        anomaly = 0
        if diff >= 1.5*mean_diff: #se cresceu mais que o esperado, é a nomalia
            anomaly = 1

        anomaly2 = 0
        last_losses = log_data['loss'].tail(3).values
        if (loss - last_losses[-1]) > 0:
            if (last_losses[-1] - last_losses[-2]) > 0:
                #if (last_losses[-2] - last_losses[-3]) > 0:
                anomaly2 = 1 #se a loss tem subido nos ultimos 3 rounds


    with open(filename, 'a') as arquivo:
        arquivo.write(f"{cid}, {config['server_round']}, {loss}, {diff}, {anomaly}, {anomaly2}\n")