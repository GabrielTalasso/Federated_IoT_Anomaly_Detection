
from client import ClientFlower
from server import FedServer
import pickle
import flwr as fl
import os
import sys

try:
	os.remove('./results/history_simulation.pickle')
except FileNotFoundError:
	pass

try:
	os.remove('./results/acc.csv')
except FileNotFoundError:
	pass

n_clients = 16
n_rounds = 29
dataset = 'SKAB'
model_name = 'CNN'
anomaly_round = 30
model_shared = 'All'
loss_type = 'mse'
local_training = True

clients_with_anomaly = list(range(n_clients))


def funcao_cliente(cid):
	return ClientFlower(int(cid), dataset = dataset,
					model_name=model_name, anomaly_round=anomaly_round,
					n_clients=n_clients, model_shared=model_shared, loss_type=loss_type,
					clients_with_anomaly = clients_with_anomaly,
					local_training = local_training)

history = fl.simulation.start_simulation(client_fn=funcao_cliente, 
								num_clients=n_clients, 
								strategy= FedServer(),
								config=fl.server.ServerConfig(n_rounds))


#testar outros modelos e outras técnicas
