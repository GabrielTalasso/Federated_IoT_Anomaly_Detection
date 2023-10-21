
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
n_rounds = 40
dataset = 'SKAB'
model_name = 'CNN'
anomaly_round = 35

def funcao_cliente(cid):
	return ClientFlower(int(cid), dataset = dataset, model_name=model_name, anomaly_round=anomaly_round, n_clients=n_clients)

history = fl.simulation.start_simulation(client_fn=funcao_cliente, 
								num_clients=n_clients, 
								strategy= FedServer(),
								config=fl.server.ServerConfig(n_rounds))

#testar apenas compartilhando o decoder
#testar outros modelos e outras técnicas
