
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

n_clients = 10
n_rounds = 12
dataset = 'SKAB'
model_name = 'CNN'
anomaly_round = 8

def funcao_cliente(cid):
	return ClientFlower(int(cid), dataset = dataset, model_name=model_name, anomaly_round=anomaly_round, n_clients=n_clients)

history = fl.simulation.start_simulation(client_fn=funcao_cliente, 
								num_clients=n_clients, 
								strategy= FedServer(),
								config=fl.server.ServerConfig(n_rounds))

#with open('./results/history_simulation.pickle', 'wb') as file:
#    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)
