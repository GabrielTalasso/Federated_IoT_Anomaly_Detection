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

n_clients = 4
n_rounds = 10

def funcao_cliente(cid):
	return ClientFlower(int(cid))

history = fl.simulation.start_simulation(client_fn=funcao_cliente, 
								num_clients=n_clients, 
								strategy= FedServer(),
								config=fl.server.ServerConfig(n_rounds))



with open('./results/history_simulation.pickle', 'wb') as file:
    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)
