
from client import ClientFlower
from client_centralized import CentralizedClientFlower
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


centralized = False

n_clients = 2
n_rounds = 20
dataset = 'SKAB'
model_name = 'CNN'
anomaly_round = 30
model_shared = 'All'
loss_type = 'mse'
local_training = True
global_data = False
test_name = 'all_w_local_training_wo_global_data'
n_components = None
clients_with_anomaly = list(range(n_clients))

def funcao_cliente(cid):
	if centralized:
		return CentralizedClientFlower(int(cid), dataset = dataset,
					model_name=model_name, anomaly_round=anomaly_round,
					n_clients=n_clients, model_shared=model_shared, loss_type=loss_type,
					clients_with_anomaly = clients_with_anomaly,
					local_training = local_training, global_data=global_data, 
					test_name = test_name, n_components = n_components)
	else:

		return ClientFlower(int(cid), dataset = dataset,
						model_name=model_name, anomaly_round=anomaly_round,
						n_clients=n_clients, model_shared=model_shared, loss_type=loss_type,
						clients_with_anomaly = clients_with_anomaly,
						local_training = local_training, global_data=global_data, 
						test_name = test_name, n_components = n_components)


history = fl.simulation.start_simulation(client_fn=funcao_cliente, 
								num_clients=n_clients, 
								strategy= FedServer(),
								config=fl.server.ServerConfig(n_rounds))


#testar outros modelos e outras técnicas
