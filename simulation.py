from client import ClientBase
from server import FedSCCS
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

n_clients = 5
n_rounds = 10

def funcao_cliente(cid):
	return ClientBase(int('cid'))

history = fl.simulation.start_simulation(client_fn=funcao_cliente, 
								num_clients=n_clients, 
								strategy= ),
								config=fl.server.ServerConfig(n_rounds))



with open('./results/history_simulation.pickle', 'wb') as file:
    pickle.dump(history, file, protocol=pickle.HIGHEST_PROTOCOL)


##python3 simulation.py --dataset MotionSense --nclients 24 --nrounds 30 --nclusters 5 --clustering True --clusterround 5 --noniid True --selectionmethod All --clustermetric CKA --metriclayer -1 --clustermethod HC --pocpercofclients 0.5
##python3 simulation.py --dataset MotionSense --nclients 24 --nrounds 30 --nclusters 5 --clustering True --clusterround 5 --noniid True --selectionmethod Random --clustermetric weights --metriclayer -1 --clustermethod KCenter --pocpercofclients 0.5
##python3 simulation.py --dataset MotionSense --nclients 24 --nrounds 30 --nclusters 1 --clustering True --clusterround 5 --noniid True --selectionmethod All --clustermetric weights --metriclayer -1 --clustermethod HC --pocpercofclients 0.5
##python3 simulation.py --dataset MotionSense --nclients 24 --nrounds 30 --nclusters 5 --clustering True --clusterround 5 --noniid True --selectionmethod All --clustermetric weights --metriclayer -1 --clustermethod HC --pocpercofclients 0.5
