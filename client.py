import flwr as fl
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from model import get_model
from load_dataset import load_dataset

class ClientFlower(fl.client.NumPyClient):

	def __init__(self, cid):
		self.cid = cid
		self.x_train, self.x_test= self.load_data()
		self.model = self.create_model()

	def create_model(self):
		model = get_model(self.x_train)
		return model

	def load_data(self):
		x_train, x_test = load_dataset(dataset_name='bearing', cid = self.cid)
		return x_train, x_test

	def get_parameters(self, config):
		return self.model.get_weights()

	def fit(self, parameters, config):
		server_round = int(config["server_round"])
		print(config["server_round"])
		self.model.set_weights(parameters)

		# create the autoencoder model
		model = get_model(self.x_train)
		model.compile(optimizer='adam', loss='mae')
		nb_epochs = 1
		batch_size = 10
		model.fit(self.x_train, self.x_train, epochs=nb_epochs, batch_size=batch_size,
						validation_split=0.05)
		
		#filename = f"local_logs/.csv"
		#os.makedirs(os.path.dirname(filename), exist_ok=True)
		#with open(filename, 'a') as arquivo:
		#	arquivo.write(f"")
		
		return self.model.get_weights(), len(self.x_train), {}


	def evaluate(self, parameters, config):
		self.model.set_weights(parameters)
		X_pred = self.model.predict(self.x_train)
		X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])

		Xtrain = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[2])
		loss = np.mean(np.abs(X_pred-Xtrain)) 

		return loss, len(self.x_test), {"mean_loss": loss}