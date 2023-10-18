import flwr as fl
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from model import get_lstm_model, get_conv_model
from load_dataset import load_dataset

class ClientFlower(fl.client.NumPyClient):

	def __init__(self, cid, dataset, model_name, anomaly_round, n_clients):
		self.cid = cid
		self.dataset = dataset
		self.model_name = model_name
		self.anomaly_round = anomaly_round
		self.n_clients = n_clients

		self.x_train, self.x_test= self.load_data()
		self.model = self.create_model(self.model_name)

	def create_model(self,model_name):
		if model_name == 'LSTM':
			model = get_lstm_model(self.x_train)
		if model_name == 'CNN':
			model = get_conv_model(self.x_train)
		return model

	def load_data(self):
		x_train, x_test = load_dataset(dataset_name=self.dataset, cid = self.cid, n_clients = self.n_clients)
		return x_train, x_test

	def get_parameters(self, config):
		return self.model.get_weights()

	def fit(self, parameters, config):
		
		server_round = int(config["server_round"])
		self.model.set_weights(parameters)

		self.model.compile(optimizer='adam', loss='mse')
		n_epochs = 5
		batch_size = 8

		if config['server_round'] == self.anomaly_round:
			hist = self.model.fit(self.x_test, self.x_test,
				 	epochs = n_epochs, batch_size = batch_size,
					validation_split=0.05)
			
		#print('----------------------------', self.x_train.shape)
		else:
			hist = self.model.fit(self.x_train, self.x_train,
						epochs = n_epochs, batch_size = batch_size,
						validation_split=0.05)
		
		loss = np.mean(hist.history['loss'])
		
		filename = f"logs/{self.dataset}/{self.model_name}/loss.csv"
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(filename, 'a') as arquivo:
			arquivo.write(f"{self.cid}, {config['server_round']}, {loss}\n")
		
		
		return self.model.get_weights(), len(self.x_train), {}


	def evaluate(self, parameters, config):
		self.model.set_weights(parameters)

		X_pred = self.model.predict(self.x_train)
		print('+++++++++++++++++++++++++++++++++++++++++++', X_pred.shape, self.x_train.shape)
		#X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])

		Xtrain = self.x_train
		#Xtrain = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[2])
		
		loss = np.mean(np.mean(np.abs(X_pred-Xtrain), axis=1) )
		loss = np.mean(((X_pred-Xtrain)**2))

		if config['server_round'] == self.anomaly_round:
			X_pred = self.model.predict(self.x_test)
			#X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
			Xtest = self.x_test
			#Xtest = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[2])
			loss = np.mean(np.mean(np.abs(X_pred-Xtest), axis=1)) 
			loss = np.mean(((X_pred-Xtest)**2))


		# filename = f"logs/{self.dataset}/{self.model_name}/loss.csv"
		# os.makedirs(os.path.dirname(filename), exist_ok=True)
		# with open(filename, 'a') as arquivo:
		# 	arquivo.write(f"{self.cid}, {config['server_round']}, {loss}\n")

		return loss, len(self.x_test), {"mean_loss": loss}