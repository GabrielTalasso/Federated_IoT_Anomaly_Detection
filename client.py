import flwr as fl
import pandas as pd
import numpy as np
import tensorflow as tf
from model import get_model
from load_dataset import load_dataset

class ClienteFlower(fl.client.NumPyClient):

	def _init_(self,cid):
		self.x_train, self.x_test= self.load_data()
		self.modelo = self.create_model()
		self.cid = cid

	def create_model(self):
		model = get_model(self.x_treino)
		return model

	def load_data(self):
		x_train, x_test = load_dataset(dataset_name='bearing', cid = self.cid)
		return x_train, x_test

	def get_parameters(self, config):
		return self.modelo.get_weights()

	def fit(self, parameters, config):
		server_round = int(config["server_round"])
		print(config["server_round"])
		self.model.set_weights(parameters)

		# create the autoencoder model
		model = get_model(self.x_train)
		model.compile(optimizer='adam', loss='mae')
		nb_epochs = 1
		batch_size = 10
		model.fit(self.x_train, self.x_test, epochs=nb_epochs, batch_size=batch_size,
						validation_split=0.05).history
		
		return self.model.get_weights(), len(self.x_treino), {}


	def evaluate(self, parameters, config):
		self.model.set_weights(parameters)
		loss, accuracy = self.modelo.evaluate(self.x_teste, self.y_teste)

		X_pred = self.model.predict(self.x_train)
		X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
		X_pred = pd.DataFrame(X_pred, columns=self.x_train.columns)
		X_pred.index = self.x_train.index

		scored = pd.DataFrame(index=self.x_train.index)
		Xtrain = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[2])
		scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)

		return loss, len(self.x_teste), {"mean_loss": scored['Loss_mae']}