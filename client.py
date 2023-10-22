import flwr as fl
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from model import get_lstm_model, get_conv_model
from load_dataset import load_dataset

class ClientFlower(fl.client.NumPyClient):

	def __init__(self, cid, dataset, model_name, anomaly_round, n_clients, model_shared = 'All', loss_type = 'mse'):
		self.cid = cid
		self.dataset = dataset
		self.model_name = model_name
		self.anomaly_round = anomaly_round
		self.n_clients = n_clients
		self.model_shared = model_shared
		self.loss_type = loss_type

		self.x_train, self.x_test= self.load_data()
		self.model = self.create_model(self.model_name)

		self.encoder_len = int((len(self.model.get_weights()) - 2) / 2)
		self.decoder_len = int(self.encoder_len + 2)

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
		if server_round == 1:
			self.model.set_weights(parameters)

		if server_round>1:
			if self.model_shared == 'All':
				self.model.set_weights(parameters)
			
			elif self.model_shared == 'Decoder':
				for i in range(int(self.decoder_len/2)):
					lay = int((self.encoder_len/2)+i)
					self.model.layers[lay].set_weights([parameters[2*i], parameters[(2*i)+1]])

			elif self.model_shared == 'Encoder':
				for i in range(int((self.encoder_len/2)-1)):
					self.model.layers[i].set_weights([parameters[2*i], parameters[(2*i)+1]])


		self.model.compile(optimizer='adam', loss=self.loss_type)

		n_epochs = 5

		if config['server_round'] == self.anomaly_round:
			hist = self.model.fit(self.x_test, self.x_test,
				 	epochs = n_epochs, batch_size = 8,
					validation_split=0.05)
		else:
			hist = self.model.fit(self.x_train, self.x_train,
						epochs = n_epochs, batch_size = 8,
						validation_split=0.05)
		
		loss = np.mean(hist.history['loss'])
		
		filename = f"logs/{self.dataset}/{self.model_name}/train/loss_{self.loss_type}_{self.model_shared}.csv"
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(filename, 'a') as arquivo:
			arquivo.write(f"{self.cid}, {config['server_round']}, {loss}\n")

		if self.model_shared == 'All':
			return self.model.get_weights(), len(self.x_train), {}
		
		elif self.model_shared == 'Decoder':
			return self.model.get_weights()[-self.decoder_len:], len(self.x_train), {}
		
		elif self.model_shared == 'Encoder':
			return self.model.get_weights()[:self.encoder_len], len(self.x_train), {}


	def evaluate(self, parameters, config):

		if self.model_shared == 'All':
				self.model.set_weights(parameters)
			
		elif self.model_shared == 'Decoder':
			for i in range(int(self.decoder_len/2)):
				lay = int((self.encoder_len/2)+i)
				self.model.layers[lay].set_weights([parameters[2*i], parameters[(2*i)+1]])

		elif self.model_shared == 'Encoder':
			for i in range(int((self.encoder_len/2)-1)):
				self.model.layers[i].set_weights([parameters[2*i], parameters[(2*i)+1]])

		#self.model.set_weights(parameters)
		
		self.model.compile(optimizer='adam', loss=self.loss_type)
		loss = self.model.evaluate(self.x_train, self.x_train)

		if config['server_round'] == self.anomaly_round:
			loss = self.model.evaluate(self.x_test, self.x_test)

		filename = f"logs/{self.dataset}/{self.model_name}/evaluate/loss_{self.loss_type}_{self.model_shared}.csv"
		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(filename, 'a') as arquivo:
			arquivo.write(f"{self.cid}, {config['server_round']}, {loss}\n")

		return loss, len(self.x_test), {"mean_loss": loss}