import flwr as fl
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from model import get_lstm_model, get_conv_model
from load_dataset import load_dataset
from client_utils import make_logs

class ClientFlower(fl.client.NumPyClient):

	def __init__(self, cid, dataset, model_name, anomaly_round, n_clients, model_shared = 'All', loss_type = 'mse',
			  clients_with_anomaly = [], local_training = True, global_data = False, test_name = 'test'):
		self.cid = cid
		self.dataset = dataset
		self.model_name = model_name
		self.anomaly_round = anomaly_round
		self.n_clients = n_clients
		self.model_shared = model_shared
		self.loss_type = loss_type
		self.clients_with_anomaly = clients_with_anomaly
		self.local_training  = local_training
		self.global_data = global_data
		self.test_name = test_name

		self.x_train, self.x_test= self.load_data(server_round=1)
		self.model = self.create_model(self.model_name)

		self.encoder_len = int((len(self.model.get_weights()) - 2) / 2)
		self.decoder_len = int(self.encoder_len + 2)

	def create_model(self, model_name):
		if model_name == 'LSTM':
			model = get_lstm_model(self.x_train)
		if model_name == 'CNN':
			model = get_conv_model(self.x_train)
		return model

	def load_data(self, server_round, dataset_size = 60):
		x_train, x_test = load_dataset(dataset_name=self.dataset, cid = self.cid, n_clients = self.n_clients,
								 server_round = server_round, dataset_size = dataset_size, global_data=self.global_data)
		return x_train, x_test

	def get_parameters(self):
		return self.model.get_weights()
	
	def set_parameters(self, config, parameters):

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

		return self.model.get_weights()

	def fit(self, parameters, config):

		self.x_train, self.x_test= self.load_data(server_round=int(config['server_round']))
		self.set_parameters(config = config, parameters=parameters)
		self.model.compile(optimizer='adam', loss=self.loss_type)

		loss = pd.Series(np.sum(np.mean(np.abs(self.x_test - self.model.predict(self.x_test)), axis=1), axis=1)).values[0]

		filename = f"logs/{self.dataset}/{self.model_name}/{self.test_name}/evaluate_before_train/loss_{self.loss_type}_{self.model_shared}.csv"
		make_logs(filename, config, cid = self.cid, loss = loss)

		if self.local_training:

			self.set_parameters(config = config, parameters=parameters )
			self.model.compile(optimizer='adam', loss=self.loss_type)

			n_epochs = 100
			hist = self.model.fit(self.x_train, self.x_train,
					epochs = n_epochs, batch_size = 32,
					validation_split=0.05)
				
			loss = hist.history['loss'][-1]		
			filename = f"logs/{self.dataset}/{self.model_name}/{self.test_name}/train/loss_{self.loss_type}_{self.model_shared}.csv"
			make_logs(filename, config, cid = self.cid, loss = loss)

		if not self.local_training:
			self.model = get_conv_model(self.x_train)

		if self.model_shared == 'All':
			return self.model.get_weights(), len(self.x_train), {}
		
		elif self.model_shared == 'Decoder':
			return self.model.get_weights()[-self.decoder_len:], len(self.x_train), {}
		
		elif self.model_shared == 'Encoder':
			return self.model.get_weights()[:self.encoder_len], len(self.x_train), {}

	def evaluate(self, parameters, config):

		self.set_parameters(config = config, parameters=parameters)
		self.model.compile(optimizer='adam', loss=self.loss_type)

		filename = f"logs/{self.dataset}/{self.model_name}/{self.test_name}/evaluate/loss_{self.loss_type}_{self.model_shared}.csv"
		loss = pd.Series(np.sum(np.mean(np.abs(self.x_test - self.model.predict(self.x_test)), axis=1), axis=1)).values[0]
		make_logs(filename, config, cid = self.cid, loss = loss)

		return loss, len(self.x_test), {"mean_loss": loss}