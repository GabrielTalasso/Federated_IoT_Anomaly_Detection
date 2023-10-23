import flwr as fl
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from model import get_lstm_model, get_conv_model
from load_dataset import load_dataset

class ClientFlower(fl.client.NumPyClient):

	def __init__(self, cid, dataset, model_name, anomaly_round, n_clients, model_shared = 'All', loss_type = 'mse',
			  clients_with_anomaly = []):
		self.cid = cid
		self.dataset = dataset
		self.model_name = model_name
		self.anomaly_round = anomaly_round
		self.n_clients = n_clients
		self.model_shared = model_shared
		self.loss_type = loss_type
		self.clients_with_anomaly = clients_with_anomaly

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

		true_anomaly = 0
		if (config['server_round'] == self.anomaly_round) and (self.cid in self.clients_with_anomaly):
			true_anomaly = 1
			hist = self.model.fit(self.x_test, self.x_test,
				 	epochs = n_epochs, batch_size = 8,
					validation_split=0.05)
		else:
			hist = self.model.fit(self.x_train, self.x_train,
						epochs = n_epochs, batch_size = 8,
						validation_split=0.05)
			
		loss = np.mean(hist.history['loss'])
		
		filename = f"teste/logs/{self.dataset}/{self.model_name}/train/loss_{self.loss_type}_{self.model_shared}.csv"

		#anomaly detect with threshold
		diff = 0
		anomaly = 0
		if config['server_round'] > 2:
			log_data = pd.read_csv(filename, names = ['cid', 'round', 'loss', 'diff', 'anomaly', 'true_anomaly'])
			log_data = log_data[log_data['cid'] == self.cid]
			last_loss = log_data['loss'].tail(1).values[0]
			mean_diff = log_data['diff'].mean()

			diff = abs(loss - last_loss )
			anomaly = 0
			if diff >= 2*mean_diff:
				anomaly = 1
	

		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(filename, 'a') as arquivo:
			arquivo.write(f"{self.cid}, {config['server_round']}, {loss}, {diff}, {anomaly}, {true_anomaly}\n")

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

		true_anomaly = 0
		if (config['server_round'] == self.anomaly_round) and (self.cid in self.clients_with_anomaly):
			loss = self.model.evaluate(self.x_test, self.x_test)
			true_anomaly = 1
		else:
			loss = self.model.evaluate(self.x_train, self.x_train)


		filename = f"teste/logs/{self.dataset}/{self.model_name}/evaluate/loss_{self.loss_type}_{self.model_shared}.csv"

		#anomaly detect with threshold
		diff = 0
		anomaly = 0
		if config['server_round'] > 2:
			log_data = pd.read_csv(filename, names = ['cid', 'round', 'loss', 'diff', 'anomaly', 'true_anomaly'])
			log_data = log_data[log_data['cid'] == self.cid]
			last_loss = log_data['loss'].tail(1).values[0]
			mean_diff = log_data['diff'].mean()

			diff = abs(loss - last_loss )
			anomaly = 0
			if diff >= 2*mean_diff:
				anomaly = 1


		os.makedirs(os.path.dirname(filename), exist_ok=True)
		with open(filename, 'a') as arquivo:
			arquivo.write(f"{self.cid}, {config['server_round']}, {loss}, {diff}, {anomaly}, {true_anomaly}\n")

		

		return loss, len(self.x_test), {"mean_loss": loss}