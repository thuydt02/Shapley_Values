#This is to load a dataset for linear regession tasks from a csv file

import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset

class Data_Loader:
	def __init__(self, train_fname = '', test_fname = ''):

		self.train_fname = train_fname
		self.test_fname = test_fname
	
		self.train_data_df = self.load_data_in_df(train = True)
		self.test_data_df = self.load_data_in_df(train = False)

	def get_train_data_in_array(self):
		return self.train_data_df.values

	def get_test_data_in_array(self):
		return self.test_data_df.values

	def get_train_data_in_batch(self, batch_size):
		batches = self.get_data_in_batch(train = True, batch_size = batch_size)
		
		return DataLoader(ConcatDataset(batches), shuffle=True, batch_size=batch_size)

	def get_test_data_in_batch(self, batch_size):

		batches = self.get_data_in_batch(train = False, batch_size = batch_size)
		return DataLoader(ConcatDataset(batches), shuffle=True, batch_size=batch_size)

	def load_data_in_df(self, train):

		fname = ''
		if train == True:
			if self.train_fname != '':
				fname = self.train_fname
			else:
				print('The file name of the train data set is empty!')
				return
		else:
			if self.test_fname != '':
				fname = self.test_fname
			else:
				print('The file name of the test data set is empty!')
				return		
		
		df = pd.read_csv(fname, index_col=None)
		return df #df.values #in the tabular shape of (x1, X2, ..., Y)

	def get_data_in_batch(self, train, batch_size):
		
		columns = self.train_data_df.columns
		n_columns = len(columns)
		
		if train == True:
			n_samples = self.train_data_df.shape[0]
			X = self.train_data_df[columns[0:n_columns - 1]].values
			Y = self.train_data_df[columns[n_columns - 1]].values
		else:
			n_samples = self.test_data_df.shape[0]	
			X = self.test_data_df[columns[0:n_columns - 1]].values
			Y = self.test_data_df[columns[n_columns - 1]].values
		
		n_batches = int(n_samples / batch_size)
		batches = []

		for i in range(n_batches):
			start = i * batch_size
			end = start + batch_size

			batch_X = X[start:end]
			batch_Y = Y[start:end]

			#print("batch_X: ", batch_X)
			#print("batch_Y: ", batch_Y)


			batch = TensorDataset(torch.Tensor(batch_X), torch.Tensor(batch_Y))
			batches.append(batch)
		return batches
	
	def get_dataloader_given_batch_ranges(self, train, batch_size, range_list):
		batches = self.get_data_in_batch(train = train, batch_size = batch_size)
		ret_batches = []
		for r in range_list:
			for batch_index in r:
				ret_batches.append(batches[batch_index])
		print ("len of ret_batches: ", len(ret_batches))
		return DataLoader(ConcatDataset(ret_batches), shuffle = False, batch_size=batch_size)

	def distribute_in_batch_given_partition(self, n_clients, batch_size, p):
		
		#a client has at least one batch
		#p is the distribution in batches of clients' train data; 
		#p[0] = 0.1 means the number of batches for client 0 is 0.1 * the number of all batches 

		batches = self.get_data_in_batch(train = True, batch_size = batch_size)
		n_batches = len(batches)
		
		clients_data = {}
		
		start = 0
		for i in range(n_clients):#
			end = start + int(p[i] * n_batches)

			clients_data[i] = batches[start:end]
			start = end

		if start < n_batches:
			clients_data[0] += batches[start: n_batches]

		for client_i, client_data in clients_data.items():
			clients_data[client_i] = DataLoader(ConcatDataset(client_data), shuffle = False, batch_size = batch_size)

		return clients_data

	def map_in_batch_given_partition(self, n_clients, batch_size, p):
		
		#a client has at least one batch
		#p is the distribution in batches of clients' train data; 
		#p[0] = 0.1 means the number of batches for client 0 is 0.1 * the number of all batches 

		batches = self.get_data_in_batch(train = True, batch_size = batch_size)
		n_batches = len(batches)
		
		clients_data = {}
		
		start = 0
		for i in range(n_clients):
			end = start + int(p[i] * n_batches)

			clients_data[i] = range(start, end, 1)
			start = end

		#if start < n_batches:
		#	clients_data[0] += batches[start: n_batches]

		#for client_i, client_data in clients_data.items():
		#	clients_data[client_i] = DataLoader(ConcatDataset(client_data), shuffle = True, batch_size = batch_size)

		return clients_data	

		
	def distribute_in_samples(self, n_clients):
		
		#distribute the train data set into n_clients by y-intervals
		#a client has at least one batch

		n_samples = self.train_data_df.shape[0]
		n_features = self.train_data_df.shape[1] - 1
		columns = self.train_data_df.columns
		n_columns = len(columns)

		Y_series = self.train_data_df[columns[n_columns - 1]]

		min_y = min(Y_series)
		max_y = max(Y_series)
		interval_y = (max_y - min_y) / n_clients

		clients_data = {}
		start = min_y

		for i in range(n_clients):
			end = start + interval_y
			if i < n_clients - 1:
				ind = (Y_series >= start) & (Y_series < end)
			else:
				ind = (Y_series >= start) & (Y_series <= end)
			clients_data[i] = self.train_data_df[ind].values
			start = end
		return clients_data

	