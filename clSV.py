import os
import torch
import copy
from tqdm import tqdm
from torch import nn
import time

from client_node import ClientNode
from data_loader import Data_Loader
from model import MLP

import numpy as np
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(9)

class Trainer:
	def __init__(self, n_clients, learning_rate, lr_decay, batch_size, epochs, n_local_epochs, 
		data_size_file, w_dir, acc_dir, z_dir, train_fname, test_fname, input_dim, output_dim,
		pre_trained_w_file = None):
		
		self.learning_rate = learning_rate
		self.lr_decay = lr_decay
		self.batch_size = batch_size
		self.epochs = epochs
		self.data_loader = Data_Loader(train_fname, test_fname)

		self.n_clients = n_clients
		self.n_local_epochs = n_local_epochs

		self.train_data = None				#will be set in run_train
		self.test_data  = None				#will be set in run_train
		
	
		self.map_client_batch_range = None 	#will be set in run_train
		self.map_client_powerset = None		#will be set in run_train
		
		self.trainers = self.generate_trainers(2**self.n_clients, input_dim, output_dim)	
		
		self.w_dir = w_dir
		self.acc_dir = acc_dir
		self.z_dir = z_dir
	
		self.model_name = "MLP"
		self.data_size_file = data_size_file
	
		self.w_file = "CL_" + self.model_name + "lr" + str(self.learning_rate) + "_dc" + str(self.lr_decay)
		self.w_file = self.w_file + "_B" + str(batch_size) + "_L" + str(n_local_epochs) + "_G" + str(epochs * n_local_epochs) + "_" + data_size_file+  ".pth"
		self.acc_file = self.w_file + "acc.csv"
		self.sv_file = "exact_clSV_" + self.w_file + ".csv"
		
		print ("Configuration: ")
		print ("num_clients: ", n_clients)
		print("L, G: ", n_local_epochs, ", ", epochs)
		print("B, lr, lr_decay: ", batch_size, ", ", learning_rate, ", ", lr_decay)
		print("data_size_file: ", data_size_file)
		if (pre_trained_w_file != None):
			print("Loading weight from " + pre_trained_w_file)
			self.model.load_state_dict(torch.load(self.w_dir + pre_trained_w_file)) 


	def get_trainer(self, input_dim, output_dim):

		trainer = ClientNode(self.learning_rate, self.batch_size)
		trainer.model["model"] = MLP(input_dim, output_dim).to(device)
		return trainer
		
	def generate_trainers(self, n_trainers, input_dim, output_dim):
		trainers = []
		for i in range(n_trainers):
			trainers.append(self.get_trainer(input_dim, output_dim))
			#client_set = get_clients_from_index(i)
			#trainers[i].data = get_dataset_from_clients(client_set)
		return trainers

	def create_map_client_powerset(self):
		client_powerset = []
		for i in range (2**self.n_clients):
			client_list = []
			b = bin(i)
			len_b = len(b)
			for k in range(2, len_b):
				if b[k] == '1':
					client_list.append(len_b - k - 1)
			client_powerset.append(client_list)
		print("client_powerset: ", client_powerset)
		return client_powerset


	def train(self):

		
		w_path = self.w_dir 					#+ "FL/run" + str(self.run) + "/"
		acc_path = self.acc_dir 				# + "FL/run" + str(self.run) + "/"

		
		#acc file
		with open(acc_path + self.acc_file, 'w') as acc_file:
			acc_file.write("global_round,train_MSE,test_MSE,\n")

		
		#the shapley value file

		w_path = self.w_dir 						#+ "FL/run" + str(self.run) + "/"
		print("SV file: ", w_path + self.sv_file)
		#n_clients = len(self.clients) 				#len(self.p_matrix[0]) - 1

		if not os.path.isfile(w_path + self.sv_file):
			with open(w_path + self.sv_file, 'w') as sv_file:
				st = "global_round,time_consuming,"
				for cl in range(self.n_clients):
					st = st + "cl" + str(cl) 
					if cl < self.n_clients - 1:
						st = st + ","
				sv_file.write(st + "\n")

		print("Start training...")
		
		lr = self.learning_rate

		
		for epoch in range(self.epochs):
			print(f"Epoch {epoch+1}/{self.epochs}")
			print("center training ", 2**self.n_clients, " models to compute SVs")
			
			for i in range (1, 2**self.n_clients):
				for epoch1 in range(self.n_local_epochs):
					self.trainers[i].train(device, lr)
			
			lr = lr * self.lr_decay
			
				#gather a dataset according to s, d
				#train the model on d, say m with L iterations
				#save m


				#if we use memory to store models => 2^n models * size of each model (1K) = 2^n K => if n = 16 => 65M =>ok
				#we need a map from a number => list of clients and vice versa

			#compute SVs

			print("computing SVs...")
			start = time.time()
			SVs = self.get_SVs(ctype = 'exact', cmethod = 'subsets')
			stop = time.time()
			time_cconsuming = stop - start

			#begin to compute and save SVs
		
			with open(w_path + self.sv_file, 'a') as sv_file:
				st = str((epoch + 1) * self.n_local_epochs) + "," + str(time_cconsuming) + "," +  ",".join(["{:.7f}".format(SVs[i]) for i in range(self.n_clients)])
				sv_file.write(st + "\n")

			#print("SVs: ", SVs)
			#print("time_cconsuming in sec: ", time_cconsuming)
			
			#end of computing SVs

			
			# Validate new model
			train_loss = self.train_stats()
			print("Training statistic: ")
			print(" Loss: ", f'{train_loss:.3}')

			test_loss = self.validate(load_weight=False)			
			with open(acc_path + self.acc_file, 'a') as acc_file:
				acc_file.write(str((epoch + 1) * self.n_local_epochs) + "," + str(train_loss) + "," + str(test_loss) +  "\n")
            
		
		acc_file.close()
		sv_file.close()
		self.save_model()
		
	

	def train_stats(self):
		train_loss = 0
		last_index = 2**self.n_clients - 1
		self.trainers[last_index].model["model"].eval()
		loss = 0

		with torch.no_grad():
			for  (batch_X, batch_y) in self.train_data:
				batch_X, batch_y = batch_X.to(device), batch_y.to(device)
				output = self.trainers[last_index].model["model"](batch_X)
				loss += self.trainers[last_index].model["criterion"](torch.squeeze(output), batch_y)

		
		loss = loss.item()/len(self.train_data) 
		self.trainers[last_index].model["model"].train()
		return loss 
		
	def validate(self, load_weight=False):

		print("Validation statistic...")
		if load_weight == True:
			self.model.load_state_dict(torch.load(self.w_dir + self.w_file))

		train_loss = 0
		last_index = 2**self.n_clients - 1
		self.trainers[last_index].model["model"].eval()
		loss = 0

		with torch.no_grad():
			for  (batch_X, batch_y) in self.test_data:
				batch_X, batch_y = batch_X.to(device), batch_y.to(device)
				output = self.trainers[last_index].model["model"](batch_X)
				loss += self.trainers[last_index].model["criterion"](torch.squeeze(output), batch_y)

		
		loss = loss.item()/len(self.test_data) 
		self.trainers[last_index].model["model"].train()
		
		print(" Loss: ", f'{loss:.3}')
		print("-------------------------------------------")

		return loss



	
	def run_train(self):
		# load and distribute data to clients
		
		print("Seting training Configuration...")
		print("Creating data distribution for clients...")
		
		if (self.data_size_file != None):
			print("Data_size_file: ", self.data_size_file)
			ddf = pd.read_csv(self.z_dir + self.data_size_file, header = None, index_col = None)
			self.train_data = self.data_loader.get_train_data_in_batch(self.batch_size)
			self.map_client_batch_range = self.data_loader.map_in_batch_given_partition(self.n_clients, self.batch_size, ddf.to_numpy())
		else:
			print("No partition_data_file")
			return

		self.test_data = self.data_loader.get_test_data_in_batch(batch_size = self.batch_size)
		self.map_client_powerset = self.create_map_client_powerset()

		#for temporary using
		#self.train_data_in_batches = self.data_loader.get_data_in_batch(train = True, batch_size = self.batch_size)

		for i in range (1, 2**self.n_clients):
			print("subsets ", i)
			self.trainers[i].data = self.get_dataloader_from_clients(self.map_client_powerset[i])		
		
		#del self.train_data_in_batches[:]
		#del self.train_data_in_batches

		self.train()
	
	def get_dataloader_from_clients(self, s):
		range_list = []
		for client_i in s:
			range_list.append(self.map_client_batch_range[client_i])
			#for batch_index in self.map_client_batch_range[client_i]:
			#	batches.append(self.train_data_in_batches[batch_index])
		#print ("s = ", s)
		#print("len batches: ", len(batches))

		return self.data_loader.get_dataloader_given_batch_ranges(train = True, batch_size = self.batch_size, range_list = range_list)
		#return DataLoader(ConcatDataset(batches), shuffle=True, batch_size=batch_size)


	
	def get_SVs(self, ctype = 'exact', cmethod = 'subsets'):

		SVs = np.zeros(self.n_clients)
		m = self.n_clients

		if (ctype == 'exact') and (cmethod == 'subsets'):
			factorial_list = [1.0]
			f = 1.0
			for i in range (1, m+1):
				f *= i
				factorial_list.append(f)
			
			for i in range(self.n_clients):
				trainer_indices = self.get_trainer_indices_without_client_i(i)
				for v in trainer_indices:
					k = self.get_size_of_subset(v)
					u = v + 2**i 					# the index of the model trained on client i and a subset of other clients
					SVs[i] += factorial_list[k] * factorial_list[m - k - 1] * (self.V_by_model(self.trainers[u].model["model"], device) - self.V_by_model(self.trainers[v].model["model"], device)) 
			
		#print("SVs in get_SVs: ", SVs)
		return SVs

	def get_size_of_subset(self, k):

		count = 0
		n = k
		while n > 0:
			count += 1
			n = n & (n-1)
		return count	

	def get_trainer_indices_without_client_i(self, k):
		n_bits_before_k, n_bits_after_k = self.n_clients - k - 1, k
		list1, list2 = [], []

		for i in range(1, 2**n_bits_before_k, 1):
			list1.append(i * (2**(k+1)))

		for i in range(1, 2**n_bits_after_k, 1):
			list2.append(i)

		trainer_indices = list1 + list2
		
		for i in list1:
			for j in list2:
				trainer_indices.append(i + j)
		
		return trainer_indices
		

		

	def V_by_model(self, the_model, device):
		return self.get_loss_by_model(the_model, device)

	def get_loss_by_model(self, the_model, device):

		loss = 0

		with torch.no_grad():
			for  (batch_X, batch_y) in self.test_data:
				batch_X, batch_y = batch_X.to(device), batch_y.to(device)
				output = the_model(batch_X)
				loss += nn.MSELoss(reduction = 'mean')(torch.squeeze(output), batch_y)

		loss = loss.item()/len(self.test_data)
		#print ("n_batches of test_dataL ", len(self.test_data))
		return loss


	def save_model(self):
		print("Saving model...")
		if not os.path.exists(self.w_dir):
			os.makedirs(self.w_dir)
		torch.save(self.trainers[2**self.n_clients-1].model['model'].state_dict(), self.w_dir + self.w_file)
		print("Model saved!")



























