import os
import torch
import copy
from tqdm import tqdm
from torch import nn
import time

from client_node import ClientNode
from data_loader import Data_Loader
from model import MLP
from exact_SV import exact_SV
from graph import graph_interface
import numpy as np
import pandas as pd

from sklearn.cluster import SpectralClustering

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(9)

class Trainer:
	def __init__(self, n_clients, learning_rate, lr_decay, batch_size, epochs, n_local_epochs, cl_percentages, n_clusters, 
		data_size_file, w_dir, acc_dir, z_dir, train_fname, test_fname, input_dim, output_dim,
		pre_trained_w_file = None):
		
		self.learning_rate = learning_rate
		self.lr_decay = lr_decay
		self.batch_size = batch_size
		self.epochs = epochs
		self.data_loader = Data_Loader(train_fname, test_fname)

		self.clients = self.generate_clients(n_clients)
		self.n_local_epochs = n_local_epochs
		self.cl_percentages = cl_percentages
		self.n_clusters = n_clusters

		self.train_data = None
		self.test_data  = None

		#self.model_name = ""
		#self.model = self.get_model()# MLP2().to(device) #MNIST().to(device) #CNNModel().to(device) MNIST FEMNIST
		#summary(self.model, (1, 28, 28))
		
		self.selected_cls_mask = []
		
		self.get_model(input_dim, output_dim)
		self.data_size_file = data_size_file
		
		self.w_dir = w_dir
		self.acc_dir = acc_dir
		self.z_dir = z_dir
		
		self.w_file = "FL_" + self.model_name + "lr" + str(self.learning_rate) + "_dc" + str(self.lr_decay)
		self.w_file = self.w_file + "_B" + str(batch_size) + "_L" + str(n_local_epochs) + "_G" + str(epochs * n_local_epochs) + "_" + data_size_file+  ".pth"
		self.acc_file = self.w_file + "acc.csv"
		self.participant_file = "participants" + str(int(self.cl_percentages * 100)) +"_"+ self.data_size_file
		self.sv_file = "exact_clustering_flSV_" + self.w_file + ".csv"

		print ("Configuration: ")
		print ("num_clients: ", n_clients)
		print("L, G: ", n_local_epochs, ", ", epochs)
		print("B, lr, lr_decay: ", batch_size, ", ", learning_rate, ", ", lr_decay)
		print("data_size_file: ", data_size_file)
		if (pre_trained_w_file != None):
			print("Loading weight from " + pre_trained_w_file)
			self.model.load_state_dict(torch.load(self.w_dir + pre_trained_w_file)) 


	def get_model(self, input_dim, output_dim):

		
			self.model = MLP(input_dim, output_dim).to(device)
			self.model_name = "MLP"
			#summary(self.model, (1, 28, 28))

		
	def generate_clients(self, n_clients):
		clients = [] 
		for i in range(n_clients):
			client = ClientNode(self.learning_rate, self.batch_size)
			clients.append(client)

		return clients


	def send_model_to_clients(self, selected_cls ):
		for cl in selected_cls:
			self.clients[cl].model["model"] = copy.deepcopy(self.model)
		
	def average_models(self, models, coefficients):
		averaged_model = copy.deepcopy(self.model)
		p = np.asarray(coefficients)/sum(coefficients)

		with torch.no_grad():
			averaged_values = {}
			for name, param in averaged_model.named_parameters():
				averaged_values[name] = nn.Parameter(torch.zeros_like(param.data))
			i = 0
			for model in models:
				for name, param in model.named_parameters():
					averaged_values[name] += p[i] * param.data
				i += 1

			for name, param in averaged_model.named_parameters():
				param.data = averaged_values[name]

		return averaged_model
	
	
	def SV_in_train(self):

		
		w_path = self.w_dir 					#+ "FL/run" + str(self.run) + "/"
		acc_path = self.acc_dir 				# + "FL/run" + str(self.run) + "/"

		coefficients = np.asarray([len(cl.data) for cl in self.clients]) # for weights in the aggregation of models

		#acc file
		with open(acc_path + self.acc_file, 'w') as acc_file:
			acc_file.write("global_round,train_MSE,test_MSE,\n")

		#participant file
		with open(w_path + self.participant_file, 'w') as participant_file:
			st = "global_round,"
			for cl in range(len(self.clients)):
				st = st + "cl" + str(cl) 
				if cl < len(self.clients) - 1:
					st = st + ","
			participant_file.write(st + "\n")
        
		#the shapley value file

		w_path = self.w_dir 						#+ "FL/run" + str(self.run) + "/"
		print("SV file: ", w_path + self.sv_file)
		n_clients = len(self.clients) 				#len(self.p_matrix[0]) - 1

		if not os.path.isfile(w_path + self.sv_file):
			with open(w_path + self.sv_file, 'w') as sv_file:
				st = "global_round,time_consuming,"
				for cl in range(n_clients):
					st = st + "cl" + str(cl) 
					if cl < n_clients - 1:
						st = st + ","
				sv_file.write(st + "\n")

		print("Start training...")
		
		lr = self.learning_rate

		m = int(len(self.clients) * self.cl_percentages)  # for 2 random indices
		print("#selected clients: ", m, "/", len(self.clients))
		self.selected_cls_mask = [0 for _ in range(len(self.clients))]

		V_pre = self.validate(load_weight=False)
			
		for epoch in range(self.epochs):
			print(f"Epoch {epoch+1}/{self.epochs}")
			# Send model to all clients
			
			selected_cls, selected_cls_mask = self.select_clients(m)

			print("selected_cls: ", selected_cls)

			self.selected_cls_mask = np.logical_or(self.selected_cls_mask, selected_cls_mask).astype(int)
			self.send_model_to_clients(selected_cls)

			# Update local model for several epochs
			
			print("Local updating on clients")
			for epoch1 in range(self.n_local_epochs):
				for i in selected_cls:
					self.clients[i].train(device, lr)
			lr = lr * self.lr_decay
			
			#saving participant file
			
			with open(w_path + self.participant_file, 'a') as participant_file:
				st = ",".join(map(str, selected_cls_mask)) + "\n"
				participant_file.write(str((epoch + 1) * self.n_local_epochs) + "," + st)

			
			#begin to compute and save SVs
			
			start = time.time()
			#SVs = self.get_SVs(ctype = 'exact', cmethod = 'subsets', selected_cls = selected_cls)
			#stop = time.time()
			#time_cconsuming = stop - start
			#with open(w_path + self.sv_file, 'a') as sv_file:
			#	st = str((epoch + 1) * self.n_local_epochs) + "," + str(time_cconsuming) + "," +  ",".join(["{:.7f}".format(SVs[i]) for i in range(n_clients)])
			#	sv_file.write(st + "\n")

			#print("SVs: ", SVs)
			#print("time_cconsuming in sec: ", time_cconsuming)
			
			#end of computing SVs

			#we need to cluster the clients into n clusters
			#compute 
			clusters = self.get_clusters(selected_cls, self.n_clusters)
			print("clusters in training: ", clusters)

			SVs = np.zeros(len(self.clients))
			#computing in_cluster SVs
			
			for k in range(self.n_clusters):
				svs = self.get_SVs(ctype = 'exact', cmethod = 'subsets', selected_cls = clusters[k], V_pre = V_pre)
				SVs += svs

			#computing out_cluster SVs
			SVs1 = np.zeros(len(self.clients))
			for i in selected_cls:
				for k in range(self.n_clusters):
					if i not in clusters[k]:
						SVs1[i] += self.get_SVi(ctype = 'exact', cmethod = 'subsets', client_i = i, cluster = clusters[k] + [i])

			SVs += SVs1

			stop = time.time()
			time_cconsuming = stop - start
			with open(w_path + self.sv_file, 'a') as sv_file:
				st = str((epoch + 1) * self.n_local_epochs) + "," + str(time_cconsuming) + "," +  ",".join(["{:.7f}".format(SVs[i]) for i in range(n_clients)])
				sv_file.write(st + "\n")

			print("SVs: ", SVs)
			print("time_cconsuming in sec: ", time_cconsuming)
		



			# Get back and average the local models
			client_models = [self.clients[i].model["model"] for i in selected_cls]
			self.model = self.average_models(client_models, coefficients)

			# Validate new model
			train_loss = self.train_stats()
			print("Training statistic: ")
			print(" Loss: ", f'{train_loss:.3}')

			test_loss = self.validate(load_weight=False)			
			with open(acc_path + self.acc_file, 'a') as acc_file:
				acc_file.write(str((epoch + 1) * self.n_local_epochs) + "," + str(train_loss) + "," + str(test_loss) +  "\n")
			V_pre = test_loss
		
		print("selected clients in training: ")
		print(self.selected_cls_mask)

		acc_file.close()
		participant_file.close()
		sv_file.close()
		self.save_model()
		
	

	def train_stats(self):
		train_loss = 0
		s = sum(np.asarray([len(cl.data) for cl in self.clients]))

		p = [len(cl.data)/s for cl in self.clients]
		
		for i, cl in enumerate(self.clients):
			if self.selected_cls_mask[i] == 0:
				continue
			loss = cl.train_stats(device)
			train_loss += p[i] * loss
			

		return train_loss

	def validate(self, load_weight=False):

		print("Validation statistic...")
		if load_weight == True:
			self.model.load_state_dict(torch.load(self.w_dir + self.w_file))

		self.model.eval()
		loss = 0
		test_data = self.test_data

		with torch.no_grad():
			for  (batch_X, batch_y) in test_data:
				batch_X, batch_y = batch_X.to(device), batch_y.to(device)
				output = self.model(batch_X)
				loss += nn.MSELoss(reduction = 'mean')(torch.squeeze(output), batch_y)

		
		loss = loss.item()/len(test_data)

		print(" Loss: ", f'{loss:.3}')
		print("-------------------------------------------")

		return loss


	def save_model(self):
		print("Saving model...")
		if not os.path.exists(self.w_dir):
			os.makedirs(self.w_dir)
		torch.save(self.model.state_dict(), self.w_dir + self.w_file)
		print("Model saved!")


	
	def run_train(self):
		# load and distribute data to clients
		print("Creating data distribution for clients...")
		if (self.data_size_file != None):
			print("Data_size_file: ", self.data_size_file)
			ddf = pd.read_csv(self.z_dir + self.data_size_file, header = None, index_col = None)
			self.train_data = self.data_loader.distribute_in_batch_given_partition(len(self.clients), self.batch_size, ddf.to_numpy())
		else:
			print("No partition_data_file")
			return

		self.test_data = self.data_loader.get_test_data_in_batch(batch_size = self.batch_size)

		print("Distributing data...")
		for client_id, client_data in tqdm(self.train_data.items()):
			self.clients[client_id].data = client_data

		
		self.SV_in_train()

	def select_clients(self, m):
		selected_cls = np.random.choice(len(self.clients), size = m, replace = False)
		mask = [0 for _ in range(len(self.clients))]
		for cl in selected_cls:
			mask[cl] = 1
		return (selected_cls, mask)

	def get_SVs(self, ctype = 'exact', cmethod = 'subsets', selected_cls = [], V_pre = 0):

		SVs = np.zeros(len(self.clients))
		if (ctype == 'exact') and (cmethod == 'subsets'):
			
			models = dict((i, self.clients[i].model["model"]) for i in selected_cls)
			eSV = exact_SV(models, self.test_data)

			SV1 = eSV.by_subsets(device, V_pre) #in dictionary
			for ag in SV1.keys():
				SVs[ag] = SV1[ag]

			
		#print("get_SVs: SVs = ", SVs)
		return SVs

	def get_SVi(self, ctype = 'exact', cmethod = 'subsets', client_i = 0, cluster = []):
		#print("get_SVi: client, cluster= ", client_i, " ", cluster)
		ret = 0
		if (ctype == 'exact') and (cmethod == 'subsets'):
			
			models = dict((i, self.clients[i].model["model"]) for i in cluster)
			models[client_i] = self.clients[client_i].model["model"]

			eSV = exact_SV(models, self.test_data)
			ret = eSV.SVi_by_subsets(device, client_i)

			#return ret
		#print("SVi = ", ret)
		return ret
	
	def get_clusters(self, selected_cls, n_clusters):

		models = dict((i, self.clients[i].model["model"]) for i in selected_cls)

		sim = self.get_similarity_matrix(models, device)

		#print("similarity matrix: ")
		#print(sim)

		G = graph_interface()
		sc = G.spectralClustering_balance(sim_matrix = sim, n_clusters = n_clusters)
		
		clusters = [[] for _ in range(n_clusters)]


		#print("get_clusters: ", sc.labels_)
		for i, agent in enumerate(selected_cls):
			clusters[sc[i]].append(agent)
		
		#return [[0, 1, 2, 3], [4, 5]]
		
		return clusters

	def get_similarity_matrix(self, models, device):
        #Soufiani paper
        
        #print ("in get_sim_mat...")
		agents = models.keys()
		n_agents = len(agents)
		sim = np.zeros((n_agents, n_agents))
		v = np.zeros((n_agents, n_agents))

		eSV = exact_SV(models, self.test_data)
	
		for i, agent_i in enumerate(agents):
			v[i][i] = eSV.V_by_models([models[agent_i]], device)

		for i, agent_i in enumerate(agents):
			for j, agent_j in enumerate(agents):
				if j <= i: 
					continue
				v[i][j] = eSV.V_by_models([models[agent_i], models[agent_j]], device)

		count_zeros = 0
		for i in range(n_agents):
			for j in range(i , n_agents, 1):
				if (v[i][i] + v[j][j] - v[i][j]>0): 
					sim[i][j] = v[i][i] + v[j][j] - v[i][j]
					sim[j][i] = sim[i][j]
				else:
					count_zeros += 1

		#print ("get_sim_matrix: count_zeros: ", count_zeros)
		#print("get_sim_matrix: min, max: ", sim.min(), sim.max())
		delta = (sim.max() - sim.min()) * 0.1
		return np.exp(- sim ** 2 / (2. * delta ** 2))
		#return sim




























