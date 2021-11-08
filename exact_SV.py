import os
import torch
import copy
from itertools import combinations
from tqdm import tqdm

from torch import nn

import numpy as np

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(9)

#to compute exact_SV in FL environment

class exact_SV:
	def __init__(self, models, test_data):
		self.models = models 			#dict
		self.test_data = test_data		#in batches
	
	def by_subsets(self, device, V_pre):
        #models in a dictionary
        #return SVs in a dictionary

        #V_pre only for the case that models have one element, say client i. In learning process, client_i 's model is improved from 
        #the previous iteration of learning. The actual contribution of client i in this step is the difference: V(model_i) - V(the global_model_in_the_previous_step)
        #V_pre  = V(the global_model_in_the_previous_step)
        #exact_flSV has nothing with V_pre since it always has more than 2 models to compute SVs whenever #clients >= 2
        #exact_clustering_flSV can have the case that the set of clients is clustered into k clusters, and there is a cluster with one client.
        #In this case the models has only one model and V_pre will be used

		agents = self.models.keys()
		sv = dict((agent,0) for agent in agents)
		m = len(agents)
		#print ("m = ", m)
		if m == 1:
			for agent in agents:
				sv[agent] = self.V_by_agents(set([agent]), device) - V_pre
			return sv

		agent_set = set(agents)

		factorial_list = [1.0]
		f = 1.0
		for i in range (1, m+1):
			f *= i
			factorial_list.append(f)

		for agent_i in agent_set:
			
			no_agenti_set = agent_set - {agent_i}
			powerset = set()
			l = len(no_agenti_set)

			for k in range(1, l+1):
				data = combinations(no_agenti_set, k)
				k_element_sets = set(data)
				for S in k_element_sets:	
					sv[agent_i] += factorial_list[k] * factorial_list[m - k - 1] * (self.V_by_agents(set(S).union({agent_i}), device) - self.V_by_agents(set(S), device))
		
		for agent in agents:
			sv[agent] = sv[agent] / factorial_list[m]
		#print("sv by subsets: ", sv)
		return sv

	def SVi_by_subsets(self, device, i):
        #models in a dictionary
        #return SVs in a dictionary
		agents = self.models.keys()
		#print("SVi_by_subsets: client: ", i, ", cluster: ", agents)
		
		sv = 0
		m = len(agents)
        
        
		agent_set = set(agents)
		agent_i = i

		factorial_list = [1.0]
		f = 1.0
		for i in range (1, m+1):
			f *= i
			factorial_list.append(f)

		#for agent_i in agent_set:
		
		no_agenti_set = agent_set - {agent_i}
		powerset = set()
		l = len(no_agenti_set)

		for k in range(1, l+1):
			data = combinations(no_agenti_set, k)
			k_element_sets = set(data)
			for S in k_element_sets:
				#print("SVi_by_subsets: set(S).union{agent_i}: ", set(S).union({agent_i}))
				sv += factorial_list[k] * factorial_list[m - k - 1] * (self.V_by_agents(set(S).union({agent_i}), device) - self.V_by_agents(set(S), device))
	
		sv /= factorial_list[m]
		#print("SVi by subsets: i, sv = ", i, " ", sv)
		return sv
	
	def V_by_agents(self, S, device):
		#print ("S = ", S)
		#print ("models: ", self.models.keys())
		#print ("V_by_agents: S = ", S, ". models.keys() = ", self.models.keys())
		models = []

		for agent in S:
			models.append(self.models[agent])
		
		#n = len(models)
		#if n == 1:
		#	avged_model = models[0]
		#else:
		#	avged_model = self.average_models(models, [1/n for i in range(n)])
		#return self.get_loss_by_avg_model(avged_model, device) #0 for loss, 1 for acc 
		return self.V_by_models(models, device)

	def V_by_models(self, models, device):
		n = len(models)
		if n == 1:
			avged_model = models[0]
		else:
			avged_model = self.average_models(models, [1/n for i in range(n)])
		#return 13000 - self.get_loss_by_avg_model(avged_model, device)
		return self.get_loss_by_avg_model(avged_model, device) #0 for loss, 1 for acc 

	def average_models(self, models, p):
		averaged_model = copy.deepcopy(models[0])

		sum_p = sum(p)
		with torch.no_grad():
			averaged_values = {}
			for name, param in averaged_model.named_parameters():
				averaged_values[name] = nn.Parameter(torch.zeros_like(param.data))

			for i, model in enumerate(models):
				for name, param in model.named_parameters():
					averaged_values[name] += p[i] * param.data

			for name, param in averaged_model.named_parameters():
				param.data = averaged_values[name] / sum_p
		return averaged_model

	def get_loss_by_avg_model(self, avged_model, device):

		loss = 0

		with torch.no_grad():
			for  (batch_X, batch_y) in self.test_data:
				batch_X, batch_y = batch_X.to(device), batch_y.to(device)
				output = avged_model(batch_X)
				loss += nn.MSELoss(reduction = 'mean')(torch.squeeze(output), batch_y)

		loss = loss.item()/len(self.test_data)
		#print ("n_batches of test_dataL ", len(self.test_data))
		return loss


