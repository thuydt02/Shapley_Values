import random
import numpy as np
import torch

from torch import nn, optim

random.seed(9)

class ClientNode:
	def __init__(self, learning_rate, batch_size):
		self.model = {"model": None, "optim": None, "criterion": None, "loss": None}
		self.data = []
		self.learning_rate = learning_rate
		self.model["criterion"] = nn.MSELoss(reduction = 'mean') 
		self.batch_size = batch_size
	

	def clear_model(self):
		del self.model["model"]
		self.model["model"] = None


	def train(self, device, lr):
		
		self.model["optim"] = optim.SGD(self.model["model"].parameters(), lr=lr)
		
		for (batch_X, batch_y) in self.data:
			batch_X, batch_y = batch_X.to(device), batch_y.to(device)
			
			self.model["optim"].zero_grad()
			output = self.model["model"].forward(batch_X)

			loss = self.model["criterion"](torch.squeeze(output), batch_y)
			#print("output: ", torch.squeeze(output))
			#print("batch_y: ", batch_y)
			#print ("loss: ", loss)
			#print("loss.item: ", loss.item())
			loss.backward()
			self.model["optim"].step()
			
	def train_stats(self, device):

		self.model["model"].eval()
		loss = 0

		with torch.no_grad():
			for  batch_X, batch_y in self.data:
				batch_X, batch_y = batch_X.to(device), batch_y.to(device)
				output = self.model["model"](batch_X)
				loss += self.model["criterion"](torch.squeeze(output), batch_y)

		
		loss = loss.item()/len(self.data) 
		self.model["model"].train()
		return loss 


	def save_model(self, fname):
		torch.save(self.model["model"].state_dict(), fname)
		
