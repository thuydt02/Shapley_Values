import os
import torch
import copy
from itertools import combinations
from tqdm import tqdm

from torch import nn

import numpy as np
from scipy.sparse.linalg import eigsh
from k_means_constrained import KMeansConstrained

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(9)

class graph_interface:
	def __init__(self):
		return
	
	def laplacian(self, weighted_matrix):
		n = len(weighted_matrix)
		D = np.zeros((n,n))
		for i in range(n):
			D[i][i] = sum(weighted_matrix[i, :])
		return D - weighted_matrix
	
	def spectralClustering_balance(self, sim_matrix, n_clusters = 2):
		#assume that sim_matrix is symmetric and does not contain negative entries.
		#the graph associated with this sim_matrix is fully connected, so the sim_matrix is the weighted graph.

		n = len(sim_matrix)
		L = self.laplacian(sim_matrix)
		eigenvalues, eigenvectors = eigsh(L, k = n_clusters)
		
		clf = KMeansConstrained(n_clusters = n_clusters, size_min = int(n / n_clusters), size_max = int(n / n_clusters) + 1, random_state=9)
		clf.fit_predict(eigenvectors)
		return clf.labels_

