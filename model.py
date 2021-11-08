import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        #print ("before forwarding x = ", x)
        x = self.fc1(x)
        #print ("after forwarding x = ", x)
        return x
        