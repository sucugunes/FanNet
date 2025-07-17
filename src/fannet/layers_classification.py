import sys
import os
import random
import scipy
import scipy.sparse.linalg as sla
import numpy as np
import torch
import torch.nn as nn
from .utils import toNP
from .geometry import to_basis, from_basis

import torch.nn.functional as F

from fannet.fanconv import FanConv
        
class Net(torch.nn.Module):
    def __init__(self, in_channels, num_classes, seq_length):    
        super(Net, self).__init__()

        #original network
        self.fc0 = nn.Linear(in_channels, 128)
        self.conv1 = FanConv(128, 256, seq_length)
        self.conv2 = FanConv(256, 256, seq_length)
        self.conv3 = FanConv(256, 256, seq_length)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        nn.init.xavier_uniform_(self.fc0.weight, gain=1)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)
        nn.init.constant_(self.fc0.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x, indices, mass):

        x = x.unsqueeze(0)
        mass = mass.unsqueeze(0)
        indices = indices.unsqueeze(0)

        x = F.elu(self.fc0(x))
        x = F.elu(self.conv1(x, indices))
        x = F.elu(self.conv2(x, indices))
        x = F.elu(self.conv3(x, indices))

        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        x = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)
        
        x = x.squeeze(0)
  
        return F.log_softmax(x, dim=-1)
