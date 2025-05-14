
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch
import math
import torch
# model imports
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
import numpy as np
from PIL import Image
import argparse




class Double_Q_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Double_Q_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]

		self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
		self.Q2 = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		q1 = self.Q1(s)
		q2 = self.Q2(s)
		return q1,q2


class Policy_Net(nn.Module):
	def __init__(self, state_dim, action_dim, hid_shape):
		super(Policy_Net, self).__init__()
		layers = [state_dim] + list(hid_shape) + [action_dim]
		self.P = build_net(layers, nn.ReLU, nn.Identity)

	def forward(self, s):
		logits = self.P(s)
		probs = F.softmax(logits, dim=1)
		return probs

def build_net(layer_shape, hid_activation, output_activation):
	'''build net with for loop'''
	layers = []
	for j in range(len(layer_shape)-1):
		act = hid_activation if j < len(layer_shape)-2 else output_activation
		layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
	return nn.Sequential(*layers)