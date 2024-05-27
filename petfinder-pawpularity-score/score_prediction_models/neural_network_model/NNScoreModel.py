import torch
import torch.nn as nn

class NNScoreModel(nn.Module):
	def __init__(self, input_size):
		super(NNScoreModel, self).__init__()
		self.layer1 = nn.Linear(input_size, 64)
		self.layer2 = nn.Linear(64, 64)
		self.layer3 = nn.Linear(64, 32)
		self.layer4 = nn.Linear(32, 32)
		self.layer5 = nn.Linear(32, 1)

	def forward(self, x):
		x = torch.relu(self.layer1(x))
		x = torch.relu(self.layer2(x))
		x = torch.relu(self.layer3(x))
		x = torch.relu(self.layer4(x))
		x = self.layer5(x)
		return x