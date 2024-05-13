import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class FeatureCNN(nn.Module):
	def __init__(self, num_classes):
		super(FeatureCNN, self).__init__()
		self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
		self.conv_layer2 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
		self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.fully_connected_layer_1 = nn.Linear(in_features=32*56*56, out_features=512)
		self.conv_layer3 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
		self.conv_layer4 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
		self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.fc1 = nn.Linear(1600, 128)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(128, num_classes)
  
	def forward(self, x):
		out = self.conv_layer1(x)
		out = self.conv_layer2(x)
		out = self.max_pool1(out)
  
		out = self.conv_layer3(x)
		out = self.conv_layer4(x)
		out = self.max_pool2(out)
		
		out = out.reshape(out.size(0), -1)

		out = self.fc1(out)
		out = self.relu1(out)
		out = self.fc2(out)

		return out

batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_transforms = transforms.Compose([
	transforms.Resize((32, 32)),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset

model = FeatureCNN(num_classes)

error = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


