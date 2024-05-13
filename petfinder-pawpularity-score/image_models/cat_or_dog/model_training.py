# use CNN to determine if the image is a cat or a dog using torchvision

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Load the data
trainData = ImageFolder(root='data/train', transform=transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor()
]))
testData = ImageFolder(root='data/test', transform=transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor()
]))

# Split the data into training and testing sets
trainLoader = DataLoader(trainData, batch_size=64, shuffle=True)
testLoader = DataLoader(testData, batch_size=64, shuffle=False)

# Define the model
model = torchvision.models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)

# Define the loss function
loss_fn = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
losses = []
for epoch in range(10):
	for i, (images, labels) in enumerate(trainLoader):
		optimizer.zero_grad()
		outputs = model(images)
		loss = loss_fn(outputs, labels)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())
		print(f'Epoch {epoch+1}, Iteration {i+1}, Loss: {loss.item()}')
  
# Test the model
correct = 0
total = 0
with torch.no_grad():
	for images, labels in testLoader:
		outputs = model(images)
		_, predicted = torch.max(outputs, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print(f'Accuracy: {correct / total}')

# Save the model
torch.save(model.state_dict(), 'model.pth')