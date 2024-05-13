import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from CatOrDogCNN import CatOrDogCNN

# Load the data

trainData = ImageFolder(root='data/train', transform=transforms.Compose([
		transforms.Resize((64, 64)),
		transforms.ToTensor()
]))

testData = ImageFolder(root='data/test', transform=transforms.Compose([
		transforms.Resize((64, 64)),
		transforms.ToTensor()
]))

trainLoader = DataLoader(trainData, batch_size=64, shuffle=True)
testLoader = DataLoader(testData, batch_size=64, shuffle=False)

model = CatOrDogCNN(num_classes=2)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model

losses = []
print('Starting training')
for epoch in range(10):
	for i, (images, labels) in enumerate(trainLoader):
		optimizer.zero_grad()
		outputs = model(images)
		loss = loss_fn(outputs, labels)
		loss.backward()
		optimizer.step()
		losses.append(loss.item())
		print(f'Epoch {epoch+1}, Iteration {i+1}, Loss: {loss.item()}')
print('Finished training')

plt.plot(losses)
plt.show()

# Test the model

correct = 0
total = 0

with torch.no_grad():
	for images, labels in testLoader:
		outputs = model(images)
		_, predicted = torch.max(outputs, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')

# save the model

torch.save(model.state_dict(), 'cat_or_dog_model.pth')