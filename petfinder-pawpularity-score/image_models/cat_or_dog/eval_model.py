import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from CatOrDogCNN import CatOrDogCNN
import matplotlib.pyplot as plt

# Load the data
testData = ImageFolder(root='../../data/cat_or_dog/test', transform=transforms.Compose([
		transforms.Resize((64, 64)),
		transforms.ToTensor()
]))

testLoader = DataLoader(testData, batch_size=64, shuffle=True)

model = CatOrDogCNN(num_classes=2)
model.load_state_dict(torch.load("cat_or_dog_model.pth"))
model.eval()

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

# Visualize the model's predictions
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
model.eval()

for i, (images, labels) in enumerate(testLoader):
		outputs = model(images)
		_, predicted = torch.max(outputs, 1)
		for j in range(10):
				ax = axes[j // 5, j % 5]
				ax.imshow(images[j].permute(1, 2, 0))
				ax.set_title(f'Predicted: {"Cat" if predicted[j] == 0 else "Dog"}\nActual: {"Cat" if labels[j] == 0 else "Dog"}')
				ax.axis('off')
		break

plt.tight_layout()
plt.show()