from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn

# Load the data
df = pd.read_csv("../../data/train.csv")

# convert the data to PyTorch tensors
X = torch.tensor(df.drop(columns=["Id", "Pawpularity"]).values, dtype=torch.float32)
y = torch.tensor(df["Pawpularity"].values, dtype=torch.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the model
model = nn.Sequential(
	nn.Linear(X_train.shape[1], 64),
	nn.ReLU(),
	nn.Linear(64, 64),
	nn.ReLU(),
	nn.Linear(64, 32),
	nn.ReLU(),
	nn.Linear(32, 32),
	nn.ReLU(),
	nn.Linear(32, 1)
)

# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

# Train the model
losses = []
for epoch in range(100):
	y_pred = model(X_train)
	loss = loss_fn(y_pred, y_train)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	print(f"Epoch {epoch+1}, Loss: {loss.item()}")
	losses.append(loss.item())

plt.plot(losses)

# evaluate the model
y_pred = model(X_test)
y_pred = y_pred.detach().numpy()
y_pred = y_pred.argmax(axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# save evaluation metrics
with open("performance_metrics.txt", "w") as f:
	f.write(f"Accuracy: {accuracy}\n")

# Save the model
torch.save(model.state_dict(), "neural_network_model.pth")
  