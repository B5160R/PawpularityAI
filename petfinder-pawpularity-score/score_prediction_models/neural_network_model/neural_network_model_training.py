import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from NNScoreModel import NNScoreModel

# Load the data
df = pd.read_csv("../../data/pawpularity/train.csv")

# convert the data to PyTorch tensors
X = torch.tensor(df.drop(columns=["Id", "Pawpularity"]).values, dtype=torch.float32)
y = torch.tensor(df["Pawpularity"].values, dtype=torch.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define the model
model = NNScoreModel(input_size=X_train.shape[1])

# Define the loss function
loss_fn = nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 

# Train the model
print("Training the model...")

losses = []
for epoch in range(200):
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

y_pred = model(X_test)
y_pred = y_pred.detach().numpy().argmax(axis=1)
cm = confusion_matrix(y_test, y_pred)
print(cm)

precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# save evaluation metrics
with open("performance_metrics.txt", "w") as f:
	f.write(f"Accuracy: {accuracy}\n Precision: {precision}\n Recall: {recall}\n F1 Score: {f1}\n Confusion Matrix: {cm}")

# Save the model
torch.save(model.state_dict(), "neural_network_model.pth")