import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Perceptron import Perceptron

#load the data
df = pd.read_csv("../../data/pawpularity/train.csv")

#split the data into features and target variable
X = df.drop(['Id', 'Pawpularity'], axis=1)
X = X.values
y = df['Pawpularity']
y = y.values

#split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train the model
ppn = Perceptron(num_features=X_train.shape[1])

print("training the model...")
ppn.train(all_x=X_train, all_y=y_train, epochs=20)
print("\nWeights:", ppn.weights)

#make predictions and evaluate the model
predictions = [ppn.forward(x) for x in X_test]
accuracy = sum(predictions == y_test) / len(y_test)
print("\nAccuracy:", accuracy)

#save the model
import joblib
joblib.dump(ppn, "perceptron_model.pkl")
