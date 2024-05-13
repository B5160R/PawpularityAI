import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the data
df = pd.read_csv("../../data/train.csv")

# Split the data into features and target variable
X = df.drop(['Id', 'Pawpularity'], axis=1)
y = df['Pawpularity']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions and evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
accuracy_score = model.score(X_test, y_test)

print ("Mean Squared Error: ", mse)
print ("Accuracy Score: ", accuracy_score)