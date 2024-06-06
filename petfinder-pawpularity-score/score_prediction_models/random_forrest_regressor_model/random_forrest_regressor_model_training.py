import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score

# Load the data
df = pd.read_csv("../../data/pawpularity/train.csv")

# Split the data into features and target variable
X = df.drop(['Id', 'Pawpularity'], axis=1)
y = df['Pawpularity']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#save model
joblib.dump(model, "random_forest_regressor_model.pkl")

# Make predictions and evaluate the model
predictions = model.predict(X_test)

print("Predictions: ", predictions)
print("Actual: ", y_test.values)

mse = mean_squared_error(y_test, predictions)
r2_score = model.score(X_test, y_test)

print ("Mean Squared Error: ", mse)
print (f"R^2 Score: ", r2_score)

# Save the performance metrics
with open("performance_metrics.txt", "w") as f:
		f.write(f"Mean Squared Error: {mse}\n")
		f.write(f"R^2 Score: {r2_score}\n")
  

importances = model.feature_importances_
f_importances = pd.Series(importances, X.columns)
f_importances.sort_values(ascending=False, inplace=True)
f_importances.plot(x='Features', y='Importance', kind='bar', figsize=(16,9), rot=45, fontsize=15)

# Show the plot
plt.tight_layout()
plt.show()

# Save plot
plt.savefig('feature_importances.png')