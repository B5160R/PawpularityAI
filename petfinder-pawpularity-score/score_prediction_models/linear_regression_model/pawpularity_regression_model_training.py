import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("../../data/pawpularity/train.csv")

# split data
X = df.drop(columns=["Id", "Pawpularity"])
y = df["Pawpularity"]

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
)

# train model with multiple dimensional linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# visualize the model
print("--------------------")
print("Model Variables: ", model.get_params())
print("Model Coefficients: ", model.coef_)
print("Model Intercept: ", model.intercept_)
print("--------------------")

# performance evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# compare visually the actual and predicted values
print("Actual: ", y_test.values[:5])
print("Predicted: ", y_pred[:5])
print("--------------------")
print("Mean Absolute Error: ", mae)
print("R2 Score: ", r2)
print("--------------------")

# save model
joblib.dump(model, "pawpularity_regression_model.pkl")

# save performance metrics
with open("performance_metrics.txt", "w") as f:
  f.write(f"Mean Absolute Error: {mae}\n")
  f.write(f"R2 Score: {r2}\n")

# save model variables
with open("model_variables.txt", "w") as f:
  f.write(f"Model Variables: {model.get_params()}\n")
  f.write(f"Model Coefficients: {model.coef_}\n")
  f.write(f"Model Intercept: {model.intercept_}\n")