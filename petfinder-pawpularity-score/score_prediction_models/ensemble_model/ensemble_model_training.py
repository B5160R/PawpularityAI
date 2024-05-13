import joblib
import pandas as pd
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("../../data/pawpularity/train.csv")

# split data
X = df.drop(columns=["Id", "Pawpularity"])
y = df["Pawpularity"]

X_train, X_test, y_train, y_test = train_test_split(
  X, y, 
  test_size=0.2, 
  random_state=42
)

decision_tree = DecisionTreeClassifier()
logistic = LogisticRegression()

# Define meta-learner (replace with your preferred model)
meta_learner = LogisticRegression()

# Bagging ensemble
bagging_ensemble = BaggingClassifier(base_estimator=decision_tree, n_estimators=10)

# Boosting ensemble
boosting_ensemble = AdaBoostClassifier(base_estimator=logistic, n_estimators=20)

ensemble = StackingClassifier(estimators=[('bagging', bagging_ensemble), ('boosting', boosting_ensemble)], 
                             final_estimator=meta_learner)

# Train the ensemble
ensemble.fit(X_train, y_train)

# Make predictions on new data
predictions = ensemble.predict(X_test)

# Performance evaluation
print("Predictions: ", predictions)
print("Actual: ", y_test.values)
print("Model Accuracy: ", ensemble.score(X_test, y_test))

# Save the model
joblib.dump(ensemble, "ensemble_model.pkl")

# Save performance metrics
with open("performance_metrics.txt", "w") as f:
	f.write(f"Model Accuracy: {ensemble.score(X_test, y_test)}\n")
	f.write(f"Model Predictions: {predictions}\n")
	f.write(f"Actual: {y_test.values}\n")

# Save model variables
with open("model_variables.txt", "w") as f:
	f.write(f"Model Variables: {ensemble.get_params()}\n")