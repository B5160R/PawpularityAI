import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("../../data/pawpularity/train.csv")
df = df.dropna()

# split data
X = df.drop(columns=["Id", "Human"])
y = df["Human"]

X_train, X_test, y_train, y_test = train_test_split(
	X, y, 
 test_size=0.2, 
 random_state=42
)

# train model with decision tree
base_estimator = DecisionTreeClassifier(
	criterion="gini",
	max_depth=10
)
	

model_ada = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=42)
model_ada.fit(X_train, y_train)

# performance evaluation
y_pred = model_ada.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("Accuracy: ", accuracy)

# save model
joblib.dump(model_ada, "is_human_decision_tree_model_boost.pkl")

# save performance metrics
with open("performance_metrics.txt", "w") as f:
	f.write(f"Accuracy: {accuracy}\n")
 
# save model variables
with open("model_variables.txt", "w") as f:
	f.write(f"Model Variables: {model_ada.get_params()}\n")

# plot feature importance