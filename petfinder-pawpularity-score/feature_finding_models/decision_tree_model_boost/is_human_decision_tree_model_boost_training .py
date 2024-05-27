import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

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

# save model
joblib.dump(model_ada, "is_human_decision_tree_model_boost.pkl")

# performance evaluation
y_pred = model_ada.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
crossval_scores = cross_val_score(model_ada, X, y, cv=5)

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("Accuracy: ", accuracy)
print("Cross Validation Scores: ", crossval_scores)
# save performance metrics
with open("performance_metrics.txt", "w") as f:
	f.write(f"Precision: {precision}\n")
	f.write(f"Recall: {recall}\n")
	f.write(f"F1 Score: {f1}\n")
	f.write(f"Accuracy: {accuracy}\n")
	f.write(f"Cross Validation Scores: {crossval_scores}\n")
 
# save model variables
with open("model_variables.txt", "w") as f:
	f.write(f"Model Variables: {model_ada.get_params()}\n")

# plot feature importance
plt.figure(figsize=(20, 20))
plt.barh(X.columns, model_ada.feature_importances_)
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.savefig("feature_importance.png")

# plot decision tree
plt.figure(figsize=(20, 20))
plot_tree(model_ada.estimators_[0], filled=True, feature_names=X.columns)
plt.savefig("decision_tree_model_boost.png")