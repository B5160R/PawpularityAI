import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

df = pd.read_csv("../../data/pawpularity/train.csv")
df = df.dropna()

# split data
X = df.drop(columns=["Id", "Human"])
y = df["Human"]

print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
	X, y, 
 test_size=0.2, 
 random_state=42
)

# train model with decision tree
base_estimator = DecisionTreeClassifier()
param_grid = {
  "criterion": ["gini", "entropy"],
  "max_depth": [None, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
  "min_samples_split": [2, 5, 10]
}

random_search_cv = RandomizedSearchCV(
  estimator=base_estimator, 
  param_distributions=param_grid, 
  n_iter=100, 
  cv=5, 
  verbose=2
)

random_search_cv.fit(X_train, y_train)

# performance evaluation
y_pred = random_search_cv.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cross_val_scores = cross_val_score(random_search_cv, X, y, cv=5)

print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)
print("Accuracy: ", accuracy)
print("Cross Validation Scores: ", cross_val_scores)

# show decision tree model image
plt.figure(figsize=(20, 20))
plot_tree(random_search_cv.best_estimator_, filled=True, feature_names=X.columns)
plt.savefig("decision_tree_model_rs.png")
  
# save model
joblib.dump(random_search_cv, "random_search_cv_is_human_decision_tree_model.pkl")

# save performance metrics
with open("performance_metrics.txt", "w") as f:
  f.write(f"Accuracy: {accuracy}\n")
  f.write(f"Precision: {precision}\n")
  f.write(f"Recall: {recall}\n")
  f.write(f"F1 Score: {f1}\n")
  f.write(f"Cross Validation Scores: {cross_val_scores}\n")
    
# save model variables
with open("random_search_cv_model_variables.txt", "w") as f:  
	f.write(f"Model Variables: {random_search_cv.get_params()}\n")

# plot feature importance
plt.figure(figsize=(20, 20))
plt.barh(X.columns, random_search_cv.best_estimator_.feature_importances_)
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.savefig("feature_importance.png")

# save decision tree model image
plt.figure(figsize=(20, 20))
plot_tree(random_search_cv.best_estimator_, filled=True, feature_names=X.columns)
plt.savefig("decision_tree_model.png")