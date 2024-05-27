import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, mean_absolute_error, precision_score, r2_score, accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

df = pd.read_csv("../../data/pawpularity/train.csv")

# split data
X = df.drop(columns=["Id", "Pawpularity"])
y = df["Pawpularity"]

X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
)

models = [
  ("logistic", LogisticRegression()),
  ("knn", KNeighborsClassifier()),
  ("dt", DecisionTreeClassifier()),
  ("svm", SVC()),
  ("nb", GaussianNB())
]

accuracy = []

for model in models:
  model[1].fit(X_train, y_train)
  y_pred = model[1].predict(X_test)
  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  accuracy.append(accuracy_score(y_test, y_pred))
  
plt.figure(figsize=(10, 5))
model_names = [x[0] for x in models] 
y_pos = range(len(models))
plt.bar(y_pos, accuracy, align='center', alpha=0.5)
plt.xticks(y_pos, [x[0] for x in models], rotation=45)
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()

stacked_model = StackingClassifier(estimators=models, final_estimator=LogisticRegression())
stacked_model.fit(X_train, y_train)

# save model
joblib.dump(stacked_model, "stacked_classifier_model.pkl")

y_pred = stacked_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy_sc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# compare visually the actual and predicted values
print("Actual: ", y_test.values[:5])
print("Predicted: ", y_pred[:5])
print("--------------------")
print("Mean Absolute Error: ", mae)
print("R2 Score: ", r2)
print("Accuracy Score: ", accuracy_sc)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1 Score: ", f1)

# save performance metrics
with open("performance_metrics.txt", "w") as f:
  f.write(f"Mean Absolute Error: {mae}\n")
  f.write(f"R2 Score: {r2}\n")
  f.write(f"Accuracy Score: {accuracy_sc}\n")
  f.write(f"Precision: {precision}\n")
  f.write(f"Recall: {recall}\n")
  f.write(f"F1 Score: {f1}\n")