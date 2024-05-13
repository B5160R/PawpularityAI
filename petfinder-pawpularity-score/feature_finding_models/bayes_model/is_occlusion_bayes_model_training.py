import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("../../data/pawpularity/train.csv")
df = df.dropna()

# split data
X = df.drop(columns=["Id", "Occlusion"])
y = df["Occlusion"]

X_train, X_test, y_train, y_test = train_test_split(
	X, y, 
 test_size=0.2, 
 random_state=42
)

# train model with naive bayes
model = GaussianNB(var_smoothing=0.000001)
model.fit(X_train, y_train)

# performance evaluation
accuracy = model.score(X_test, y_test)

print("Accuracy: ", accuracy)

# save model
joblib.dump(model, "occlusion_bayes_model.pkl")

# save performance metrics
with open("performance_metrics.txt", "w") as f:
	f.write(f"Accuracy: {accuracy}\n")

# save model variables
with open("model_variables.txt", "w") as f:
	f.write(f"Model Variables: {model.get_params()}\n")