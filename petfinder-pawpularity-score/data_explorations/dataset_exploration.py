import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv("../data/pawpularity/train.csv")

print("\n*** Number of rows and columns in the dataset ***")
print(df.shape)

print("\n*** First few rows of the dataset ***")
print(df.head())

print("\n*** Missing values are in the dataset ***")
print(df.isnull().sum())

print("\n*** Sum of each column ***")
print(df.drop(columns=["Id", "Pawpularity"]).sum())

plt.figure(figsize=(10, 5))
sns.histplot(df["Pawpularity"], bins=30, kde=True)
plt.title("Pawpularity Distribution")
plt.show()

print("\n*** Number of rows with Pawpularity > 80 for each feature ***")
result = df[df["Pawpularity"] > 80].drop(columns=["Id", "Pawpularity"]).sum()
print(result)

print("\n*** Number of rows with Pawpularity < 20 for each feature ***")
result = df[df["Pawpularity"] < 20].drop(columns=["Id", "Pawpularity"]).sum()
print(result)

print("\n*** Correlation matrix ***")
corr_matrix = df.drop(columns=["Id", "Pawpularity"]).corr()
plt.figure(figsize=(20, 10))
sns.heatmap(corr_matrix, annot=True)
plt.show()

features = df.columns.values.tolist()
for variable in features[1:-1]:
    plt.figure(figsize=(10, 10))
    sns.boxplot(data=df, x=variable, y='Pawpularity')
    plt.suptitle(variable, fontsize=20, fontweight='bold')
    plt.show()
