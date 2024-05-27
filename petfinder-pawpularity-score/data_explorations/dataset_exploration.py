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
plt.figure(figsize=(20, 20))
for variable in features[1:-1]:
    plt.subplot(3, 3, features.index(variable))
    sns.boxplot(data=df, x=variable, y='Pawpularity')
    plt.title(variable, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# save plot
plt.savefig("exploration_outputs/box_plots.png")