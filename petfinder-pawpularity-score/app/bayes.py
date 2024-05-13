import pandas as pd

# Load the data
df = pd.read_csv('../data/train.csv')

# Calculate the probability Occulsion
p_a = df['Occlusion'].value_counts(normalize=True)
print(p_a)

# Calculate the probability of Pawpularity score being over 50
high_paw_score = df['Pawpularity'] > 50
p_b = high_paw_score.value_counts(normalize=True)
print (p_b)

# Calculate the probability of Occulsion given Pawpularity score being over 50
p_b_a = df.groupby('Occlusion')['Pawpularity'].apply(lambda x: (x > 50).mean())
print(p_b_a)

p_a_b = (p_b_a * p_a) / p_b

print(p_a_b)

  


