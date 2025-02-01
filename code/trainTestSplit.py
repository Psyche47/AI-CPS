import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/joint_data_collection.csv")

training_df, test_df = train_test_split(df, test_size=0.2, random_state=23)

# Selecting a random row from the dataset
activation_df = df.sample(n=1, random_state=23)

training_df.to_csv("data/training_data.csv", index=False, header=True)

test_df.to_csv("data/test_data.csv", index=False, header=True)

activation_df.to_csv("data/activation_data.csv", index=False, header=True)