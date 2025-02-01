import pandas as pd

df = pd.read_csv("data/joint_data_collection.csv")

print(df.isna().sum())