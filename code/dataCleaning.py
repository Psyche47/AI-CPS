import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/AirQualityUCI.csv")

# Dropping 2 unnamed columns
df.drop(columns=["Unnamed: 15", "Unnamed: 16"], axis= 1, inplace=True)

# Replacing -200 with NaN values in the dataset
df.replace(to_replace=-200,value=np.nan,inplace=True)

df['Date'] = pd.to_datetime(df['Date'],dayfirst=True) 

# df['Time'] = pd.to_datetime(df['Time'], format= '%H.%M.%S' ).dt.time
print(df.head())
# print(df.describe())

#print(df['NMHC(GT)'].isnull().sum())

# Dropped NMHC(GT) because it has 8557 null values
df.drop(columns="NMHC(GT)", axis= 1, inplace=True)
print(df.info())
print(df.isnull().sum())