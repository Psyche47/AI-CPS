import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/AirQualityUCI.csv")

# Dropping 2 unnamed columns
df.drop(columns=["Unnamed: 15", "Unnamed: 16"], axis= 1, inplace=True)

# Replacing -200 with NaN values in the dataset
df.replace(to_replace=-200, value=np.nan, inplace=True)

df['Date'] = pd.to_datetime(df['Date'],dayfirst=True) 

# print(df.describe())

print("The column NMHC(GT) has", df['NMHC(GT)'].isnull().sum(), "null values.")

# Dropped NMHC(GT) because it has 8557 null values
df.drop(columns="NMHC(GT)", axis= 1, inplace=True)
print(df.info())
print(df.isnull().sum())
# print("The null values in the column are", df["CO(GT)"].isnull().sum())
missing_percentage = (df.isna().sum()/len(df)) * 100
print(missing_percentage)
print(len(df))

# Filling missing values in each column
for column in df.columns:
    if (df[column].name) == "Date" or (df[column].name) == "Time":
        continue
    else: 
        if missing_percentage[column] <= 10:
            df[column] = df[column].fillna(df[column].median()) 
        elif df[column].dtype in ['float64', 'int64']:
            df[column] = df[column].interpolate(method='linear')
        else:
            df[column] = df[column].fillna(method='ffill')

print(df.isnull().sum())

sns.boxplot(df)
plt.xticks(rotation=45)
plt.title('Box Plot of the sensor data')
plt.show()

# df.to_csv("data/cleaned2.csv", index=False, header=True)