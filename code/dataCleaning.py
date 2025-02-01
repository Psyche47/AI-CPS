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

# sns.boxplot(df)
# plt.xticks(rotation=45)
# plt.title('Box Plot of the sensor data')
# plt.show()

numerical_columns = df.select_dtypes(include=['number']).columns
print(numerical_columns)

# Plotting the histogram to view the distribution of each column
# for column in numerical_columns:
#     plt.figure(figsize=(8, 5))  
#     sns.histplot(df[column], bins=30, kde=True) 
#     plt.title(f"Distribution of {column}") 
#     plt.xlabel(column)  
#     plt.ylabel("Frequency")
#     plt.show()

# Removing outliers using the IQR (Interquartile Range) Method
def remove_outliers_iqr(df, columns):
    copied_df = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_fence = Q1 - (1.5 * IQR)
        upper_fence = Q3 + (1.5 * IQR)
        copied_df = copied_df[(copied_df[col] >= lower_fence) & (copied_df[col] <= upper_fence)]
    return copied_df

df_post_iqr = remove_outliers_iqr(df, numerical_columns)

print("Shape of the Dataframe pre IQR:", df.shape)
print("Shape of the Dataframe pre IQR: ", df_post_iqr.shape)
# df.to_csv("data/cleaned2.csv", index=False, header=True)
sns.boxplot(df_post_iqr)
plt.xticks(rotation=45)
plt.title('Box Plot of the sensor data')
plt.show()