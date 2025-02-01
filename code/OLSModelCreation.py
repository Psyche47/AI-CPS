import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("data/training_data.csv")
#print(df.head())

Y = df["CO(GT)"]
X = df.drop(columns=["CO(GT)", "AH", "DayOfWeek", "IsWeekend"])
# The columns "AH", "DayOfWeek", and "IsWeekend" we dropped because the p values were > 0.05.

# print(X.head())

X = sm.add_constant(X)

ols_model = sm.OLS(Y, X).fit()

print(ols_model.summary())
