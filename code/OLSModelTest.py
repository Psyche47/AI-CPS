from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle

df_test = pd.read_csv("data/test_data.csv")
# print(df_test.head())

Y_test = df_test["CO(GT)"]
X_test = df_test.drop(columns=["CO(GT)", "AH", "DayOfWeek", "IsWeekend"])
X_test = sm.add_constant(X_test)

# Loading the OLS model from the data folder
with open("data/currentOlsSolution.pkl", "rb") as file:
    OLS_model = pickle.load(file)

Y_predict = OLS_model.predict(X_test)
print(Y_predict)

mse = mean_squared_error(Y_test, Y_predict)
rmse = np.sqrt(mse)
r2 = r2_score(Y_test, Y_predict)

print("Mean Squared Error (MSE):", mse)

print("Root Mean Squared Error (RMSE):", rmse)

print("R-squared (R2):", r2)