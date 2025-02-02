from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

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

residuals = Y_test - Y_predict

# Plotting the diagnostic plots and scatter plots

# Residual plot
plt.figure(figsize=(8, 5))
sns.residplot(x=Y_predict, y=residuals, lowess=True, line_kws={"color": "red"})
plt.xlabel("Fitted Values (Predictions)")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted Values")
plt.savefig("documentation/OLSTestingFigures/residual_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Histogram of Residuals
plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=30)
plt.xlabel("Residuals")
plt.title("Histogram of Residuals")
plt.savefig("documentation/OLSTestingFigures/histogram.png", dpi=300, bbox_inches='tight')
plt.show()

# Q-Q Plot
sm.qqplot(residuals, line='45', fit=True)
plt.title("Q-Q Plot of Residuals")
plt.savefig("documentation/OLSTestingFigures/qq_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Scatter Plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=Y_test, y=Y_predict)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], color="red", linestyle="--")
plt.xlabel("Actual CO(GT) Values")
plt.ylabel("Predicted CO(GT) Values")
plt.title("Actual vs. Predicted CO(GT) Values")
plt.savefig("documentation/OLSTestingFigures/scatter_plot.png", dpi=300, bbox_inches='tight')
plt.show()

