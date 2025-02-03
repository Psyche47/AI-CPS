import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

model = tf.keras.models.load_model("data/currentAiSolution.keras")

df = pd.read_csv("data/test_data.csv")
# print(df.head())

Y_test = df["CO(GT)"]
X_test = df.drop(columns=["CO(GT)", "AH", "DayOfWeek", "IsWeekend"])

Y_pred = model.predict(X_test).flatten()
residuals = Y_test - Y_pred

mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

performance_indicators = {
    "Metric": ["Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)", "R-squared (R2)", "Mean Absolute Error (MAE)"],
    "Values": [mse, rmse, r2, mae]
}

performance_df = pd.DataFrame(performance_indicators)
performance_df.to_csv("data/ANNModelPerformanceMetricsTest.csv", header=True)

# Diagonstic plots
# Residual plot
plt.figure(figsize=(8, 5))
sns.residplot(x=Y_pred, y=residuals, lowess=True, line_kws={"color": "red"})
plt.xlabel("Predicted CO(GT)")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.savefig("documentation/ANNTestingFigures/residual_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Histogram of residuals
plt.figure(figsize=(8, 5))
sns.histplot(residuals, bins=30, kde=True)
plt.axvline(0, color='r', linestyle="--")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals")
plt.savefig("documentation/ANNTestingFigures/histogram.png", dpi=300, bbox_inches='tight')
plt.show()

# Q-Q Plot
sm.qqplot(residuals, line='s')
plt.title("Q-Q Plot of Residuals")
plt.savefig("documentation/ANNTestingFigures/qq_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# Scatter plot
plt.figure(figsize=(8, 5))
sns.scatterplot(x=Y_test, y=Y_pred, alpha=0.5)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'r', linestyle="--")  # 45-degree reference line
plt.xlabel("Actual CO(GT)")
plt.ylabel("Predicted CO(GT)")
plt.title("Actual vs. Predicted Values")
plt.savefig("documentation/ANNTestingFigures/scatter_plot.png", dpi=300, bbox_inches='tight')
plt.show()