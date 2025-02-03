import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model = tf.keras.models.load_model("data/currentAiSolution.keras")

df = pd.read_csv("data/test_data.csv")
print(df.head())

Y_test = df["CO(GT)"]
X_test = df.drop(columns=["CO(GT)", "AH", "DayOfWeek", "IsWeekend"])

Y_pred = model.predict(X_test)

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