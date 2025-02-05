import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


model = tf.keras.models.load_model("/tmp/knowledgeBase/currentAiSolution.keras")

df = pd.read_csv("/tmp/activationBase/activation_data.csv")
# print(df.head())

Y_test = df["CO(GT)"]
X_test = df.drop(columns=["CO(GT)", "AH", "DayOfWeek", "IsWeekend"])

Y_pred = model.predict(X_test).flatten()
residuals = Y_test - Y_pred

mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("Actual Value: ", Y_test.iloc[0])
print("ANN Predicted Value:", Y_pred[0])

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R² Score: {r2}")

