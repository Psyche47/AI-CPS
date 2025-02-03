import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/training_data.csv")
#print(df.head())

Y = df["CO(GT)"]
X = df.drop(columns=["CO(GT)", "AH", "DayOfWeek", "IsWeekend"])
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=23)

# Defining the ANN model
ann_model = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=([X.shape[1],])),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1)
])

ann_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

train = ann_model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_data=(X_val, Y_val), verbose=1)

val_loss, val_mae = ann_model.evaluate(X_val, Y_val, verbose=1)

print(f"Validation Loss: {val_loss}")
print(f"Validation MAE: {val_mae}")

ann_model.save("data/currentAiSolution.keras")