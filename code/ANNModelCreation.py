import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

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

train_dict = train.history
Y_pred = ann_model.predict(X_val).flatten()
residuals = Y_val - Y_pred

# Storing loss and MAE for training
train_loss = train_dict['loss']
val_loss = train_dict['val_loss']
train_mae = train_dict['mae']
val_mae = train_dict['val_mae']
epochs = range(1, len(train_loss) + 1)

performance_indicators = {
    "Final Training Loss (MSE)": train_loss[-1],
    "Final Validation Loss (MSE)": val_loss[-1],
    "Final Training MAE": train_mae[-1],
    "Final Validation MAE": val_mae[-1],
    "Total Training Iterations (Epochs)": len(epochs)
}

performance_df = pd.DataFrame(performance_indicators, index=["ANN Model Performance Indicators"])
performance_df.to_csv("data/ANNModelPerformanceMetrics.csv", header=True)

sns.set_style("whitegrid")
# Training and Validation Loss (MSE) curve
plt.figure(figsize=(10, 5))
sns.lineplot(x=epochs, y=train_loss, label='Training Loss (MSE)', marker='o', color='blue')
sns.lineplot(x=epochs, y=val_loss, label='Validation Loss (MSE)', marker='o', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.savefig("documentation/ANNTrainingFigures/training_validation_loss.png", dpi=300, bbox_inches='tight')
plt.show()

# Training and Validation MAE Curve
plt.figure(figsize=(10, 5))
sns.lineplot(x=epochs, y=train_mae, label='Training MAE', marker='o', color='blue')
sns.lineplot(x=epochs, y=val_mae, label='Validation MAE', marker='o', color='red')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Training and Validation MAE Over Epochs')
plt.legend()
plt.savefig("documentation/ANNTrainingFigures/training_validation.png", dpi=300, bbox_inches='tight')
plt.show()

# Residual Plot
plt.figure(figsize=(10, 5))
sns.residplot(x=Y_pred, y=residuals, lowess=True, color="blue", scatter_kws={'alpha': 0.5})
plt.xlabel("Predicted CO(GT) Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.axhline(y=0, color='black', linestyle="--") 
plt.savefig("documentation/ANNTrainingFigures/residual.png", dpi=300, bbox_inches='tight')
plt.show()

# Histogram of Residuals
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=30, kde=True, color="blue")
plt.xlabel("Residuals (Errors)")
plt.ylabel("Frequency")
plt.title("Error Distribution (Residuals Histogram)")
plt.axvline(x=0, color='red', linestyle='--')
plt.savefig("documentation/ANNTrainingFigures/residual_histogram.png", dpi=300, bbox_inches='tight')
plt.show()

# Scatter plots
# Predicted vs. Actual Values
plt.figure(figsize=(10, 5))
sns.scatterplot(x=Y_val, y=Y_pred, alpha=0.5, color="green")
plt.xlabel("Actual CO(GT) Values")
plt.ylabel("Predicted CO(GT) Values")
plt.title("Predicted vs. Actual Values")
plt.plot([min(Y_val), max(Y_val)], [min(Y_val), max(Y_val)], color="red", linestyle="--")
plt.savefig("documentation/ANNTrainingFigures/predicted_actual.png", dpi=300, bbox_inches='tight')
plt.show()

# Residuals vs. Actual Values
plt.figure(figsize=(10, 5))
sns.scatterplot(x=Y_val, y=residuals, alpha=0.5, color="blue")
plt.axhline(y=0, color='r', linestyle='--') 
plt.xlabel("Actual CO(GT) Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs. Actual Values")
plt.savefig("documentation/ANNTrainingFigures/residual_actual.png", dpi=300, bbox_inches='tight')
plt.show()

# Residuals vs. Predicted Values
plt.figure(figsize=(10, 5))
sns.scatterplot(x=Y_pred, y=residuals, alpha=0.5, color="green")
plt.axhline(y=0, color='r', linestyle='--')  
plt.xlabel("Predicted CO(GT) Values")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residuals vs. Predicted Values")
plt.savefig("documentation/ANNTrainingFigures/residual_predicted.png", dpi=300, bbox_inches='tight')
plt.show()

ann_model.save("data/currentAiSolution.keras")