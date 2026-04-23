import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# Load Data
df = pd.read_csv("player_attributes.csv")

# Define Features and Target
X = df.drop(columns=['market_value_in_eur', 'name'])
y = df['market_value_in_eur']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# BASELINE MODEL
# =========================
baseline_prediction = y_train.mean()
baseline_preds = np.full(len(y_test), baseline_prediction)

baseline_mse = mean_squared_error(y_test, baseline_preds)
baseline_rmse = np.sqrt(baseline_mse)
baseline_mae = mean_absolute_error(y_test, baseline_preds)
baseline_r2 = r2_score(y_test, baseline_preds)

print("Baseline Model Results")
print(f"Mean Market Value: {baseline_prediction:.2f}")
print(f"MSE: {baseline_mse:.2f}")
print(f"RMSE: {baseline_rmse:.2f}")
print(f"MAE: {baseline_mae:.2f}")
print(f"R^2: {baseline_r2:.4f}")

# =========================
# LINEAR REGRESSION
# =========================

# Scaling Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Prediction
lr_preds = lr.predict(X_test_scaled)

# Evaluation
lr_mse = mean_squared_error(y_test, lr_preds)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

print("\nLinear Regression Results")
print(f"MSE: {lr_mse:.2f}")
print(f"RMSE: {lr_rmse:.2f}")
print(f"MAE: {lr_mae:.2f}")
print(f"R^2: {lr_r2:.4f}")

# =========================
# DECISION TREE
# =========================
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

dt_preds = dt.predict(X_test)

dt_mse = mean_squared_error(y_test, dt_preds)
dt_rmse = np.sqrt(dt_mse)
dt_mae = mean_absolute_error(y_test, dt_preds)
dt_r2 = r2_score(y_test, dt_preds)

print("\nDecision Tree Results")
print(f"MSE: {dt_mse:.2f}")
print(f"RMSE: {dt_rmse:.2f}")
print(f"MAE: {dt_mae:.2f}")
print(f"R^2: {dt_r2:.4f}")

# COMPARISON GRAPHS
models = ['Baseline', 'Linear Regression', 'Decision Tree']
rmse_values = [baseline_rmse, lr_rmse, dt_rmse]
mae_values = [baseline_mae, lr_mae, dt_mae]
r2_values = [baseline_r2, lr_r2, dt_r2]

# RMSE Plot
plt.figure()
plt.bar(models, rmse_values)
plt.title("RMSE Comparison")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.tight_layout()
plt.show()

# MAE Plot
plt.figure()
plt.bar(models, mae_values)
plt.title("MAE Comparison")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.tight_layout()
plt.show()

# R² Plot
plt.figure()
plt.bar(models, r2_values)
plt.title("R² Comparison")
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.tight_layout()
plt.show()