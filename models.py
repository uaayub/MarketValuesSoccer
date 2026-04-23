import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

#Load Data
df = pd.read_csv("player_attributes.csv")

#Define Features and Target
X = df.drop(columns=['market_value_in_eur', 'name'])
y = df['market_value_in_eur']

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Baseline Model
baseline_prediction = y_train.mean()
baseline_preds = np.full(len(y_test), baseline_prediction)

#Evaluation
mse = mean_squared_error(y_test, baseline_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, baseline_preds)
r2 = r2_score(y_test, baseline_preds)

print("Baseline Model Results")
print(f"Mean Market Value: {baseline_prediction:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R^2: {r2:.4f}")

#Linear Regression

#Scaling Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Train Model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

#Prediction
lr_preds = lr.predict(X_test_scaled)

#Evaluation
mse = mean_squared_error(y_test, lr_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, lr_preds)
r2 = r2_score(y_test, lr_preds) 

print("\nLinear Regression Results")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R^2: {r2:.4f}")

#Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

dt_preds = dt.predict(X_test)

mse = mean_squared_error(y_test, dt_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, dt_preds)
r2 = r2_score(y_test, dt_preds)

print("\nDecision Tree Results")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R^2: {r2:.4f}")