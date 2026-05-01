import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# Fixed random seed so results can be reproduced
np.random.seed(42)

# Load the cleaned dataset created from the preprocessing file
df = pd.read_csv("player_attributes.csv")

# Define Features and Target
# The target is market value because that is what the models are trying to predict.
# The name column is dropped because it is only used to identify players
X = df.drop(columns=['market_value_in_eur', 'name'])
y = df['market_value_in_eur']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baseline Model
# This model predicts the average market value from the training data, a comparison point to see if the other models actually improve.
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

# Linear Regression Model
# This model is used to test if there is a linear relationship between the features and market value.

# Scaling Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Make predictions on the test data
lr_preds = lr.predict(X_test_scaled)

# Evaluation
# RMSE, MAE, and R² are used to compare how close the predictions are to the actual values.
lr_mse = mean_squared_error(y_test, lr_preds)
lr_rmse = np.sqrt(lr_mse)
lr_mae = mean_absolute_error(y_test, lr_preds)
lr_r2 = r2_score(y_test, lr_preds)

print("\nLinear Regression Results")
print(f"MSE: {lr_mse:.2f}")
print(f"RMSE: {lr_rmse:.2f}")
print(f"MAE: {lr_mae:.2f}")
print(f"R^2: {lr_r2:.4f}")

# Default Decision Tree
# This model is used because market value is likely not based on a strictly linear relationship.
dt_default = DecisionTreeRegressor(random_state=42)
dt_default.fit(X_train, y_train)

dt_default_preds = dt_default.predict(X_test)

dt_default_mse = mean_squared_error(y_test, dt_default_preds)
dt_default_rmse = np.sqrt(dt_default_mse)
dt_default_mae = mean_absolute_error(y_test, dt_default_preds)
dt_default_r2 = r2_score(y_test, dt_default_preds)

print("\nDefault Decision Tree Results")
print(f"RMSE: {dt_default_rmse:.2f}")
print(f"MAE: {dt_default_mae:.2f}")
print(f"R^2: {dt_default_r2:.4f}")

# Default Random Forest
# Random forest is used because it combines multiple decision trees and can reduce overfitting.
rf_default = RandomForestRegressor(n_estimators=100, random_state=42)
rf_default.fit(X_train, y_train)

rf_default_preds = rf_default.predict(X_test)

rf_default_mse = mean_squared_error(y_test, rf_default_preds)
rf_default_rmse = np.sqrt(rf_default_mse)
rf_default_mae = mean_absolute_error(y_test, rf_default_preds)
rf_default_r2 = r2_score(y_test, rf_default_preds)

print("\nDefault Random Forest Results")
print(f"RMSE: {rf_default_rmse:.2f}")
print(f"MAE: {rf_default_mae:.2f}")
print(f"R^2: {rf_default_r2:.4f}")

# Decision Tree with Hyperparameter Tuning
# GridSearchCV tests different parameter combinations to find a better decision tree model.
dt_params = {
    'max_depth': [3, 5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt = DecisionTreeRegressor(random_state=42)

grid_dt = GridSearchCV(
    estimator=dt,
    param_grid=dt_params,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_dt.fit(X_train, y_train)

best_dt = grid_dt.best_estimator_
dt_preds = best_dt.predict(X_test)

dt_mse = mean_squared_error(y_test, dt_preds)
dt_rmse = np.sqrt(dt_mse)
dt_mae = mean_absolute_error(y_test, dt_preds)
dt_r2 = r2_score(y_test, dt_preds)

print("\nTuned Decision Tree Results")
print(f"Best Parameters: {grid_dt.best_params_}")
print(f"MSE: {dt_mse:.2f}")
print(f"RMSE: {dt_rmse:.2f}")
print(f"MAE: {dt_mae:.2f}")
print(f"R^2: {dt_r2:.4f}")

# Random Forest with Hyperparameter Tuning
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestRegressor(random_state=42)

grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=rf_params,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

grid_rf.fit(X_train, y_train)

best_rf = grid_rf.best_estimator_
rf_preds = best_rf.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_preds)
rf_rmse = np.sqrt(rf_mse)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print("\nTuned Random Forest Results")
print(f"Best Parameters: {grid_rf.best_params_}")
print(f"MSE: {rf_mse:.2f}")
print(f"RMSE: {rf_rmse:.2f}")
print(f"MAE: {rf_mae:.2f}")
print(f"R^2: {rf_r2:.4f}")

# Random Forest Feature Importance
# This shows which features had the biggest effect on the random forest prediction.
importances = best_rf.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 10 Most Important Features (Random Forest)")
print(feature_importance_df.head(10))

plt.figure()
plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importance (Random Forest)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Graphs
# These graphs are saved so the model results can be compared visually

models = [
    'Baseline',
    'Linear Regression',
    'DT (Default)',
    'DT (Tuned)',
    'RF (Default)',
    'RF (Tuned)'
]

rmse_values = [
    baseline_rmse,
    lr_rmse,
    dt_default_rmse,
    dt_rmse,
    rf_default_rmse,
    rf_rmse
]

mae_values = [
    baseline_mae,
    lr_mae,
    dt_default_mae,
    dt_mae,
    rf_default_mae,
    rf_mae
]

r2_values = [
    baseline_r2,
    lr_r2,
    dt_default_r2,
    dt_r2,
    rf_default_r2,
    rf_r2
]

# RMSE Plot
# Lower RMSE means the model's predictions are closer to the actual market values.
plt.figure()
plt.bar(models, rmse_values)
plt.title("RMSE Comparison")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("rmse_comparison.png")
plt.close()

# MAE Plot
# Lower MAE means the average prediction error is smaller.
plt.figure()
plt.bar(models, mae_values)
plt.title("MAE Comparison")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("mae_comparison.png")
plt.close()

# R² Plot
# Higher R² means the model explains more of the variation in market value.
plt.figure()
plt.bar(models, r2_values)
plt.title("R² Comparison")
plt.xlabel("Model")
plt.ylabel("R² Score")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("r2_comparison.png")
plt.close()

# Linear Regression Scatterplot
# This compares the actual market values to the predicted values for linear regression.
plt.figure(figsize=(8, 6))
plt.scatter(y_test, lr_preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.title("Linear Regression: Actual vs Predicted Market Value")
plt.xlabel("Actual Market Value")
plt.ylabel("Predicted Market Value")
plt.tight_layout()
plt.savefig("linear_regression_scatterplot.png")
plt.close()

# Tuned Decision Tree Scatterplot
# This shows how close the tuned decision tree predictions are to the actual values.
plt.figure(figsize=(8, 6))
plt.scatter(y_test, dt_preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.title("Tuned Decision Tree: Actual vs Predicted Market Value")
plt.xlabel("Actual Market Value")
plt.ylabel("Predicted Market Value")
plt.tight_layout()
plt.savefig("decision_tree_scatterplot.png")
plt.close()

# Tuned Random Forest Scatterplot
# This shows how close the tuned random forest predictions are to the actual values.
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.title("Tuned Random Forest: Actual vs Predicted Market Value")
plt.xlabel("Actual Market Value")
plt.ylabel("Predicted Market Value")
plt.tight_layout()
plt.savefig("random_forest_scatterplot.png")
plt.close()