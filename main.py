import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.ensemble import RandomForestRegressor
# Load dataset
data = pd.read_csv("house_data.csv")

print("Dataset:")
print(data.head())

# Input features
X = data[["Area", "Bedrooms", "Bathrooms",
          "Age", "Location_Score"]]

# Output
y = data["Price"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)

print("\nActual Prices:")
print(y_test.values)

print("\nPredicted Prices:")
print(predictions)

# Evaluate model
print("\nMAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))

# Predict new house price
new_house = pd.DataFrame({
    "Area": [1500],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "Age": [5],
    "Location_Score": [8]
})

price = model.predict(new_house)

print("\nPredicted House Price: ₹", round(price[0], 2))
# Model evaluation

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
import matplotlib.pyplot as plt

# Actual vs Predicted prices
plt.figure(figsize=(8, 5))

plt.scatter(y_test, predictions)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")

plt.title("Actual vs Predicted House Prices")

plt.grid(True)
plt.show()
# Random Forest Model

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train model
rf_model.fit(X_train, y_train)

# Predict house prices
rf_predictions = rf_model.predict(X_test)

# Evaluate model
rf_mae = mean_absolute_error(y_test, rf_predictions)

rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Results")

print("MAE:", rf_mae)

print("R2 Score:", rf_r2)
