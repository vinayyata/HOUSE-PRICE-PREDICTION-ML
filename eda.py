import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd

# Load the dataset
data = pd.read_csv("california_housing.csv")

# Display first 5 rows
print("\nFIRST 5 ROWS")
print(data.head())

# Display dataset size
print("\nDATASET SHAPE")
print(data.shape)

# Display column names
print("\nCOLUMN NAMES")
print(data.columns)

# Display data types and non-null values
print("\nDATASET INFORMATION")
data.info()

# Display statistical summary
print("\nSTATISTICAL SUMMARY")
print(data.describe())
# Check missing values
print("\nMISSING VALUES")
print(data.isnull().sum())

# Check duplicate records
print("\nDUPLICATE RECORDS")
print(data.duplicated().sum())

# Total missing values
print("\nTOTAL MISSING VALUES")
print(data.isnull().sum().sum())
plt.figure(figsize=(8, 5))

plt.hist(data["MedHouseVal"], bins=40)

plt.title("Distribution of House Values")
plt.xlabel("Median House Value ($100,000 units)")
plt.ylabel("Number of Records")

plt.show()
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load and preprocess dataset
df = pd.read_csv("california_housing.csv")
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train Model 1: Linear Regression
print("--- Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

print(f"MAE: {mean_absolute_error(y_test, lr_pred):.4f}")
print(f"R2 Score: {r2_score(y_test, lr_pred):.4f}")

# 3. Train Model 2: Random Forest Regressor
print("\n--- Random Forest Regressor ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

print(f"MAE: {mean_absolute_error(y_test, rf_pred):.4f}")
print(f"R2 Score: {r2_score(y_test, rf_pred):.4f}")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load and preprocess dataset
df = pd.read_csv("california_housing.csv")
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train Model 1: Linear Regression
print("--- Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

print(f"MAE: {mean_absolute_error(y_test, lr_pred):.4f}")
print(f"R2 Score: {r2_score(y_test, lr_pred):.4f}")

# 3. Train Model 2: Random Forest Regressor
print("\n--- Random Forest Regressor ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

print(f"MAE: {mean_absolute_error(y_test, rf_pred):.4f}")
print(f"R2 Score: {r2_score(y_test, rf_pred):.4f}")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load and preprocess dataset
df = pd.read_csv("california_housing.csv")
X = df.drop(columns=["MedHouseVal"])
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Train Model 1: Linear Regression
print("--- Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)

print(f"MAE: {mean_absolute_error(y_test, lr_pred):.4f}")
print(f"R2 Score: {r2_score(y_test, lr_pred):.4f}")

# 3. Train Model 2: Random Forest Regressor
print("\n--- Random Forest Regressor ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)

print(f"MAE: {mean_absolute_error(y_test, rf_pred):.4f}")
print(f"R2 Score: {r2_score(y_test, rf_pred):.4f}")
# Graph 2: Income vs House Value

plt.figure(figsize=(8, 5))

plt.scatter(
    data["MedInc"],
    data["MedHouseVal"],
    alpha=0.3
)

plt.title("Income vs House Value")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.show()
# Graph 3: Correlation Heatmap

import matplotlib.pyplot as plt

correlation = data.corr(numeric_only=True)

plt.figure(figsize=(10, 7))

plt.imshow(correlation, cmap="coolwarm", interpolation="nearest")

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap")

for i in range(len(correlation.columns)):
    for j in range(len(correlation.columns)):
        plt.text(j, i, f"{correlation.iloc[i, j]:.2f}",
                 ha="center", va="center", fontsize=7)

plt.tight_layout()
plt.show()
# Graph 4: Boxplot - Outlier Detection

plt.figure(figsize=(8, 5))

plt.boxplot(data["MedHouseVal"])

plt.title("Boxplot of Median House Values")
plt.ylabel("Median House Value ($100,000 units)")

plt.show()


