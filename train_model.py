import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load dataset
data = pd.read_csv("california_housing.csv")

# 2. Separate input features and target
X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 4. Create Linear Regression pipeline
linear_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

# 5. Train Linear Regression
linear_pipeline.fit(X_train, y_train)

# 6. Make predictions
linear_predictions = linear_pipeline.predict(X_test)

# 7. Evaluate Linear Regression
linear_mae = mean_absolute_error(
    y_test, linear_predictions
)

linear_r2 = r2_score(
    y_test, linear_predictions
)

print("\n--- Linear Regression Results ---")
print("MAE:", round(linear_mae, 4))
print("R2 Score:", round(linear_r2, 4))

# 8. Create Random Forest pipeline
forest_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

# 9. Train Random Forest
forest_pipeline.fit(X_train, y_train)

# 10. Make predictions
forest_predictions = forest_pipeline.predict(X_test)

# 11. Evaluate Random Forest
forest_mae = mean_absolute_error(
    y_test, forest_predictions
)

forest_r2 = r2_score(
    y_test, forest_predictions
)

print("\n--- Random Forest Results ---")
print("MAE:", round(forest_mae, 4))
print("R2 Score:", round(forest_r2, 4))

# 12. Compare models
results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, forest_mae],
    "R2 Score": [linear_r2, forest_r2]
})

print("\n--- Model Comparison ---")
print(results)

# Save trained pipelines
import joblib

joblib.dump(linear_pipeline, "linear_model.pkl")
joblib.dump(forest_pipeline, "random_forest_model.pkl")

print("\nModels saved successfully!")