import joblib
import pandas as pd

# Load saved Random Forest model
model = joblib.load("random_forest_model.pkl")

# Create sample house data
sample = pd.DataFrame([{
    "MedInc": 5.0,
    "HouseAge": 20,
    "AveRooms": 5.5,
    "AveBedrms": 1.1,
    "Population": 1000,
    "AveOccup": 3.0,
    "Latitude": 34.05,
    "Longitude": -118.25
}])

# Predict house value
prediction = model.predict(sample)

print("Predicted house value:")
print("$", round(prediction[0] * 100000, 2))