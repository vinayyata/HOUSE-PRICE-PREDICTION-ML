import os
print("Current folder:", os.getcwd())
import pandas as pd
from sklearn.datasets import fetch_california_housing

# Download real-world housing dataset
housing = fetch_california_housing(as_frame=True)

# Convert into DataFrame
data = housing.frame

# Save as CSV
data.to_csv("california_housing.csv", index=False)

# Display dataset information
print("Dataset downloaded successfully!")

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 Records:")
print(data.head())

print("\nColumn Names:")
print(data.columns.tolist())

print("\nDataset Information:")
print(data.info())