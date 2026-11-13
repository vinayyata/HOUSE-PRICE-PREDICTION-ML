import pandas as pd

# Load dataset
data = pd.read_csv("california_housing.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", data.shape)

# Select numerical columns
numerical_columns = data.select_dtypes(
    include=["number"]
).columns

# Detect outliers using IQR
for column in numerical_columns:

    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = data[
        (data[column] < lower_limit) |
        (data[column] > upper_limit)
    ]

    print("\nColumn:", column)
    print("Lower limit:", round(lower_limit, 2))
    print("Upper limit:", round(upper_limit, 2))
    print("Number of outliers:", len(outliers))

print("\nOutlier detection completed!")
# Step 2: Check and handle missing values

print("\nMissing values before handling:")

print(data.isnull().sum())

# Fill missing numerical values using median
for column in numerical_columns:

    if data[column].isnull().sum() > 0:

        median_value = data[column].median()

        data[column] = data[column].fillna(median_value)

print("\nMissing values after handling:")

print(data.isnull().sum())

print("\nMissing value handling completed!")
# Step 3: Feature Scaling

from sklearn.preprocessing import StandardScaler

# Separate input features and target
X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]

# Create scaler
scaler = StandardScaler()

# Scale input features
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame
X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

print("\nOriginal data:")
print(X.head())

print("\nScaled data:")
print(X_scaled.head())

print("\nFeature scaling completed!")
# Step 4: Train-Test Split

from sklearn.model_selection import train_test_split

# Separate features and target
X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Display dataset sizes
print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("\nTraining target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

print("\nTrain-test split completed!")