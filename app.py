import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# Load dataset and trained models
@st.cache_data
def load_data():
    return pd.read_csv("california_housing.csv")

@st.cache_resource
def load_models():
    linear = joblib.load("linear_model.pkl")
    forest = joblib.load("random_forest_model.pkl")
    return linear, forest

data = load_data()
linear_model, forest_model = load_models()

# Dashboard title
st.title("🏠 House Price Prediction")
st.write("Predict median house values using Machine Learning.")

st.divider()

# Model selection
model_choice = st.selectbox(
    "Select Machine Learning Model",
    ["Random Forest", "Linear Regression"]
)

# Select model
if model_choice == "Random Forest":
    model = forest_model
else:
    model = linear_model

st.subheader("Enter Housing Details")

# Input fields
col1, col2 = st.columns(2)

with col1:
    medinc = st.number_input(
        "Median Income",
        min_value=float(data["MedInc"].min()),
        max_value=float(data["MedInc"].max()),
        value=float(data["MedInc"].median())
    )

    houseage = st.number_input(
        "House Age",
        min_value=float(data["HouseAge"].min()),
        max_value=float(data["HouseAge"].max()),
        value=float(data["HouseAge"].median())
    )

    averooms = st.number_input(
        "Average Rooms",
        min_value=float(data["AveRooms"].min()),
        max_value=float(data["AveRooms"].max()),
        value=float(data["AveRooms"].median())
    )

    avebedrms = st.number_input(
        "Average Bedrooms",
        min_value=float(data["AveBedrms"].min()),
        max_value=float(data["AveBedrms"].max()),
        value=float(data["AveBedrms"].median())
    )

with col2:
    population = st.number_input(
        "Population",
        min_value=float(data["Population"].min()),
        max_value=float(data["Population"].max()),
        value=float(data["Population"].median())
    )

    aveoccup = st.number_input(
        "Average Occupancy",
        min_value=float(data["AveOccup"].min()),
        max_value=float(data["AveOccup"].max()),
        value=float(data["AveOccup"].median())
    )

    latitude = st.number_input(
        "Latitude",
        min_value=float(data["Latitude"].min()),
        max_value=float(data["Latitude"].max()),
        value=float(data["Latitude"].median())
    )

    longitude = st.number_input(
        "Longitude",
        min_value=float(data["Longitude"].min()),
        max_value=float(data["Longitude"].max()),
        value=float(data["Longitude"].median())
    )

# Prediction button
if st.button("Predict House Value", type="primary"):

    input_data = pd.DataFrame([{
        "MedInc": medinc,
        "HouseAge": houseage,
        "AveRooms": averooms,
        "AveBedrms": avebedrms,
        "Population": population,
        "AveOccup": aveoccup,
        "Latitude": latitude,
        "Longitude": longitude
    }])

    prediction = model.predict(input_data)[0]

    predicted_price = prediction * 100000

    st.success("Prediction completed!")

    st.metric(
        "Estimated Median House Value",
        f"${predicted_price:,.2f}"
    )

    st.info(f"Model used: {model_choice}")

    st.caption(
        "Educational estimate based on the California Housing "
        "1990 census dataset. This predicts area-level median "
        "house values, not the market price of an individual property."
    )
    
data = load_data()
linear_model, forest_model = load_models()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

@st.cache_data
def evaluate_models():

    X = data.drop("MedHouseVal", axis=1)
    y = data["MedHouseVal"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    linear_predictions = linear_model.predict(X_test)
    forest_predictions = forest_model.predict(X_test)

    linear_mae = mean_absolute_error(y_test, linear_predictions)
    forest_mae = mean_absolute_error(y_test, forest_predictions)

    linear_r2 = r2_score(y_test, linear_predictions)
    forest_r2 = r2_score(y_test, forest_predictions)

    actual_predictions = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": forest_predictions
    })

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": forest_model.named_steps[
            "model"
        ].feature_importances_
    }).sort_values("Importance", ascending=False)

    return (
        linear_mae, forest_mae,
        linear_r2, forest_r2,
        actual_predictions, importance
    )

(
    linear_mae, forest_mae,
    linear_r2, forest_r2,
    actual_predictions, importance
) = evaluate_models()
st.header("📊 Model Performance Dashboard")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Linear Regression")
    st.metric("MAE", f"${linear_mae * 100000:,.0f}")
    st.metric("R2 Score", f"{linear_r2:.4f}")

with col2:
    st.subheader("Random Forest")
    st.metric("MAE", f"${forest_mae * 100000:,.0f}")
    st.metric("R2 Score", f"{forest_r2:.4f}")

st.divider()

st.subheader("Actual vs Predicted House Values")

st.scatter_chart(
    actual_predictions,
    x="Actual",
    y="Predicted"
)

st.caption(
    "Values are in $100,000 units. Points closer to the "
    "diagonal relationship indicate more accurate predictions."
)

st.subheader("Random Forest Feature Importance")

st.bar_chart(
    importance.set_index("Feature")
)

st.divider()