🏠 House Price Prediction Using Machine Learning

📌 Project Overview

This project predicts median house values using Machine Learning algorithms and the California Housing Dataset.

It includes data preprocessing, Exploratory Data Analysis (EDA), model training, performance evaluation, and an interactive Streamlit web application.

🎯 Project Objectives

- Analyze real-world housing data.
- Perform data cleaning and preprocessing.
- Visualize housing trends using EDA.
- Train and compare machine learning models.
- Predict median house values through a web application.

📊 Dataset

California Housing Dataset

- Total records: 20,640
- Total columns: 9
- Input features: 8
- Target variable: "MedHouseVal"

The dataset represents California census block groups from 1990. It predicts area-level median house values, not individual property prices.

🤖 Machine Learning Algorithms

1. Linear Regression
2. Random Forest Regression

🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib
- VS Code

⚙️ Project Workflow

1. Dataset collection
2. Data inspection and cleaning
3. Exploratory Data Analysis
4. Outlier detection using IQR
5. Missing-value handling
6. Feature scaling
7. Train-test splitting
8. Machine Learning model training
9. Model evaluation using MAE and R²
10. Streamlit application development

📁 Project Structure

House_Price_Prediction/
│
├── california_housing.csv
├── download_dataset.py
├── main.py
├── eda.py
├── preprocessing.py
├── train_model.py
├── test_model.py
├── app.py
├── linear_model.pkl
├── random_forest_model.pkl
├── requirements.txt
└── README.md

🚀 How to Run the Project

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

2. Open the project folder

cd House_Price_Prediction

3. Install dependencies

pip install -r requirements.txt

4. Download the dataset if it is missing

python download_dataset.py

5. Train the models

python train_model.py

6. Launch the Streamlit application

streamlit run app.py

📈 Model Evaluation

The models are evaluated using:

- Mean Absolute Error (MAE)
- R² Score

The application also displays actual-versus-predicted values and Random Forest feature importance.

💡 Key Learning Outcomes

- Understanding the end-to-end ML workflow.
- Working with real-world datasets.
- Applying preprocessing techniques.
- Training regression algorithms.
- Evaluating machine learning models.
- Building an interactive ML web application.

⚠️ Limitations

This project uses historical California census data from 1990. Its predictions should not be used as current real-estate valuations or as estimates for Indian properties.

👨‍💻 Author

AIML Student | Machine Learning Project

Project: House Price Prediction Using Machine Learning