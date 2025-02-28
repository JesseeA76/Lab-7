import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np
import os

# Load the dataset
@st.cache_data
def load_data():
    try:
        df = pd.read_excel("AmesHousing.xlsx")
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

if df is not None:
    # Select relevant features
    features = ['LotArea', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea']
    
    # Handle missing values
    df = df[features + ['SalePrice']].dropna()

    # Define X (features) and y (target)
    X = df[features]
    y = df['SalePrice']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save trained model
    model_file = "model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)

    # Streamlit App UI
    st.title("🏡 Ames Housing Price Predictor")
    st.write("Enter the details of the house to get a predicted price.")

    # User input fields
    lot_area = st.number_input("Lot Area (sq ft)", min_value=500, max_value=100000, value=7500, step=100)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000, step=1)
    total_bsmt_sf = st.number_input("Total Basement SF", min_value=0, max_value=5000, value=1000, step=50)
    gr_liv_area = st.number_input("Above Ground Living Area SF", min_value=500, max_value=5000, value=1500, step=50)

    # Predict price
    if st.button("Predict Price"):
        if os.path.exists(model_file):
            with open(model_file, "rb") as f:
                loaded_model = pickle.load(f)

            input_data = np.array([[lot_area, year_built, total_bsmt_sf, gr_liv_area]])
            prediction = loaded_model.predict(input_data)

            st.success(f"🏠 Estimated House Price: **${prediction[0]:,.2f}**")
        else:
            st.error("Model file not found. Please retrain the model.")

    # Model Performance
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    st.write("### Model Performance")
    st.write(f"📉 Mean Absolute Error: **${mae:,.2f}**")

    # Instructions for Deployment
    st.write("## 🚀 Deployment Guide")
    st.markdown("""
    1. Upload `AmesHousing.xlsx`, `app.py`, and `requirements.txt` to GitHub.
    2. Deploy the app using [Streamlit Community Cloud](https://streamlit.io/community).
    3. Share the link to your deployed app.
    """)
