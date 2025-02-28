import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
def load_data():
    try:
        df = pd.read_csv("AmesHousing.xlsx.csv")  # Ensure this file is in the same folder
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

df = load_data()

if df is not None:
    # Select relevant features
    features = ['LotArea', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea']
    df = df[features + ['SalePrice']].dropna()  # Remove missing values

    # Define X (features) and y (target)
    X = df[features]
    y = df['SalePrice']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save trained model
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Streamlit UI
    st.title("🏡 Ames Housing Price Predictor")

    # User input
    lot_area = st.number_input("Lot Area", min_value=500, value=7500, step=100)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000, step=1)
    total_bsmt_sf = st.number_input("Total Basement SF", min_value=0, value=1000, step=50)
    gr_liv_area = st.number_input("Above Ground Living Area SF", min_value=500, value=1500, step=50)

    # Predict price
    if st.button("Predict Price"):
        with open("model.pkl", "rb") as f:
            loaded_model = pickle.load(f)

        input_data = np.array([[lot_area, year_built, total_bsmt_sf, gr_liv_area]])
        prediction = loaded_model.predict(input_data)

        st.success(f"🏠 Estimated House Price: **${prediction[0]:,.2f}**")
