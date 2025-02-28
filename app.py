import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Load the dataset
@st.cache_data
def load_data():
    df = pd.read_excel('AmesHousing.xlsx')
    return df

df = load_data()

# Step 2: Preprocess the Data
st.title("Ames Housing Price Predictor")

st.write("### Data Overview")
st.write(df.head())

# Select relevant features
features = ['LotArea', 'YearBuilt', 'TotalBsmtSF', 'GrLivArea']
df = df[features + ['SalePrice']].dropna()

X = df[features]
y = df['SalePrice']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Step 4: Build Streamlit UI
st.write("## Predict Housing Prices")

# User input fields
lot_area = st.number_input("Lot Area (sq ft)", min_value=0)
year_built = st.number_input("Year Built", min_value=1800, max_value=2025)
total_bsmt_sf = st.number_input("Total Basement SF", min_value=0)
gr_liv_area = st.number_input("Above Ground Living Area SF", min_value=0)

# Load model and predict
if st.button("Predict Price"):
    with open('model.pkl', 'rb') as f:
        loaded_model = pickle.load(f)
    
    input_data = [[lot_area, year_built, total_bsmt_sf, gr_liv_area]]
    prediction = loaded_model.predict(input_data)
    
    st.write(f"### Estimated House Price: **${prediction[0]:,.2f}**")

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

st.write("### Model Performance")
st.write(f"Mean Absolute Error: **${mae:,.2f}**")

# Instructions to Deploy
st.write("## How to Deploy")
st.write("1. Upload `AmesHousing.xlsx`, `app.py`, and `requirements.txt` to GitHub.")
st.write("2. Deploy on [Streamlit Community Cloud](https://streamlit.io/community).")
