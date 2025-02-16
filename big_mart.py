import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model and feature names
model = joblib.load("Model/model.pkl")
expected_features = joblib.load("Model/expected_features.pkl")

# Function for preprocessing
def pre_processing(df):
    # Filling Missing Values
    df['Outlet_Size'] = df['Outlet_Size'].fillna(df['Outlet_Size'].mode()[0])
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())

    # Feature encoding
    df = pd.get_dummies(df, columns=['Item_Fat_Content', 'Item_Type', 'Outlet_Type', 'Outlet_Identifier', 'Outlet_Location_Type'], drop_first=True, dtype='int')

    # Mapping Outlet_Size
    size_mapping = {'Small': 1, 'Medium': 2, 'High': 3}
    df['Outlet_Size'] = df['Outlet_Size'].map(size_mapping)

    # Drop unnecessary columns
    df.drop(columns=['Item_Identifier'], inplace=True, errors='ignore')

    # Ensure input features match the trained model
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with 0
    df = df[expected_features]  # Reorder columns

    return df

# Streamlit App Layout
st.title("🛒 Big Mart Sales Prediction")

# Sidebar
with st.sidebar:
    st.header("📌 About the App")
    st.write(
        "This app predicts **Item Outlet Sales** using a trained Machine Learning model."
        "\n\n🔹 Enter the required details below."
        "\n🔹 Click **Predict Sales** to get an estimated sales value."
        "\n"
        "\n🔹Evaluation Scores🔹"
        "\n\n- RMSE: 1150.07"
        "\n\n- Accuracy : 61%"
    )

# Layout using columns
col1, col2 = st.columns(2)

# Collect user input
with col1:
    Item_Weight = st.number_input("Item Weight (kg)", min_value=1.0, max_value=50.0, value=10.0, step=0.1)
    Item_Fat_Content = st.selectbox("Item Fat Content", ['Low Fat', 'Regular'])
    Item_Visibility = st.number_input("Item Visibility", min_value=0.0, max_value=0.5, value=0.1, step=0.01)
    Item_Type = st.selectbox("Item Type", ['Fruits and Vegetables', 'Dairy', 'Meat', 'Baking Goods', 'Frozen Foods'])
    Item_MRP = st.number_input("Item MRP (₹)", min_value=1.0, max_value=500.0, value=100.0, step=1.0)

with col2:
    Outlet_Identifier = st.selectbox("Outlet Identifier", ['OUT049', 'OUT018', 'OUT027', 'OUT013'])
    Outlet_Establishment_Year = st.number_input("Outlet Establishment Year", min_value=1985, max_value=2020, value=2000)
    Outlet_Size = st.selectbox("Outlet Size", ['Small', 'Medium', 'High'])
    Outlet_Location_Type = st.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
    Outlet_Type = st.selectbox("Outlet Type", ['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'])

# Create input dataframe
input_data = pd.DataFrame({
    'Item_Weight': [Item_Weight],
    'Item_Fat_Content': [Item_Fat_Content],
    'Item_Visibility': [Item_Visibility],
    'Item_Type': [Item_Type],
    'Item_MRP': [Item_MRP],
    'Outlet_Identifier': [Outlet_Identifier],
    'Outlet_Establishment_Year': [Outlet_Establishment_Year],
    'Outlet_Size': [Outlet_Size],
    'Outlet_Location_Type': [Outlet_Location_Type],
    'Outlet_Type': [Outlet_Type]
})

# Preprocess the input
processed_data = pre_processing(input_data)

# Predict
if st.button("🚀 Predict Sales"):
    prediction = model.predict(processed_data)
    st.subheader("Predicted Sales")
    st.success(f"💰 ₹{round(prediction[0], 2)}")
