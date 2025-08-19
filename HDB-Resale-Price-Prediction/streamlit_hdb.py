import pandas as pd
import joblib
import streamlit as st
# Load the model
loaded_model = joblib.load(r'./random_forest_model.pkl')
# App title
st.title(":house: HDB Resale Price Predictor")
st.write("Please fill in the values below to predict the resale price:")
# User inputs
floor_area = st.number_input("Floor Area (sqm)", min_value=0, max_value=1000, value=80, step=1)
exec_sold = st.number_input("Executive Flats Sold Nearby", min_value=0, max_value=200, value=0, step=1)
five_room_sold = st.number_input("5-Room Flats Sold Nearby", min_value=0, max_value=200, value=0, step=1)
three_room_sold = st.number_input("3-Room Flats Sold Nearby", min_value=0, max_value=200, value=0, step=1)
max_floor_lvl = st.number_input("Max Floor Level", min_value=1, max_value=50, value=12, step=1)
hdb_age = st.number_input("Age of the HDB (years)", min_value=0, max_value=99, value=30, step=1)
total_dwelling_units = st.number_input("Total Dwelling Units in Block", min_value=0, max_value=1000, value=100, step=1)
# Dropdown for region selection
region = st.selectbox('Town Region', ['North', 'South', 'East', 'West', 'Central'])
# Prepare dummy variables for the region
region_features = {
    'zone_north': 0,
    'zone_south': 0,
    'zone_east': 0,
    'zone_west': 0
}
if region == "North":
    region_features['zone_north'] = 1
elif region == "South":
    region_features['zone_south'] = 1
elif region == "East":
    region_features['zone_east'] = 1
elif region == "West":
    region_features['zone_west'] = 1
# Central is baseline (0s for all zone_*)
# Combine all features
input_features = {
    'floor_area_sqm': floor_area,
    'exec_sold': exec_sold,
    '5room_sold': five_room_sold,
    '3room_sold': three_room_sold,
    'max_floor_lvl': max_floor_lvl,
    'hdb_age': hdb_age,
    'total_dwelling_units': total_dwelling_units,
    'zone_north': region_features['zone_north'],
    'zone_south': region_features['zone_south'],
    'zone_east': region_features['zone_east'],
    'zone_west': region_features['zone_west']
}
# Create DataFrame from input
input_df = pd.DataFrame(input_features, index=[0])
# Match expected model features
expected_features = loaded_model.feature_names_in_
input_df = input_df[expected_features]  # Ensure correct order and columns
# Predict
if st.button("Predict"):
    predicted_price = loaded_model.predict(input_df)[0]
    st.subheader(":chart_with_upwards_trend: Predicted Resale Price")
    st.success(f"${predicted_price:,.2f}")