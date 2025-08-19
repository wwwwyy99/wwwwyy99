import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.title("🛍️ High-Spender Prediction Tool")

# Sidebar user input
st.sidebar.header("📋 Input Customer Features")

age = st.sidebar.number_input("Age", min_value=14, max_value=80, value=30, step=1)
time_to_decision = st.sidebar.number_input("Time to Decision (days)", min_value=0, max_value=20, value=5, step=1)
brand_loyalty = st.sidebar.number_input("Brand Loyalty (1-5)", min_value=1, max_value=5, value=4, step=1)
product_rating = st.sidebar.number_input("Product Rating (1-5)", min_value=1, max_value=5, value=4, step=1)
return_rate = st.sidebar.number_input("Return Rate (0-2)", min_value=0, max_value=2, value=0, step=1)

purchase_channel_online_str = st.sidebar.selectbox("Did the customer purchase it online?", ["Yes", "No"])
purchase_channel_online = 1 if purchase_channel_online_str == "Yes" else 0

shipping_preference = st.sidebar.selectbox("Shipping Preference", ["Standard", "No Preference", "Others"])
shipping_standard = 1 if shipping_preference == "Standard" else 0
shipping_no_pref = 1 if shipping_preference == "No Preference" else 0

social_media_influence_str = st.sidebar.selectbox("Is Social Media Influence 'None'?", ["Yes", "No"])
social_media_influence_none = 1 if social_media_influence_str == "Yes" else 0

customer_loyalty_program_member = st.sidebar.checkbox("Customer is a Loyalty Program Member?", value=True)

# Input dictionary
input_dict = {
    'Age': age,
    'Time_to_Decision': time_to_decision,
    'Brand_Loyalty': brand_loyalty,
    'Product_Rating': product_rating,
    'Return_Rate': return_rate,
    'Customer_Loyalty_Program_Member': int(customer_loyalty_program_member),
    'Purchase_Channel_Online': purchase_channel_online,
    'Shipping_Preference_Standard': shipping_standard,
    'Shipping_Preference_No Preference': shipping_no_pref,
    'Social_Media_Influence_None': social_media_influence_none
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Reorder columns to match training data
expected_features = scaler.feature_names_in_
input_df = input_df.reindex(columns=expected_features)

# Scale inputs
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]  # Probability of being a High Spender

    if prediction[0] == 1:
        st.success(f"🎉 This customer is likely a **High Spender**!")
        st.balloons()
    else:
        st.info(f"This customer is **unlikely** a High Spender 😔")
