import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Zomato Delivery Prediction",
    page_icon="🍔",
    layout="centered"
)

@st.cache_resource
def load_models():
    eta = joblib.load("models/eta_model.pkl")
    late = joblib.load("models/late_model.pkl")
    return eta, late

st.title("🍔 Zomato Delivery Prediction")
st.write("Machine Learning based delivery time and late-delivery risk prediction.")

try:
    eta_model, late_model = load_models()
except Exception as e:
    st.error(f"Unable to load models: {e}")
    st.stop()

st.subheader("Enter Delivery Details")

distance = st.number_input("Delivery distance (km)", 0.1, 100.0, 5.0)
age = st.number_input("Delivery person age", 18, 70, 25)
rating = st.number_input("Delivery person rating", 1.0, 5.0, 4.5, step=0.1)
vehicle = st.selectbox("Vehicle condition", [0, 1, 2], index=1)
multiple = st.selectbox("Multiple deliveries", [0, 1, 2, 3], index=1)

traffic = st.selectbox(
    "Road traffic density",
    ["Low", "Medium", "High", "Jam"]
)
weather = st.selectbox(
    "Weather conditions",
    ["Sunny", "Cloudy", "Windy", "Fog", "Sandstorms", "Stormy", "Rainy"]
)
festival = st.selectbox("Festival", ["No", "Yes"])

if st.button("Predict Delivery", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        "distance_km": distance,
        "Delivery_person_Age": age,
        "Delivery_person_Ratings": rating,
        "Vehicle_condition": vehicle,
        "multiple_deliveries": multiple,
        "Road_traffic_density": traffic,
        "Weather_conditions": weather,
        "Festival": festival
    }])

    try:
        eta = float(eta_model.predict(input_data)[0])
        late_probability = float(late_model.predict_proba(input_data)[0, 1])

        st.subheader("Prediction Results")
        st.metric("Estimated delivery time", f"{max(0, eta):.1f} minutes")
        st.metric("Late-delivery probability", f"{late_probability * 100:.1f}%")
        st.progress(min(max(late_probability, 0.0), 1.0))

        if late_probability >= 0.5:
            st.warning("The model predicts a higher risk of late delivery.")
        else:
            st.success("The model predicts a lower risk of late delivery.")

        st.caption("These are model estimates, not guaranteed delivery times.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption("Student ML Project | Linear Regression + Logistic Regression")
