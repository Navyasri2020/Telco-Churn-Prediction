import streamlit as st
import joblib
import pandas as pd


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("telco_churn_model.pkl")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📞",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📞 Telco Customer Churn Prediction")

st.write(
    "Enter the customer details below to predict whether "
    "the customer is likely to churn."
)

st.success("✅ Machine Learning Model Loaded Successfully")


# =========================================================
# CUSTOMER INFORMATION
# =========================================================

st.header("📋 Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    senior_citizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

with col3:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        max_value=72,
        value=1
    )


col1, col2 = st.columns(2)

with col1:
    partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

with col2:
    dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )


# =========================================================
# PHONE SERVICES
# =========================================================

st.header("📱 Phone Services")

col1, col2 = st.columns(2)

with col1:
    phone_service = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )

with col2:
    multiple_lines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )


# =========================================================
# INTERNET SERVICES
# =========================================================

st.header("🌐 Internet Services")

col1, col2, col3 = st.columns(3)

with col1:
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col2:
    online_security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

with col3:
    online_backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    device_protection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

with col2:
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

with col3:
    streaming_tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )


streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)


# =========================================================
# CONTRACT AND BILLING
# =========================================================

st.header("💳 Contract & Billing")

col1, col2, col3 = st.columns(3)

with col1:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    paperless_billing = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col3:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


# =========================================================
# CHARGES
# =========================================================

st.header("💰 Charges")

col1, col2 = st.columns(2)

with col1:
    monthly_charges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=50.0,
        step=1.0
    )

with col2:
    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=500.0,
        step=1.0
    )


# =========================================================
# PREDICTION
# =========================================================

st.divider()

st.subheader("🔮 Churn Prediction")

predict_button = st.button(
    "🚀 Predict Churn",
    use_container_width=True
)


if predict_button:

    # Create customer DataFrame
    customer_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    # Prediction
    prediction = model.predict(customer_data)

    # Churn probability
    churn_probability = model.predict_proba(
        customer_data
    )[0][1]

    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    st.divider()

    if prediction[0] == "Yes":

        st.error(
            "⚠️ Customer is predicted to Churn"
        )

    else:

        st.success(
            "✅ Customer is predicted to stay"
        )

    # Probability
    st.metric(
        "Churn Probability",
        f"{churn_probability * 100:.2f}%"
    )

    # Progress bar
    st.progress(
        float(churn_probability)
    )