import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------
# Light & Subtle Styling
# -------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

h1 {
    color: #1e3a5f;
    text-align: center;
    font-size: 38px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 16px;
    margin-bottom: 30px;
}

.section-title {
    color: #1e3a5f;
    font-size: 22px;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 10px;
}

div.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 48px;
    background-color: #4f8fc9;
    color: white;
    font-size: 17px;
    font-weight: 600;
    border: none;
}

div.stButton > button:hover {
    background-color: #397caf;
    color: white;
}

.result-box {
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    margin-top: 20px;
    font-size: 20px;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Load Saved Files
# -------------------------------------------------

# Files are inside the same Customer_Churn folder
model = joblib.load("customer_churn_model.pk1")
columns = joblib.load("model_columns.pk1")
scaler = joblib.load("scaler.pk1")


# -------------------------------------------------
# Heading
# -------------------------------------------------

st.markdown(
    "<h1>📊 Customer Churn Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Enter customer details to predict whether the customer is likely to churn.</p>',
    unsafe_allow_html=True
)


# -------------------------------------------------
# Customer Information
# -------------------------------------------------

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

with col2:
    SeniorCitizen = st.selectbox(
        "Senior Citizen",
        [0, 1]
    )

with col3:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0,
        value=0
    )


col1, col2, col3 = st.columns(3)

with col1:
    Partner = st.selectbox(
        "Partner",
        ["Yes", "No"]
    )

with col2:
    Dependents = st.selectbox(
        "Dependents",
        ["Yes", "No"]
    )

with col3:
    PhoneService = st.selectbox(
        "Phone Service",
        ["Yes", "No"]
    )


# -------------------------------------------------
# Service Information
# -------------------------------------------------

st.markdown(
    '<div class="section-title">📱 Service Information</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    MultipleLines = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

with col2:
    InternetService = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

with col3:
    OnlineSecurity = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )


col1, col2, col3 = st.columns(3)

with col1:
    OnlineBackup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    DeviceProtection = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

with col3:
    TechSupport = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )


col1, col2 = st.columns(2)

with col1:
    StreamingTV = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

with col2:
    StreamingMovies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )


# -------------------------------------------------
# Contract & Billing
# -------------------------------------------------

st.markdown(
    '<div class="section-title">💳 Contract & Billing</div>',
    unsafe_allow_html=True
)

col1, col2, col3 = st.columns(3)

with col1:
    Contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

with col2:
    PaperlessBilling = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

with col3:
    PaymentMethod = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )


col1, col2 = st.columns(2)

with col1:
    MonthlyCharges = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=0.0
    )

with col2:
    TotalCharges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=0.0
    )


# -------------------------------------------------
# Prediction
# -------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔮 Predict Churn"):

    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }

    # Create DataFrame
    df = pd.DataFrame([data])

    # Convert categorical variables into dummy variables
    df = pd.get_dummies(df)

    # Match training columns exactly
    df = df.reindex(columns=columns, fill_value=0)

    # Scale the data
    df = scaler.transform(df)

    # Prediction
    prediction = model.predict(df)

    # Display result
    if prediction[0] == 1:

        st.markdown(
            """
            <div class="result-box"
            style="background-color:#fff1f2; color:#be123c;">
            ⚠️ Customer is likely to churn
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="result-box"
            style="background-color:#f0fdf4; color:#15803d;">
            ✅ Customer is likely to stay
            </div>
            """,
            unsafe_allow_html=True
        )


# -------------------------------------------------
# Footer
# -------------------------------------------------

st.markdown(
    '<div class="footer">Customer Churn Prediction • Machine Learning Project</div>',
    unsafe_allow_html=True
)