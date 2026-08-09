import streamlit as st
import pandas as pd
import joblib

# Load saved files
model = joblib.load(r"C:\users\Anchal\customer_churn_model.pk1")
columns = joblib.load(r"C:\users\Anchal\model_columns.pk1")
scaler = joblib.load(r"c:\users\Anchal\scaler.pk1")

st.title("Customer Churn Prediction")

st.write("Enter Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
Partner = st.selectbox("Partner", ["Yes","No"])
Dependents = st.selectbox("Dependents", ["Yes","No"])

tenure = st.number_input("Tenure", min_value=0)

PhoneService = st.selectbox("Phone Service", ["Yes","No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])

InternetService = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])

OnlineSecurity = st.selectbox("Online Security", ["Yes","No","No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes","No","No internet service"])

StreamingTV = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])

Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])

PaperlessBilling = st.selectbox("Paperless Billing", ["Yes","No"])

PaymentMethod = st.selectbox(
    "Payment Method",
    ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
)

MonthlyCharges = st.number_input("Monthly Charges")
TotalCharges = st.number_input("Total Charges")


if st.button("Predict"):

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

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    df = df.reindex(columns=columns, fill_value=0)

    df = scaler.transform(df)
   
    prediction = model.predict(df)

    if prediction[0] == 1:
        st.error("Customer will Churn")
    else:
        st.success("Customer will not Churn")