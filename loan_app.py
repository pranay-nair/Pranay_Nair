import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("loan_large_dataset.csv")
    return df

# Train the model
@st.cache_resource
def train_model(df):
    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier()
    model.fit(X_scaled, y)
    return model, scaler

# EMI Calculator
def calculate_emi(P, R_annual, N):
    R = R_annual / (12 * 100)  # Convert annual rate to monthly decimal
    if R == 0:
        return P / N
    emi = (P * R * (1 + R) ** N) / ((1 + R) ** N - 1)
    return emi

# Streamlit App Title
st.title("🏦 Loan Eligibility Predictor with EMI Calculator")

# Load data and train model
df = load_data()
model, scaler = train_model(df)

# Sidebar Inputs
st.sidebar.header("📋 Applicant Details")

st.sidebar.markdown("💡 *No upper limit for income, loan amount, or term. Enter realistic values.*")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", [0, 1, 2, 3])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

# Income fields without max limit
applicant_income = st.sidebar.number_input("Applicant Income (₹)", min_value=0, value=5000)
coapplicant_income = st.sidebar.number_input("Coapplicant Income (₹)", min_value=0, value=0)

# Loan fields with no max limit
st.sidebar.markdown("💡 *Enter loan amount in ₹ thousands. No upper limit.*")
loan_amount = st.sidebar.number_input("Loan Amount (₹ in thousands)", min_value=0, value=150)
loan_term = st.sidebar.number_input("Loan Term (in months)", min_value=12, value=360)

credit_history = st.sidebar.selectbox("Credit History", ["Good (1)", "Bad (0)"])
property_area = st.sidebar.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

# Interest rate and EMI calculation
interest_rate = st.sidebar.slider("Interest Rate (Annual %)", 5.0, 20.0, 8.5)
emi = calculate_emi(P=loan_amount * 1000, R_annual=interest_rate, N=loan_term)
st.sidebar.markdown(f"💸 **Estimated EMI:** ₹{emi:,.2f} / month")

# Map inputs to numeric
input_data = {
    'Gender': 1 if gender == "Male" else 0,
    'Married': 1 if married == "Yes" else 0,
    'Dependents': int(dependents),
    'Education': 0 if education == "Graduate" else 1,
    'Self_Employed': 1 if self_employed == "Yes" else 0,
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_term,
    'Credit_History': 1 if credit_history == "Good (1)" else 0,
    'Property_Area': ["Rural", "Semiurban", "Urban"].index(property_area)
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])
scaled_input = scaler.transform(input_df)

# Predict eligibility
if st.button("Check Loan Eligibility"):
    prediction = model.predict(scaled_input)
    result = "✅ Eligible for Loan" if prediction[0] == 1 else "❌ Not Eligible for Loan"
    st.subheader("Prediction Result:")
    st.success(result)

    # Show input summary
    st.markdown("---")
    st.markdown("### 🧾 Input Summary")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}))
