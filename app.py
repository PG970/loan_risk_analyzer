import streamlit as st
import numpy as np
import pandas as pd
import joblib
#import shap
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="AI Loan Risk Analyzer",
    layout="wide",
    page_icon="💰"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.05);
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
    backdrop-filter: blur(10px);
    margin-bottom: 15px;
}
.title {
    font-size: 32px;
    font-weight: bold;
}
.subtitle {
    font-size: 18px;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD ----------------
@st.cache_resource
def load_model():
    return joblib.load("xgboost.pkl")

model = load_model()

scaler = joblib.load("scaler.pkl")

df = pd.read_csv('student_loan_data.csv')
X = df.drop('default_risk', axis=1)
explainer = shap.Explainer(model, X)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🏦 AI Loan Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Predict • Explain • Analyze with Explainable AI</div>', unsafe_allow_html=True)

st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.sidebar.header("📥 Applicant Details")

credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
income = st.sidebar.number_input("Monthly Income", value=5000)
loan = st.sidebar.number_input("Loan Amount", value=20000)
dti = st.sidebar.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.3)
payment = st.sidebar.slider("Payment History", 0.0, 1.0, 0.8)
employment = st.sidebar.slider("Employment Years", 0.0, 10.0, 3.0)
prev_default = st.sidebar.selectbox("Previous Defaults", [0, 1, 2])

annual_income = income * 12
loan_to_income = loan / annual_income if annual_income != 0 else 0

input_data = pd.DataFrame([[
    credit_score, income, dti, loan,
    payment, employment, prev_default
]], columns=X.columns)

# Scale the input data
input_scaled = scaler.transform(input_data)
# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Risk"):

    prob = model.predict_proba(input_scaled)[0][1]

    if prob < 0.3:
        risk = "Low Risk 🟢"
        color = "#22c55e"
    elif prob < 0.7:
        risk = "Medium Risk 🟡"
        color = "#facc15"
    else:
        risk = "High Risk 🔴"
        color = "#ef4444"

    # ---------------- CARDS ----------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
        <h4>📊 Default Probability</h4>
        <h2>{prob:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
        <h4>⚠️ Risk Level</h4>
        <h2 style="color:{color}">{risk}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
        <h4>💰 Loan Ratio</h4>
        <h2>{loan_to_income:.2f}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ---------------- SHAP ----------------
    #st.subheader("🔍 Explainable AI (SHAP)")

    #shap_values = explainer.shap_values(input_data)
    #fig, ax = plt.subplots()
    #shap.plots._waterfall.waterfall_legacy(explainer.expected_value,shap_values[0],feature_names=X.columns)
    #st.pyplot(fig)

    # ---------------- INSIGHTS ----------------
    st.subheader("💡 Smart Insights")

    insights = []

    if credit_score < 600:
        insights.append("Low credit score increases risk")

    if prev_default > 0:
        insights.append("Previous defaults detected")

    if loan_to_income > 0.5:
        insights.append("High loan compared to income")

    if payment < 0.6:
        insights.append("Poor payment history")

    if len(insights) == 0:
        st.success("Applicant looks financially stable ✅")
    else:
        for i in insights:
            st.warning(i)

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("thank you for using AI loan risk analyzer!")
