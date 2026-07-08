import streamlit as st
import requests
import json

API_URL = "http://localhost:8000/predict"

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide"
)

st.title("🔍 Fraud Detection Dashboard")
st.markdown("*ML-powered transaction fraud detection with SHAP explainability*")
st.divider()

# ── Sidebar inputs ──────────────────────────────────────────
st.sidebar.header("Transaction Details")

amount = st.sidebar.number_input(
    "Transaction Amount ($)", min_value=0.01, max_value=50000.0, value=150.0)

product = st.sidebar.selectbox(
    "Product Category", ["W", "H", "C", "S", "R"])

card4 = st.sidebar.selectbox(
    "Card Type", ["visa", "mastercard", "american express", "discover"])

card6 = st.sidebar.selectbox(
    "Card Category", ["debit", "credit"])

email = st.sidebar.selectbox(
    "Purchaser Email Domain",
    ["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "other"])

hour = st.sidebar.slider("Transaction Hour (0-23)", 0, 23, 14)
TransactionDT = hour * 3600

card1 = st.sidebar.number_input("Card1 ID", value=10000)

predict_btn = st.sidebar.button("Analyze Transaction", type="primary")

# ── Main panel ───────────────────────────────────────────────
if predict_btn:
    payload = {
        "TransactionAmt"  : amount,
        "ProductCD"       : product,
        "card4"           : card4,
        "card6"           : card6,
        "P_emaildomain"   : email,
        "TransactionDT"   : TransactionDT,
        "card1"           : card1
    }

    with st.spinner("Analyzing transaction..."):
        try:
            response = requests.post(API_URL, json=payload)
            result   = response.json()

            # ── Verdict ─────────────────────────────────────
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Prediction", f"{result['prediction']}")

            with col2:
                st.metric("Fraud Probability",
                          f"{result['fraud_probability']*100:.1f}%")

            with col3:

                st.metric("Risk Level",
                          f"{result['risk_level']}")

            st.divider()

            # ── SHAP top features ────────────────────────────
            st.subheader("Why did the model make this decision?")
            st.markdown("Top 5 features that influenced this prediction (SHAP values):")

            top_features = result['top_features']
            for feat, val in sorted(
                    top_features.items(), key=lambda x: x[1], reverse=True):
                st.progress(
                    min(val / max(top_features.values()), 1.0),
                    text=f"**{feat}** — impact score: {val:.4f}"
                )

            # ── Transaction summary ──────────────────────────
            st.divider()
            st.subheader("Transaction Summary")
            st.json(payload)

        except Exception as e:
            st.error(f"API Error: {e}. Make sure the FastAPI server is running.")

else:
    # Landing state
    st.info("Fill in transaction details in the sidebar and click **Analyze Transaction**")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model", "XGBoost")
    col2.metric("PR-AUC", "0.707")
    col3.metric("Training Samples", "590,540")

    st.markdown("""
    ### How it works
    1. Enter transaction details in the sidebar
    2. The model predicts fraud probability in real time
    3. SHAP values explain *why* the model made its decision
    4. Risk level is assigned based on probability threshold
    """)