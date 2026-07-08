from fastapi import FastAPI, HTTPException
from api.schemas import TransactionInput, PredictionResponse
import joblib
import numpy as np
import pandas as pd
import shap
import os

app = FastAPI(
    title="Fraud Detection API",
    description="ML-powered fraud detection with SHAP explainability",
    version="1.0.0"
)

# Load model once at startup
MODEL_PATH = "models/xgb_model.pkl"
model = joblib.load(MODEL_PATH)
explainer = shap.TreeExplainer(model)

# Load a sample to get feature names
df_sample = pd.read_csv("data/processed/train_processed.csv", nrows=1)
FEATURE_COLS = [c for c in df_sample.columns if c != 'isFraud']

def build_feature_row(data: TransactionInput) -> pd.DataFrame:
    """Build a single-row dataframe with all features."""
    # Start with zeros for all features
    row = pd.DataFrame(np.zeros((1, len(FEATURE_COLS))), columns=FEATURE_COLS)

    # Fill known inputs
    row['TransactionAmt']     = data.TransactionAmt
    row['TransactionDT']      = data.TransactionDT
    row['card1']              = data.card1
    row['card2']              = data.card2
    row['card3']              = data.card3
    row['card5']              = data.card5

    # Engineered features
    row['TransactionAmt_log'] = np.log1p(data.TransactionAmt)
    row['hour']               = (data.TransactionDT // 3600) % 24
    row['day']                = (data.TransactionDT // (3600 * 24)) % 7
    row['week']               = (data.TransactionDT // (3600 * 24 * 7)) % 52
    row['amt_deviation']      = data.TransactionAmt - row['card1']
    row['card1_amt_mean']     = row['card1']
    row['card1_amt_std']      = 0
    row['card1_freq']         = 0.01

    return row

def get_risk_level(prob: float) -> str:
    if prob < 0.3:
        return "LOW"
    elif prob < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"

@app.get("/")
def root():
    return {"message": "Fraud Detection API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "model": "XGBoost"}

@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: TransactionInput):
    try:
        row = build_feature_row(transaction)
        prob = float(model.predict_proba(row)[:, 1][0])
        pred = "FRAUD" if prob >= 0.5 else "LEGIT"

        # SHAP for top features
        shap_vals = explainer.shap_values(row)[0]
        shap_series = pd.Series(np.abs(shap_vals), index=FEATURE_COLS)
        top5 = shap_series.nlargest(5).to_dict()

        return PredictionResponse(
            transaction_amount=transaction.TransactionAmt,
            fraud_probability=round(prob, 4),
            prediction=pred,
            risk_level=get_risk_level(prob),
            top_features={k: round(float(v), 4) for k, v in top5.items()}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))