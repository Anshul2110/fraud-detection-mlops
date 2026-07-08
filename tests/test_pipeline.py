import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys


# Skip data-dependent tests if data not available (e.g. in CI)
DATA_AVAILABLE = os.path.exists('data/processed/train_processed.csv')
MODEL_AVAILABLE = os.path.exists('models/xgb_model.pkl')

skip_no_data  = pytest.mark.skipif(not DATA_AVAILABLE,  reason="Processed data not available")
skip_no_model = pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not available")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── 1. Preprocessing tests ───────────────────────────────────

def test_processed_data_exists():
    """Processed data file must exist after preprocessing."""
    assert os.path.exists('data/processed/train_processed.csv'), \
        "Processed data not found — run 02_preprocessing.ipynb first"

def test_processed_data_shape():
    """Processed data must have correct number of rows."""
    df = pd.read_csv('data/processed/train_processed.csv')
    assert len(df) == 590540, f"Expected 590540 rows, got {len(df)}"

def test_processed_data_has_target():
    """Processed data must contain isFraud column."""
    df = pd.read_csv('data/processed/train_processed.csv')
    assert 'isFraud' in df.columns, "Target column isFraud missing"

def test_no_missing_values():
    """Processed data must have no missing values."""
    df = pd.read_csv('data/processed/train_processed.csv')
    assert df.isnull().sum().sum() == 0, "Missing values found in processed data"

def test_fraud_rate():
    """Fraud rate must be approximately 3.5%."""
    df = pd.read_csv('data/processed/train_processed.csv')
    fraud_rate = df['isFraud'].mean()
    assert 0.03 < fraud_rate < 0.04, \
        f"Unexpected fraud rate: {fraud_rate:.4f}"

def test_engineered_features_exist():
    """Key engineered features must be present."""
    df = pd.read_csv('data/processed/train_processed.csv')
    required = ['hour', 'day', 'week', 'TransactionAmt_log',
                'card1_amt_mean', 'card1_amt_std', 'amt_deviation']
    for col in required:
        assert col in df.columns, f"Engineered feature '{col}' missing"

# ── 2. Model tests ───────────────────────────────────────────

def test_model_exists():
    """Saved model file must exist."""
    assert os.path.exists('models/xgb_model.pkl'), \
        "Model not found — run 03_modeling.ipynb first"

def test_model_loads():
    """Model must load without errors."""
    model = joblib.load('models/xgb_model.pkl')
    assert model is not None

def test_model_predicts_probability():
    """Model must return valid probabilities between 0 and 1."""
    df = pd.read_csv('data/processed/train_processed.csv')
    X = df.drop(columns=['isFraud']).head(100)

    model = joblib.load('models/xgb_model.pkl')
    proba = model.predict_proba(X)[:, 1]

    assert proba.min() >= 0.0, "Probability below 0"
    assert proba.max() <= 1.0, "Probability above 1"
    assert len(proba) == 100, "Wrong number of predictions"

def test_model_output_is_binary():
    """Model predictions must be 0 or 1."""
    df = pd.read_csv('data/processed/train_processed.csv')
    X = df.drop(columns=['isFraud']).head(100)

    model = joblib.load('models/xgb_model.pkl')
    preds = model.predict(X)

    assert set(preds).issubset({0, 1}), "Predictions contain values other than 0/1"

def test_model_detects_some_fraud():
    """Model must flag at least some transactions as fraud."""
    df = pd.read_csv('data/processed/train_processed.csv')
    fraud_samples = df[df['isFraud'] == 1].head(100)
    X = fraud_samples.drop(columns=['isFraud'])

    model = joblib.load('models/xgb_model.pkl')
    proba = model.predict_proba(X)[:, 1]

    # At least 50% of known fraud should have prob > 0.3
    high_risk = (proba > 0.3).sum()
    assert high_risk >= 50, \
        f"Model not detecting fraud well — only {high_risk}/100 above threshold"

# ── 3. Feature engineering tests ────────────────────────────

def test_hour_feature_range():
    """Hour feature must be between 0 and 23."""
    df = pd.read_csv('data/processed/train_processed.csv')
    assert df['hour'].between(0, 23).all(), "Hour feature out of range"

def test_day_feature_range():
    """Day feature must be between 0 and 6."""
    df = pd.read_csv('data/processed/train_processed.csv')
    assert df['day'].between(0, 6).all(), "Day feature out of range"

def test_log_transform_positive():
    """Log-transformed amount must be non-negative."""
    df = pd.read_csv('data/processed/train_processed.csv')
    assert (df['TransactionAmt_log'] >= 0).all(), \
        "Log transform produced negative values"