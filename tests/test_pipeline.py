import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Skip conditions ──────────────────────────────────────────
DATA_AVAILABLE  = os.path.exists('data/processed/train_processed.csv')
MODEL_AVAILABLE = os.path.exists('models/xgb_model.pkl')

skip_no_data  = pytest.mark.skipif(not DATA_AVAILABLE,  reason="Processed data not available in CI")
skip_no_model = pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model not available in CI")


# ════════════════════════════════════════════════════════════
# SECTION 1 — Synthetic tests (always run, even in CI)
# ════════════════════════════════════════════════════════════

def make_synthetic_data(n=500):
    """Create a small synthetic dataset that mimics the real one."""
    np.random.seed(42)
    df = pd.DataFrame({
        'TransactionAmt'     : np.random.exponential(150, n),
        'TransactionDT'      : np.random.randint(0, 15811131, n),
        'card1'              : np.random.randint(1000, 18000, n).astype(float),
        'card2'              : np.random.uniform(100, 600, n),
        'card3'              : np.random.uniform(100, 200, n),
        'card5'              : np.random.uniform(100, 300, n),
        'isFraud'            : np.random.choice([0, 1], n, p=[0.965, 0.035])
    })

    # Engineered features
    df['TransactionAmt_log'] = np.log1p(df['TransactionAmt'])
    df['hour']               = (df['TransactionDT'] // 3600) % 24
    df['day']                = (df['TransactionDT'] // (3600 * 24)) % 7
    df['week']               = (df['TransactionDT'] // (3600 * 24 * 7)) % 52
    df['card1_amt_mean']     = df.groupby('card1')['TransactionAmt'].transform('mean')
    df['card1_amt_std']      = df.groupby('card1')['TransactionAmt'].transform('std').fillna(0)
    df['card1_freq']         = df['card1'].map(df['card1'].value_counts(normalize=True))
    df['amt_deviation']      = df['TransactionAmt'] - df['card1_amt_mean']

    return df


def make_synthetic_model(df):
    """Train a tiny XGBoost model on synthetic data."""
    from xgboost import XGBClassifier
    X = df.drop(columns=['isFraud'])
    y = df['isFraud']
    model = XGBClassifier(n_estimators=10, max_depth=3,
                          random_state=42, eval_metric='logloss')
    model.fit(X, y)
    return model, X, y


# ── Synthetic: data shape & structure ───────────────────────

def test_synthetic_data_shape():
    """Synthetic data must have correct shape."""
    df = make_synthetic_data(500)
    assert df.shape[0] == 500
    assert df.shape[1] > 10


def test_synthetic_engineered_features():
    """All engineered features must exist in synthetic data."""
    df = make_synthetic_data(500)
    required = ['hour', 'day', 'week', 'TransactionAmt_log',
                'card1_amt_mean', 'card1_amt_std',
                'card1_freq', 'amt_deviation']
    for col in required:
        assert col in df.columns, f"Missing engineered feature: {col}"


def test_synthetic_hour_range():
    """Hour must be between 0 and 23."""
    df = make_synthetic_data(500)
    assert df['hour'].between(0, 23).all()


def test_synthetic_day_range():
    """Day must be between 0 and 6."""
    df = make_synthetic_data(500)
    assert df['day'].between(0, 6).all()


def test_synthetic_log_transform_positive():
    """Log transform of amount must be non-negative."""
    df = make_synthetic_data(500)
    assert (df['TransactionAmt_log'] >= 0).all()


def test_synthetic_no_missing_values():
    """Synthetic data must have no missing values."""
    df = make_synthetic_data(500)
    assert df.isnull().sum().sum() == 0


def test_synthetic_model_predicts_probability():
    """Model must return probabilities between 0 and 1."""
    df    = make_synthetic_data(500)
    model, X, _ = make_synthetic_model(df)
    proba = model.predict_proba(X)[:, 1]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_synthetic_model_output_is_binary():
    """Model predictions must be 0 or 1 only."""
    df    = make_synthetic_data(500)
    model, X, _ = make_synthetic_model(df)
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})


def test_synthetic_amt_deviation_calculation():
    """amt_deviation must equal TransactionAmt minus card1_amt_mean."""
    df = make_synthetic_data(500)
    expected = df['TransactionAmt'] - df['card1_amt_mean']
    assert np.allclose(df['amt_deviation'], expected)


def test_synthetic_freq_encoding_between_0_and_1():
    """Frequency encoding must be between 0 and 1."""
    df = make_synthetic_data(500)
    assert df['card1_freq'].between(0, 1).all()


# ════════════════════════════════════════════════════════════
# SECTION 2 — Real data tests (skipped in CI)
# ════════════════════════════════════════════════════════════

@skip_no_data
def test_processed_data_shape():
    """Processed data must have correct number of rows."""
    df = pd.read_csv('data/processed/train_processed.csv')
    assert len(df) == 590540, f"Expected 590540 rows, got {len(df)}"


@skip_no_data
def test_processed_data_has_target():
    """Processed data must contain isFraud column."""
    df = pd.read_csv('data/processed/train_processed.csv')
    assert 'isFraud' in df.columns


@skip_no_data
def test_no_missing_values_real():
    """Real processed data must have no missing values."""
    df = pd.read_csv('data/processed/train_processed.csv')
    assert df.isnull().sum().sum() == 0


@skip_no_data
def test_fraud_rate():
    """Fraud rate must be approximately 3.5%."""
    df = pd.read_csv('data/processed/train_processed.csv')
    fraud_rate = df['isFraud'].mean()
    assert 0.03 < fraud_rate < 0.04, \
        f"Unexpected fraud rate: {fraud_rate:.4f}"


@skip_no_data
def test_engineered_features_exist():
    """Key engineered features must exist in real data."""
    df = pd.read_csv('data/processed/train_processed.csv')
    required = ['hour', 'day', 'week', 'TransactionAmt_log',
                'card1_amt_mean', 'card1_amt_std', 'amt_deviation']
    for col in required:
        assert col in df.columns, f"Engineered feature '{col}' missing"


@skip_no_model
def test_model_loads():
    """Real model must load without errors."""
    model = joblib.load('models/xgb_model.pkl')
    assert model is not None


@skip_no_data
@skip_no_model
def test_model_predicts_probability():
    """Real model must return valid probabilities."""
    df    = pd.read_csv('data/processed/train_processed.csv')
    X     = df.drop(columns=['isFraud']).head(100)
    model = joblib.load('models/xgb_model.pkl')
    proba = model.predict_proba(X)[:, 1]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0
    assert len(proba) == 100


@skip_no_data
@skip_no_model
def test_model_output_is_binary():
    """Real model predictions must be 0 or 1."""
    df    = pd.read_csv('data/processed/train_processed.csv')
    X     = df.drop(columns=['isFraud']).head(100)
    model = joblib.load('models/xgb_model.pkl')
    preds = model.predict(X)
    assert set(preds).issubset({0, 1})


@skip_no_data
@skip_no_model
def test_model_detects_some_fraud():
    """Real model must flag at least 50% of known fraud above 0.3 threshold."""
    df            = pd.read_csv('data/processed/train_processed.csv')
    fraud_samples = df[df['isFraud'] == 1].head(100)
    X             = fraud_samples.drop(columns=['isFraud'])
    model         = joblib.load('models/xgb_model.pkl')
    proba         = model.predict_proba(X)[:, 1]
    high_risk     = (proba > 0.3).sum()
    assert high_risk >= 50, \
        f"Model not detecting fraud well — only {high_risk}/100 above threshold"