# 🔍 Fraud Detection with MLOps

![CI](https://github.com/Anshul2110/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end machine learning pipeline for credit card fraud detection, built with production MLOps practices — including experiment tracking, model explainability, REST API deployment, interactive dashboard, automated testing, and data drift monitoring.

---

## Project Highlights

| | |
|---|---|
| Dataset | IEEE-CIS Fraud Detection — 590,540 transactions |
| Best Model | XGBoost |
| PR-AUC | **0.707** (primary metric for imbalanced data) |
| Fraud Rate | 3.5% — severe class imbalance |
| Imbalance Strategy | SMOTE + `scale_pos_weight` |
| Top Feature | V317 (identified via SHAP) |
| Test Coverage | 10 synthetic CI tests + 9 real data tests |

---

##  Architecture

```
Raw Data (Kaggle)
      │
      ▼
  01 EDA ──────────────────────── Class imbalance, missing values, distributions
      │
      ▼
  02 Preprocessing ─────────────── Drop high-missing cols, impute, label encode
      │
      ▼
  03 Feature Engineering ────────── Time, aggregation, frequency, log features
      │
      ▼
  04 Model Training ─────────────── XGBoost + LightGBM + Ensemble
      │                             Tracked with MLflow (PR-AUC: 0.707)
      ▼
  05 Explainability ─────────────── SHAP global + local explanations
      │
      ▼
  06 Deployment
      ├── FastAPI ────────────────── REST endpoint + Swagger UI
      └── Streamlit ──────────────── Interactive dashboard
      │
      ▼
  07 Drift Monitoring ───────────── Evidently AI drift report
      │
      ▼
  08 CI/CD ──────────────────────── GitHub Actions automated test suite
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| ML Models | XGBoost, LightGBM, scikit-learn |
| Imbalance Handling | SMOTE (imbalanced-learn) |
| Explainability | SHAP (TreeExplainer) |
| Experiment Tracking | MLflow |
| Drift Monitoring | Evidently AI |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Data | pandas, numpy |

---

## Results

### Model Comparison

| Model | PR-AUC | Notes |
|---|---|---|
| **XGBoost** | **0.707** | Best model — used for deployment |
| Ensemble (XGB + LGB) | 0.698 | Average of both probabilities |
| LightGBM | 0.681 | Fast training, slightly lower AUC |

> **Why PR-AUC?** With only 3.5% fraud, accuracy is misleading — a model predicting "all legit" gets 96.5% accuracy but catches zero fraud. PR-AUC measures performance specifically on the minority (fraud) class.

### Top Predictive Features (SHAP)

The most influential feature was **V317** — one of Vesta's anonymized behavioral features — followed by transaction amount and card-level aggregation features. SHAP waterfall plots provide per-transaction explanations showing exactly why each prediction was made.

---

## Quickstart

### 1. Clone and install
```bash
git clone https://github.com/Anshul2110/fraud-detection-mlops.git
cd fraud-detection-mlops
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. Download data
1. Go to [kaggle.com/c/ieee-fraud-detection/data](https://www.kaggle.com/c/ieee-fraud-detection/data)
2. Accept competition rules → Download All
3. Unzip and place CSVs in `data/raw/`

```
data/raw/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

### 3. Run notebooks in order
```bash
jupyter notebook
```

```
notebooks/01_eda.ipynb               ← Exploratory Data Analysis
notebooks/02_preprocessing.ipynb     ← Cleaning + Feature Engineering
notebooks/03_modeling.ipynb          ← XGBoost + LightGBM + MLflow
notebooks/04_explainability.ipynb    ← SHAP explanations
notebooks/05_drift_monitoring.ipynb  ← Evidently drift report
```

### 4. Start the API
```bash
uvicorn api.main:app --reload
```
Swagger UI → **http://localhost:8000/docs**

### 5. Start the dashboard
```bash
streamlit run dashboard/app.py
```
Dashboard → **http://localhost:8501**

### 6. View MLflow experiments
```bash
mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
```
MLflow UI → **http://localhost:5000**

### 7. Run tests
```bash
pytest tests/test_pipeline.py -v
```

---

## Project Structure

```
fraud-detection-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
│
├── data/
│   ├── raw/                        # Original Kaggle data (gitignored)
│   └── processed/                  # Cleaned feature matrix (gitignored)
│
├── notebooks/
│   ├── 01_eda.ipynb                # EDA — distributions, imbalance, missing
│   ├── 02_preprocessing.ipynb      # Imputation, encoding, feature engineering
│   ├── 03_modeling.ipynb           # Training + MLflow experiment tracking
│   ├── 04_explainability.ipynb     # SHAP global + local explanations
│   └── 05_drift_monitoring.ipynb   # Evidently data drift report
│
├── src/
│   ├── data_preprocessing.py       # Reusable preprocessing functions
│   ├── feature_engineering.py      # Reusable feature engineering functions
│   ├── train.py                    # Training pipeline
│   ├── evaluate.py                 # Evaluation utilities
│   └── explain.py                  # SHAP utilities
│
├── api/
│   ├── main.py                     # FastAPI app with /predict endpoint
│   └── schemas.py                  # Pydantic request/response models
│
├── dashboard/
│   └── app.py                      # Streamlit interactive dashboard
│
├── tests/
│   └── test_pipeline.py            # pytest suite (synthetic + real data tests)
│
├── models/                         # Saved models + SHAP plots (gitignored)
├── mlflow/                         # MLflow SQLite DB (gitignored)
├── requirements.txt
├── config.yaml
└── README.md
```

---

## Key MLOps Concepts Demonstrated

**Experiment Tracking** — Every training run (XGBoost, LightGBM, Ensemble) is logged to MLflow with full hyperparameters, metrics, and model artifacts. Runs are comparable side-by-side in the MLflow UI.

**Explainability** — SHAP TreeExplainer generates both global feature importance (which features matter most overall) and local explanations (why was *this specific transaction* flagged as fraud).

**Model Serving** — FastAPI REST endpoint accepts transaction inputs, returns fraud probability, risk level, and top SHAP features per prediction. Auto-documented via Swagger UI.

**Drift Monitoring** — Evidently AI compares training vs production data distributions using statistical tests (Wasserstein distance for numerical, Chi-squared for categorical). Triggers retraining alerts when drift is detected.

**CI/CD** — GitHub Actions runs 10 synthetic tests on every push — validating preprocessing logic, feature engineering correctness, and model behavior without requiring the full dataset.

---

## Screenshots

| MLflow Experiment Tracking | SHAP Summary Plot |
|---|---|
| All 3 runs logged with metrics | Top 20 features by mean \|SHAP\| |

| Streamlit Dashboard | Evidently Drift Report |
|---|---|
| Real-time fraud prediction + explanations | Feature distribution shift detection |

---


## License

MIT License — see [LICENSE](LICENSE) for details.
