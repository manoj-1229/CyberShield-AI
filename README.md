# 🛡️ CyberShield AI: Real-Time Network Threat Detection & MLOps Monitoring

An end-to-end MLOps intrusion detection platform that combines machine learning inference with statistical model monitoring to protect network infrastructure.

---

## 🏗️ Architecture Overview

1. **Model Pipeline (`train.py`)**:

   - Dataset: UNSW-NB15 network traffic dataset.
   - Preprocessing: Standard feature scaling and SMOTE balancing to handle class imbalance.
   - Algorithm: Extreme Gradient Boosting (XGBoost).
   - Artifacts generated: `cybershield_xgb.pkl`, `scaler.pkl`, `feature_columns.pkl`, `reference_data.pkl`.
2. **Inference Backend (`app/main.py`)**:

   - Built on FastAPI with asynchronous routing.
   - Receives network packet telemetry via `/predict` POST requests.
   - Applies trained transformations and returns threat classifications along with prediction probabilities.
3. **Security Operations Center Dashboard (`dashboard.py`)**:

   - Built on Streamlit.
   - **Tab 1 — Live Threat Assessment**: Interactive packet generation, API payload submission, and real-time security verdict alerts.
   - **Tab 2 — Automated Model & Drift Monitoring**: Native Kolmogorov-Smirnov (KS) statistical drift testing across key numerical indicators (`dur`, `sbytes`, `rate`, etc.) with distribution shift visualizations.

---

## 🚀 Setup & Execution

### 1. Clone & Install Dependencies

```bash
git clone <repository_url>
cd CyberShield_AI
pip install -r requirements.txt
```

# CyberShield AI — Starter Project

Starter pipeline: FastAPI prediction API, Random Forest demo model, Streamlit dashboard, and Docker.

**Important:** `train.py` uses synthetic, rule-generated data solely to verify the software pipeline. Metrics from it are not evidence of real-world detection performance. The app analyzes submitted flow features; it does not capture live packets.

## Setup (Windows PowerShell)

```powershell
cd CyberShield_AI
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python train.py
```

Terminal 1 — API:

```powershell
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs

Terminal 2 — dashboard:

```powershell
streamlit run dashboard.py
```

Example POST body for `/predict`:

```json
{"duration":2,"src_bytes":500,"dst_bytes":700,"failed_logins":0,"same_srv_rate":0.4,"serror_rate":0.1,"count":10}
```

## Docker

```powershell
docker build -t cybershield-ai .
docker run --rm -p 8000:8000 cybershield-ai
```

## Roadmap

1. Replace synthetic data with a documented intrusion-detection dataset and map its features carefully.
2. Evaluate using precision, recall, F1, and a confusion matrix.
3. Add prediction logging with privacy safeguards and model versioning.
4. Add Evidently AI drift reports and dashboard monitoring.
5. Add tests and CI.
6. Consider live traffic only in an authorized lab environment.
