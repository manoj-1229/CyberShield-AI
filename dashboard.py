import os
import requests
import joblib
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import streamlit as st

st.set_page_config(page_title="CyberShield AI | SOC", page_icon="🛡️", layout="wide")

st.title("🛡️ CyberShield AI: Security Operations Center")
st.markdown("Real-time network intrusion detection and automated MLOps data drift monitoring.")

tab1, tab2 = st.tabs(["🚀 Live Threat Assessment", "📊 Automated Model & Drift Monitoring"])

# Reference data loading
ref_df = None
if os.path.exists("models/reference_data.pkl"):
    ref_df = joblib.load("models/reference_data.pkl")

# TAB 1: Real-Time Threat Assessment
with tab1:
    st.subheader("Simulate Network Traffic Ingestion")
    col1, col2 = st.columns([1.2, 1.8])
    
    with col1:
        if st.button("Generate Network Packet", type="primary"):
            if ref_df is not None:
                st.session_state["packet"] = ref_df.sample(1, random_state=int(np.random.randint(0, 10000))).to_dict(orient="records")[0]
            else:
                st.error("Missing reference data. Please verify models/reference_data.pkl exists.")
                
        if "packet" in st.session_state:
            st.write("**Intercepted Packet Summary:**")
            st.dataframe(pd.DataFrame([st.session_state["packet"]]).iloc[:, :6], hide_index=True)

    with col2:
        if "packet" in st.session_state:
            try:
                res = requests.post("http://127.0.0.1:8000/predict", json=st.session_state["packet"]).json()
                st.write("### Inference Verdict")
                if res.get("prediction") == 1:
                    st.error("🚨 **MALICIOUS ATTACK DETECTED**")
                else:
                    st.success("✅ **BENIGN TRAFFIC**")
                st.metric("Threat Probability", f"{res.get('threat_probability', 0) * 100:.2f}%")
            except requests.exceptions.ConnectionError:
                st.warning("FastAPI backend is offline. Start it using: uvicorn app.main:app --reload")

# TAB 2: Automated Statistical Drift Monitoring
with tab2:
    st.subheader("Automated Data Drift & Feature Distribution Analysis")
    st.markdown("Automated two-sample Kolmogorov-Smirnov (KS) test comparing baseline network distribution against incoming production traffic.")
    
    if ref_df is not None:
        key_numerical_features = [col for col in ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate'] if col in ref_df.columns]
        
        # Simulate production batch with distribution shift
        np.random.seed(42)
        prod_df = ref_df.sample(250, random_state=42).copy()
        prod_df['dur'] = prod_df['dur'] * np.random.uniform(1.2, 1.8, size=len(prod_df))
        prod_df['sbytes'] = prod_df['sbytes'] * np.random.uniform(1.1, 1.5, size=len(prod_df))
        
        drift_results = []
        for feat in key_numerical_features:
            stat, p_value = ks_2samp(ref_df[feat].dropna(), prod_df[feat].dropna())
            drift_detected = p_value < 0.05
            drift_results.append({
                "Feature": str(feat),
                "KS Statistic": f"{stat:.4f}",
                "p-value": f"{p_value:.5f}",
                "Status": "⚠️ Drift Detected" if drift_detected else "✅ In Bounds"
            })

        drift_summary_df = pd.DataFrame(drift_results)[["Feature", "KS Statistic", "p-value", "Status"]]
        
        col_m1, col_m2, col_m3 = st.columns(3)
        drift_count = sum(1 for r in drift_results if "Drift" in r["Status"])
        col_m1.metric("Features Monitored", len(key_numerical_features))
        col_m2.metric("Drifted Features", drift_count)
        col_m3.metric("Dataset Drift Status", "ACTION REQUIRED" if drift_count > 0 else "STABLE")
        
        st.write("#### Statistical Drift Test Table")
        st.dataframe(drift_summary_df, use_container_width=True, hide_index=True)
        
        st.write("#### Feature Distribution Comparison")
        selected_feature = st.selectbox("Select feature to inspect:", key_numerical_features)
        
        chart_data = pd.DataFrame({
            "Baseline (Training)": ref_df[selected_feature].sample(200, random_state=42).values,
            "Production (Recent)": prod_df[selected_feature].sample(200, random_state=42).values
        })
        st.line_chart(chart_data)
    else:
        st.warning("Reference dataset not found in models/ directory. Run train.py first.")