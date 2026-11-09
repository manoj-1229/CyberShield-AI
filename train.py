import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import mlflow

def run_training():
    os.makedirs("models", exist_ok=True)
    mlflow.set_experiment("CyberShield_Intrusion_Detection")
    
    with mlflow.start_run(run_name="XGBoost_Production_Run"):
        print("Loading dataset...")
        df = pd.read_csv("data/UNSW_NB15_training-set.csv")
        
        # Drop non-predictive IDs and multiclass labels
        X = df.drop(columns=["id", "attack_cat", "label"], errors="ignore")
        y = df["label"]
        
        # One-hot encode categorical features
        X_encoded = pd.get_dummies(X, drop_first=True)
        feature_columns = list(X_encoded.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
        
        print("Scaling features and applying SMOTE...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        
        params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": 42}
        mlflow.log_params(params)
        
        print("Training XGBoost Classifier...")
        model = XGBClassifier(**params)
        model.fit(X_train_balanced, y_train_balanced)
        
        print("Evaluating Model...")
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        roc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric("roc_auc", roc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        
        print(f"Results: ROC-AUC={roc:.4f} | F1={f1:.4f}")
        
        # Save artifacts for the API and Dashboard
        joblib.dump(model, "models/cybershield_xgb.pkl")
        joblib.dump(scaler, "models/scaler.pkl")
        joblib.dump(feature_columns, "models/feature_columns.pkl")
        
        # Save reference data required by the dashboard for packet simulation
        joblib.dump(pd.DataFrame(X_train, columns=feature_columns).sample(500, random_state=42), "models/reference_data.pkl")
        
        mlflow.log_artifacts("models")
        print("Pipeline complete. Artifacts saved in /models.")

if __name__ == "__main__":
    run_training()