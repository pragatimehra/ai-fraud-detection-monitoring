import os
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

def run_autoencoder_fraud_detection():
    print("🤖 Running autoencoder-based fraud detection...")

    # Define base paths
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "modules/non_fraud_transactions.csv"
    model_path = base_dir / "modules/fraud_autoencoder_model.keras"
    preprocessor_path = base_dir / "modules/fraud_preprocessor.pkl"
    threshold_path = base_dir / "modules/fraud_threshold.txt"
    output_path = base_dir / "modules/fraud_cases_for_llm.csv"

    # Check file existence
    if not csv_path.exists():
        print(f"⚠️ Skipping autoencoder: '{csv_path}' not found.")
        return
    if not model_path.exists() or not preprocessor_path.exists() or not threshold_path.exists():
        print("❌ Model, preprocessor, or threshold file missing.")
        return

    # Load data and model components
    df = pd.read_csv(csv_path)
    preprocessor = joblib.load(preprocessor_path)
    autoencoder = load_model(model_path)

    with open(threshold_path, "r") as f:
        threshold = float(f.read().strip())

    print(f"✅ Loaded threshold: {threshold:.6f}")

    # Features used
    features_used = [
        "transaction_id", "timestamp", "amount", "transaction_type", "merchant",
        "location", "is_foreign", "is_high_risk_country", "opening_balance",
        "closing_balance", "account_id", "account_type_txn", "account_type_acct",
        "account_number", "balance", "created_at_acct", "customer_id", "name",
        "email", "phone", "address", "dob", "created_at_cust",
        "past_txn_count", "past_avg_amount", "past_common_merchant", "past_common_location",
        "agg_txn_count", "agg_avg_amount", "agg_std_amount", "agg_max_amount",
        "agg_unique_merchants", "agg_unique_locations"
    ]

    # Transform input
    X = df[features_used].copy()
    X_processed = preprocessor.transform(X)

    # Validate shape
    if X_processed.shape[1] != autoencoder.input_shape[1]:
        raise ValueError(
            f"❌ Feature mismatch: transformed shape {X_processed.shape[1]} ≠ model input {autoencoder.input_shape[1]}"
        )

    # Predict anomaly scores
    reconstructions = autoencoder.predict(X_processed)
    mse = np.mean(np.power(X_processed - reconstructions, 2), axis=1)
    predicted_fraud = (mse > threshold).astype(int)

    # Save results
    df["anomaly_score"] = mse
    df["predicted_fraud"] = predicted_fraud
    fraud_cases = df[df["predicted_fraud"] == 1]

    fraud_cases.to_csv(output_path, index=False)
    print(f"✅ Saved {len(fraud_cases)} predicted frauds to '{output_path}'")

if __name__ == "__main__":
    run_autoencoder_fraud_detection()