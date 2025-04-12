import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

# Define project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path("/Users/pragatimehra/Desktop/project/server/modules/non_fraud_transactions.csv")
PREPROCESSOR_PATH = BASE_DIR / "modules/fraud_preprocessor.pkl"
MODEL_PATH = BASE_DIR / "modules/fraud_autoencoder_model.keras"
THRESHOLD_PATH = BASE_DIR / "modules/fraud_threshold.txt"

def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"❌ Data not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    return df

def build_preprocessor(df, features):
    numeric_features = df[features].select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df[features].select_dtypes(include=["object", "bool"]).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    bottleneck = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(bottleneck)
    decoded = Dense(64, activation='relu')(decoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_autoencoder():
    print("📦 Loading data...")
    df = load_data()

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

    X = df[features_used].copy()

    print("🔧 Building preprocessor...")
    preprocessor, num_feats, cat_feats = build_preprocessor(X, features_used)

    print("⚙️ Fitting preprocessor and transforming data...")
    X_processed = preprocessor.fit_transform(X)

    print("🧠 Building autoencoder...")
    autoencoder = build_autoencoder(X_processed.shape[1])

    print("🚀 Training model...")
    es = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    autoencoder.fit(
        X_processed, X_processed,
        epochs=50,
        batch_size=128,
        shuffle=True,
        callbacks=[es],
        verbose=1
    )

    print("📉 Calculating reconstruction error threshold...")
    reconstructions = autoencoder.predict(X_processed)
    mse = np.mean(np.power(X_processed - reconstructions, 2), axis=1)
    threshold = np.percentile(mse, 95)

    print(f"✅ Threshold set at: {threshold:.6f}")

    print("💾 Saving model and preprocessor...")
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    autoencoder.save(MODEL_PATH)
    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(threshold))

    print(f"✅ Model saved to: {MODEL_PATH}")
    print(f"✅ Preprocessor saved to: {PREPROCESSOR_PATH}")
    print(f"✅ Threshold saved to: {THRESHOLD_PATH}")

if __name__ == "__main__":
    train_autoencoder()
