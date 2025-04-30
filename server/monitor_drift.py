import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime
from email.message import EmailMessage
import smtplib
from dotenv import load_dotenv
import os
import joblib
from tensorflow.keras.models import load_model
import uuid
from faker import Faker
import threading
import polars as pl
import random
from modules.train_autoencoder import train_autoencoder  # <-- Import retraining function
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


st.set_page_config(page_title="Drift Monitor", layout="wide")

# Load env variables
load_dotenv()
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
TO_EMAIL = os.getenv("TO_EMAIL")

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "modules/fraud_autoencoder_model.keras"
PREPROCESSOR_PATH = BASE_DIR / "modules/fraud_preprocessor.pkl"
THRESHOLD_PATH = BASE_DIR / "modules/fraud_threshold.txt"
HISTORY_LOG = BASE_DIR / "modules/monitoring_history.json"
TEMP_DATA_PATH = BASE_DIR / "modules/tmp_monitor_data.csv"
ENRICHED_DATA_PATH = BASE_DIR / "modules/denoised_enriched_transactions.csv"

DRIFT_THRESHOLD = 0.1
EMAIL_ALERT_THRESHOLD = 0.1
AUTO_RUN_INTERVAL = 30  # seconds
RETRAIN_WAIT_TIME = 120  # 2 minutes in seconds

# ----------- Generate Realistic Synthetic Data  -----------
def generate_data(num_customers=10, accounts_per_customer=10, num_transactions=1000, anomaly_prob=0.05):
    fake = Faker()
    np.random.seed(int(time.time()))

    customers, accounts, transactions = [], [], []
    account_balances = {}
    anomaly_types = [
        "simulated_high_amount", "simulated_high_risk_country",
        "simulated_account_draining", "simulated_unusual_type",
        "simulated_blacklisted_merchant"
    ]
    blacklisted_merchants = ['Vision Corp', 'Pinnacle Ltd', 'Omega LLC']
    high_risk_countries = ['North Korea', 'Syria', 'Iran']
    anomaly_indices = set(np.random.choice(num_transactions, int(num_transactions * anomaly_prob), replace=False))

    for _ in range(num_customers):
        cust_id = str(uuid.uuid4())
        customers.append({
            "customer_id": cust_id,
            "name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "address": fake.address().replace("\n", ", "),
            "dob": fake.date_of_birth(minimum_age=18, maximum_age=85).isoformat(),
            "created_at": fake.date_time_this_decade().isoformat()
        })

        for _ in range(accounts_per_customer):
            acct_id = str(uuid.uuid4())
            acct_type = fake.random_element(["savings", "checking", "credit"])
            bal = np.random.uniform(1000, 100000)
            accounts.append({
                "account_id": acct_id,
                "customer_id": cust_id,
                "account_type": acct_type,
                "account_number": fake.unique.bban(),
                "created_at": fake.date_time_this_decade().isoformat(),
                "balance": round(bal, 2)
            })
            account_balances[acct_id] = bal

    df_customers = pd.DataFrame(customers)
    df_accounts = pd.DataFrame(accounts)
    df_accounts.set_index("account_id", inplace=True)

    for i in range(num_transactions):
        acct = df_accounts.sample(1)
        acct_id = acct.index[0]
        account_type = acct["account_type"].values[0]

        t_type = fake.random_element(["purchase", "transfer", "withdrawal", "deposit", "payment"])
        amount = round(np.random.uniform(5, 2000), 2)
        is_foreign = np.random.rand() < 0.1
        is_high_risk_country = is_foreign and np.random.rand() < 0.5
        merchant = fake.company()
        location = fake.city()

        is_fraud = False
        fraud_reasons = []

        if i in anomaly_indices:
            is_fraud = True
            anomaly = np.random.choice(anomaly_types)
            if anomaly == "simulated_high_amount":
                amount = round(np.random.uniform(50000, 100000), 2)
                fraud_reasons.append(anomaly)
            elif anomaly == "simulated_high_risk_country":
                is_foreign, is_high_risk_country = True, True
                location = np.random.choice(high_risk_countries)
                fraud_reasons.append(anomaly)
            elif anomaly == "simulated_account_draining":
                amount = round(np.random.uniform(0.9 * account_balances[acct_id], account_balances[acct_id]), 2)
                t_type = "withdrawal"
                fraud_reasons.append(anomaly)
            elif anomaly == "simulated_unusual_type":
                t_type = "ATM Withdrawal" if account_type == "credit" else "Unusual Transaction"
                fraud_reasons.append(anomaly)
            elif anomaly == "simulated_blacklisted_merchant":
                merchant = np.random.choice(blacklisted_merchants)
                fraud_reasons.append(anomaly)

        opening = account_balances[acct_id]
        closing = opening + amount if t_type in ["deposit", "payment"] else opening - amount
        account_balances[acct_id] = closing

        transactions.append({
            "transaction_id": str(uuid.uuid4()),
            "account_id": acct_id,
            "account_type": account_type,
            "timestamp": fake.date_time_this_year().isoformat(),
            "amount": amount,
            "transaction_type": t_type,
            "merchant": merchant,
            "location": location,
            "is_foreign": is_foreign,
            "is_high_risk_country": is_high_risk_country,
            "opening_balance": round(opening, 2),
            "closing_balance": round(closing, 2),
            "is_fraud": is_fraud,
            "fraud_reasons": ", ".join(fraud_reasons)
        })

    df_accounts.reset_index(inplace=True)
    df_transactions = pd.DataFrame(transactions)
    df_transactions["account_type_txn"] = df_transactions["account_type"]  # Added to match preprocessor

    df = df_transactions.merge(df_accounts.rename(columns={
        "account_type": "account_type_acct",
        "created_at": "created_at_acct"
    }), on="account_id", how="inner").merge(df_customers.rename(columns={
        "created_at": "created_at_cust"
    }), on="customer_id", how="inner")

    df = df.drop_duplicates(subset="transaction_id")
    df.to_csv(TEMP_DATA_PATH, index=False)

    # Enrichment via Polars (transformation logic)
    df_pl = pl.read_csv(TEMP_DATA_PATH).sort("account_id").sort("timestamp")
    agg = df_pl.group_by("account_id").agg([
        pl.col("amount").count().alias("agg_txn_count"),
        pl.col("amount").mean().alias("agg_avg_amount"),
        pl.col("amount").std().alias("agg_std_amount"),
        pl.col("amount").max().alias("agg_max_amount"),
        pl.col("merchant").n_unique().alias("agg_unique_merchants"),
        pl.col("location").n_unique().alias("agg_unique_locations")
    ])
    df_pl = df_pl.join(agg, on="account_id", how="left")
    df_pl = df_pl.with_columns([
        (pl.col("amount").cum_sum().over("account_id") - pl.col("amount")).alias("cumulative_amount"),
        (pl.col("amount").count().over("account_id") - 1).alias("past_txn_count")
    ])
    df_pl = df_pl.with_columns((pl.col("cumulative_amount") / pl.when(pl.col("past_txn_count") > 0)
                                .then(pl.col("past_txn_count")).otherwise(1)).alias("past_avg_amount"))

    def get_past_mode(values):
        history = []
        for i in range(len(values)):
            if i == 0:
                history.append("None")
            else:
                freq = {}
                for v in values[:i]:
                    freq[v] = freq.get(v, 0) + 1
                history.append(max(freq, key=freq.get))
        return history

    groups = []
    for group in df_pl.partition_by("account_id", as_dict=False):
        merchants = group["merchant"].to_list()
        locations = group["location"].to_list()
        group = group.with_columns([
            pl.Series("past_common_merchant", get_past_mode(merchants)),
            pl.Series("past_common_location", get_past_mode(locations))
        ])
        groups.append(group)

    df_final = pl.concat(groups).drop("cumulative_amount").with_columns([
        pl.col("past_avg_amount").fill_null(0),
        pl.col("past_txn_count").fill_null(0),
        pl.col("past_common_merchant").fill_null("None"),
        pl.col("past_common_location").fill_null("None")
    ])

    df_final.write_csv(ENRICHED_DATA_PATH)
    return pd.read_csv(ENRICHED_DATA_PATH)

# ----------- Rule-Based Detection, History, Autoencoder Scoring -----------

def complete_fraud_pipeline(df):
    # RULE-BASED FRAUD DETECTION
    blacklisted_merchants = ['Vision Corp', 'Pinnacle Ltd', 'Omega LLC']
    blacklisted_locations = ['Lakeview', 'Springfield', 'Newport']
    df['flag_low_balance'] = df['closing_balance'] < 1000
    df['flag_blacklisted_merchant'] = df['merchant'].isin(blacklisted_merchants)
    df['flag_blacklisted_location'] = df['location'].isin(blacklisted_locations)
    df['flag_high_amount'] = df['amount'] > df['amount'].quantile(0.95)
    df['flag_high_risk_foreign'] = (df['is_foreign']) & (df['is_high_risk_country'])
    df['flag_credit_withdrawal'] = (
        (df['account_type_acct'] == 'credit') &
        (df['transaction_type'].isin(['withdrawal', 'payment'])) &
        (df['amount'] > 3000)
    )

    flag_columns = [
        'flag_low_balance', 'flag_blacklisted_merchant', 'flag_blacklisted_location',
        'flag_high_amount', 'flag_high_risk_foreign', 'flag_credit_withdrawal'
    ]

    df['rule_predicted_fraud'] = df[flag_columns].any(axis=1).astype(int)

    # ACCOUNT-LEVEL HISTORY
    history = df.groupby("account_id").agg({
        "transaction_id": "count",
        "amount": ["mean", "std", "max", "min", "sum"],
        "opening_balance": "mean",
        "closing_balance": ["mean", "max"],
        "transaction_type": pd.Series.nunique,
        "merchant": pd.Series.nunique,
        "location": pd.Series.nunique
    })
    history.columns = ['_'.join(col).strip() for col in history.columns.values]
    history.reset_index(inplace=True)
    df = df.merge(history, on="account_id", how="left")

    # AUTOENCODER-BASED FRAUD DETECTION
    if not (MODEL_PATH.exists() and PREPROCESSOR_PATH.exists() and THRESHOLD_PATH.exists()):
        st.warning("🔁 Model artifacts not found. Skipping autoencoder detection.")
        return df, 0.0

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = load_model(MODEL_PATH, compile=False)
    with open(THRESHOLD_PATH) as f:
        threshold = float(f.read())

    features_used = [
        "transaction_id", "timestamp", "amount", "transaction_type", "merchant",
        "location", "is_foreign", "is_high_risk_country", "opening_balance",
        "closing_balance", "account_id", "account_type", "account_type_txn",
        "account_type_acct", "account_number", "balance", "created_at_acct",
        "customer_id", "name", "email", "phone", "address", "dob", "created_at_cust",
        "past_txn_count", "past_avg_amount", "past_common_merchant", "past_common_location",
        "agg_txn_count", "agg_avg_amount", "agg_std_amount", "agg_max_amount",
        "agg_unique_merchants", "agg_unique_locations"
    ]

    df_model = df[features_used].copy().dropna()
    if df_model.empty:
        return df, 0.0

    X = preprocessor.transform(df_model)
    reconstructions = model.predict(X)
    mse = np.mean(np.power(X - reconstructions, 2), axis=1)

    df.loc[df_model.index, "anomaly_score"] = mse
    df.loc[df_model.index, "predicted_fraud"] = (mse > threshold).astype(int)
    fraud_ratio = df["predicted_fraud"].mean()

    return df, fraud_ratio


# ---------- Drift Monitoring Utilities ----------

def send_alert_email(fraud_ratio):
    try:
        msg = EmailMessage()
        msg["Subject"] = "⚠️ Model Drift Detected"
        msg["From"] = EMAIL_USER
        msg["To"] = TO_EMAIL
        msg.set_content(f"Fraud ratio reached {fraud_ratio:.2%}. Model retraining may be needed.")

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        print("📧 Alert email sent.")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")


def append_to_history_log(fraud_ratio):
    HISTORY_LOG.parent.mkdir(exist_ok=True)
    if HISTORY_LOG.exists():
        with open(HISTORY_LOG, "r") as f:
            history = json.load(f)
    else:
        history = []

    history.append({
        "timestamp": datetime.now().isoformat(),
        "fraud_ratio": fraud_ratio
    })
    history = history[-100:]

    with open(HISTORY_LOG, "w") as f:
        json.dump(history, f, indent=2)


def load_history():
    if HISTORY_LOG.exists():
        with open(HISTORY_LOG, "r") as f:
            return json.load(f)
    return []

# ---------- Streamlit UI ----------
st.title("📡 Real-Time Drift Monitor")

if "history" not in st.session_state:
    st.session_state.history = []

df = generate_data()

REFERENCE_PATH = BASE_DIR / "modules/non_fraud_transactions_reference.csv"

# Save reference once if it doesn't exist
if not REFERENCE_PATH.exists():
    df.to_csv(REFERENCE_PATH, index=False)
    print("✅ Reference dataset saved for drift comparison.")

df, fraud_ratio = complete_fraud_pipeline(df)

# --- Run Evidently Data Drift Report ---
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

reference_df = pd.read_csv(REFERENCE_PATH)
current_df = df.copy()

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference_df, current_data=current_df)

EVIDENTLY_REPORT_PATH = BASE_DIR / "modules/evidently_drift_report.html"
report.save_html(str(EVIDENTLY_REPORT_PATH))
print(f"📊 Evidently report saved: {EVIDENTLY_REPORT_PATH}")
st.success(f"📄 [View Evidently Drift Report](modules/evidently_drift_report.html)")

# ✅ Now it's safe to use the result
result_dict = report.as_dict()
evidently_drift_detected = result_dict["metrics"][0]["result"]["dataset_drift"]
st.write(f"📊 Evidently Drift Detected: {evidently_drift_detected}")

# Optional: retrain based on Evidently drift
if evidently_drift_detected and fraud_ratio > DRIFT_THRESHOLD:
    st.warning("🚨 Evidently & Fraud thresholds exceeded. Retraining model.")
    train_autoencoder()

append_to_history_log(fraud_ratio)

st.session_state.history.append({
    "timestamp": datetime.now().isoformat(),
    "fraud_ratio": fraud_ratio
})
st.session_state.history = st.session_state.history[-100:]

# Draw chart
timestamps = [entry["timestamp"] for entry in st.session_state.history]
ratios = [entry["fraud_ratio"] for entry in st.session_state.history]
st.line_chart(pd.DataFrame({"Drift Ratio": ratios}, index=pd.to_datetime(timestamps)))
st.info(f"🔁 Last run at {timestamps[-1]} — Drift: {ratios[-1]:.2%}")

import streamlit.components.v1 as components
EVIDENTLY_REPORT_PATH = BASE_DIR / "modules/evidently_drift_report.html"

if EVIDENTLY_REPORT_PATH.exists():
    with open(EVIDENTLY_REPORT_PATH, "r") as f:
        html = f.read()
    st.subheader("📊 Evidently Data Drift Report")
    components.html(html, height=800, scrolling=True)
else:
    st.warning("❗ Evidently report not found yet.")

# Send alerts & retrain if needed
if fraud_ratio > EMAIL_ALERT_THRESHOLD:
    send_alert_email(fraud_ratio)
    st.warning(f"📧 Alert: Fraud ratio reached {fraud_ratio:.2%}. Email notification has been sent.")

if fraud_ratio > DRIFT_THRESHOLD:
    send_alert_email(fraud_ratio)
    st.warning("🚨 Drift threshold exceeded. Automatic model retraining triggered.")
    with st.status("🛠️ Retraining model... Please wait 3 minutes", expanded=True):
        train_autoencoder()
        st.write("✅ Retraining complete. Cooling down...")
        st.write(f"⏳ Waiting {RETRAIN_WAIT_TIME // 60} minutes before next run...")
        time.sleep(RETRAIN_WAIT_TIME)

# Refresh every 60s
time.sleep(AUTO_RUN_INTERVAL)
st.rerun()
