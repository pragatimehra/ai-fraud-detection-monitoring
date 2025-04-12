# 🛡️ AI-Driven Fraud Detection & Drift Monitoring System

This project simulates a real-time fraud detection pipeline using synthetic data. It uses Kafka for streaming, Spark for processing, FastAPI for serving predictions, and Streamlit for model/data drift monitoring.

---

## 📁 Project Setup

### 1. Clone the Repository

```bash
cd Desktop
git clone https://github.com/pragatimehra/project.git
cd project
code .
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install faker pandas numpy kafka-python pyspark fastapi uvicorn python-dotenv polars scikit-learn matplotlib joblib tensorflow pydantic streamlit google-generativeai
```

---

## ⚙️ Kafka Setup

### 4. Start Kafka Using Docker

```bash
cd server/modules/
docker compose up -d
```

---

## 📊 Generate and Consume Data

### 5. Run Data Generator

```bash
python3 datagen.py
```

This will output:

```
✅ 1000 transactions generated.
🚨 50 anomalous transactions injected.
```

And generate:

- `accounts.csv`
- `customer.csv`
- `transactions.csv`

### 6. Run Kafka Consumer to Create Parquet Files

```bash
python3 consumer.py
```

Once logs confirm processing, stop it with `Ctrl+C`.

Generated folders:

- `parquet/accounts/`
- `parquet/customers/`
- `parquet/transactions/`

---

## 🚀 Launch the FastAPI Server

### 7. Run API Server

```bash
cd ..
python3 run.py
```

Example Output:

```
🚀 Starting FastAPI server... Running data processing pipeline.
✅ Deduplicated dataset saved to 'denormalized_transactions'.
✅ Historical features added and saved to 'denoised_enriched_transactions.csv'.
✅ 140 fraud transactions saved.
✅ 860 non-fraud transactions saved.
✅ Account-level history saved.
🤖 Running autoencoder-based fraud detection...
✅ Saved predicted frauds to 'fraud_cases_for_llm.csv'
```

---

## 📬 Configure Email Alerts

### 8. Create `.env` File in `modules/` Folder

```env
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_16_character_app_password
TO_EMAIL=recipient_email@gmail.com
GEMINI_API_KEY=your_gemini_key
```

To generate the app password:

1. Go to https://myaccount.google.com/apppasswords  
2. Select **Mail** and **Other (e.g., "Streamlit Drift")**  
3. Copy the 16-character password and paste it into `.env`

To generate free gemini key:

1. Go to https://aistudio.google.com/apikey
2. Select **Create API Key**
3. Select a project from your existing Google Cloud projects 
4. Click **Create API Key in existing project**
5. Copy the key and paste it into `.env`

---

## 📉 Monitor Drift & Auto Retrain

### 9. Run Streamlit Dashboard

```bash
streamlit run monitor_drift.py
```

Features:

- Toggle drift simulation
- Control drift intensity via slider
- Automatic model retraining
- Email alert if performance drops

---

## ✅ Summary of Generated Artifacts

| File                          | Description                            |
|-------------------------------|----------------------------------------|
| `accounts.csv`                | Synthetic account data                 |
| `customers.csv`               | Synthetic customer data                |
| `transactions.csv`            | Synthetic transactions                 |
| `parquet/`                    | Kafka-consumed Parquet files           |
| `denormalized_transactions/` | Processed transaction data             |
| `fraud_transactions.csv/json`| Identified fraud cases                 |
| `non_fraud_transactions.csv` | Legit transactions                     |
| `fraud_cases_for_llm.csv`    | Predicted frauds for explanation       |
| `account_history.csv`        | Account-level historical summary       |
| `fraud_autoencoder_model.h5` | Trained anomaly detection model        |
| `fraud_preprocessor.pkl`     | Preprocessing pipeline                 |
| `fraud_threshold.txt`        | Threshold used for fraud scoring       |

---

## 👨‍💻 Tech Stack

- Python
- Kafka + Docker
- Apache Spark (PySpark)
- FastAPI
- TensorFlow & Scikit-learn
- Streamlit
- Faker (synthetic data generation)

---

## 📩 Contributions

Feel free to fork, enhance, or open issues!  
Happy Hacking! 🚀
