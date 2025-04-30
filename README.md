# 🛡️ AI-Driven Fraud Detection & Drift Monitoring System

This project simulates a real-time fraud detection pipeline using synthetic data. It uses Kafka for streaming, Spark for processing, FastAPI for serving predictions, and Streamlit for model/data drift monitoring.

---

# Reference Screenshots

## Dashboard
<img width="1440" alt="Screenshot 2025-04-18 at 11 41 04 AM" src="https://github.com/user-attachments/assets/98f808f9-0fdb-4cc9-b5e5-dc2fabcbfb55" />

## Drift Monitoring
<img width="1440" alt="Screenshot 2025-04-18 at 11 40 08 AM" src="https://github.com/user-attachments/assets/074a7535-022e-4bfc-839a-48b80f9fe704" />

## Evidently
<img width="1440" alt="Screenshot 2025-04-18 at 11 40 50 AM" src="https://github.com/user-attachments/assets/ddea60d0-0ed5-4740-83f3-09af2f0d3c0c" />


## 📁 Project Setup

### 1. Clone the Repository

```bash
cd Desktop
git clone https://github.com/pragatimehra/ai-fraud-detection-monitoring.git
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
pip install faker pandas numpy kafka-python pyspark fastapi uvicorn python-dotenv polars scikit-learn matplotlib joblib tensorflow pydantic streamlit google-generativeai evidently
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

## 📬 Configure Email Alerts and Gemini Key

### 8. Create `.env` File in `server/` Folder

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

### 9. Create `.env` File in `modules/` Folder

```env
GEMINI_API_KEY=your_gemini_key
```

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

## 🎁 Bonus: Reset Kafka Topics (Start Fresh)
To avoid appending records and ensure a clean start, you can delete existing Kafka topics:

### 1. List All Kafka Topics
```bash
cd server
python3 kafka_list.py
```
This script will display all the current Kafka topics.

### 2. Delete All Kafka Topics
```bash
python3 kafka_dlt_topics.py
```
This will delete all listed Kafka topics, allowing you to restart your pipeline from a clean slate.

---

## 👨‍💻 Tech Stack

- Python
- Kafka + Docker
- Apache Spark (PySpark)
- FastAPI
- TensorFlow & Scikit-learn
- Streamlit
- Faker (synthetic data generation)
