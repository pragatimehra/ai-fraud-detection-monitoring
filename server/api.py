from pathlib import Path
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import os
import pandas as pd
import json
from datetime import datetime

# Module imports
from modules.combine import merge_parquet_to_csv
from modules.transformation import enrich_with_historical_features
from modules.rule_based_fraud_detection import apply_rule_based_fraud_detection
from modules.history import generate_account_level_history
from modules.model import run_autoencoder_fraud_detection
from modules.generate_fraud_explanations import generate_fraud_explanations

fraud_explanations_cache = []

# Initialize FastAPI app
app = FastAPI(title="Sentinel Fraud API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with allowed origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------- Models ----------------------------

class FeedbackCreate(BaseModel):
    category: str
    details: str

class TransactionFeedback(BaseModel):
    transaction_id: str
    is_correct: bool
    feedback: Optional[str] = None

class Transaction(BaseModel):
    id: str
    timestamp: str
    amount: float
    accountNumber: str
    transactionType: str
    score: float
    reason: str

class LoginData(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    success: bool
    message: Optional[str] = None

# ---------------------------- Auth ----------------------------

DEMO_USERS = {
    "admin": {
        "password": "admin123",
        "name": "Admin User"
    }
}

# ------------------------ Utility Functions ------------------------

def load_fraud_explanations(path: str) -> list:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ File not found: {path}")
        return []

def append_to_fraud_explanations(new_fraud_json_path: str, full_explanations_path: str):
    try:
        with open(new_fraud_json_path, "r") as f:
            new_frauds = json.load(f)

        if Path(full_explanations_path).exists():
            with open(full_explanations_path, "r") as f:
                existing_explanations = json.load(f)
        else:
            existing_explanations = []

        existing_ids = {entry["id"] for entry in existing_explanations}
        new_entries = [tx for tx in new_frauds if tx["id"] not in existing_ids]

        print(f"➕ Appending {len(new_entries)} new fraud entries to explanations JSON")

        combined = existing_explanations + new_entries

        with open(full_explanations_path, "w") as f:
            json.dump(combined, f, indent=2)

        print(f"✅ Total entries in '{full_explanations_path}': {len(combined)}")

    except Exception as e:
        print(f"❌ Failed to append to fraud explanations: {e}")

# ------------------------ Server Startup ------------------------

@app.on_event("startup")
def startup_event():
    global fraud_explanations_cache

    print("🚀 Starting FastAPI server... Running data processing pipeline.")
    
    output_dir = merge_parquet_to_csv()
    enrich_with_historical_features(output_dir)

    input_csv = os.path.join(output_dir, "denoised_enriched_transactions.csv")
    apply_rule_based_fraud_detection(input_csv)

    generate_account_level_history(output_dir)
    run_autoencoder_fraud_detection()
    generate_fraud_explanations() 

    # Append new frauds to the full JSON
    base_dir = Path(__file__).resolve().parent
    fraud_json_path = base_dir / "modules/fraud_transactions.json"
    full_json_path = base_dir / "modules/fraud_explanations_full.json"
    append_to_fraud_explanations(fraud_json_path, full_json_path)

    # Load full data into memory for API
    fraud_explanations_cache = load_fraud_explanations(full_json_path)
    print(f"✅ Loaded {len(fraud_explanations_cache)} fraud transactions into memory.")

# ------------------------ Routes ------------------------

@app.get("/")
def read_root():
    return {"message": "FastAPI is running with Spark integration"}

@app.post("/api/auth/login", response_model=LoginResponse)
async def login(data: LoginData):
    print(f"Login attempt: username={data.username}")
    user = DEMO_USERS.get(data.username)
    if user and user["password"] == data.password:
        return {"success": True}
    return {"success": False, "message": "Invalid credentials"}

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}

@app.get("/api/transactions", response_model=List[Transaction])
async def get_transactions():
    return fraud_explanations_cache

# Feedback
system_feedback = []
transaction_feedback = []

@app.post("/api/feedback/transaction")
async def submit_transaction_feedback(feedback: TransactionFeedback):
    transaction_feedback.append({
        "transaction_id": feedback.transaction_id,
        "is_correct": feedback.is_correct,
        "feedback": feedback.feedback,
        "timestamp": datetime.now().isoformat()
    })
    return {"success": True, "message": "Feedback submitted successfully"}

@app.post("/api/feedback/system")
async def submit_system_feedback(feedback: FeedbackCreate):
    system_feedback.append({
        "category": feedback.category,
        "details": feedback.details,
        "timestamp": datetime.now().isoformat()
    })
    return {"success": True, "message": "System feedback submitted successfully"}

# ------------------------ Run the Server ------------------------

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
