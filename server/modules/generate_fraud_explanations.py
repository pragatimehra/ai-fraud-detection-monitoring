import os
import json
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

def generate_fraud_explanations():
    print("🤖 Generating fraud explanations using Gemini...")

    # Configure Gemini API
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)

    # Define paths
    base_dir = Path(__file__).resolve().parent.parent
    fraud_csv_path = base_dir / "modules/fraud_cases_for_llm.csv"
    history_csv_path = base_dir / "modules/denormalized_transactions" / "account_history.csv"
    output_json_path = base_dir / "modules/fraud_explanations.json"

    # Load data
    fraud_df = pd.read_csv(fraud_csv_path)
    history_df = pd.read_csv(history_csv_path)

    # Merge data on 'account_id'
    if "account_id" not in fraud_df.columns or "account_id" not in history_df.columns:
        raise KeyError("Missing 'account_id' column in one of the input files.")
    merged_df = pd.merge(fraud_df, history_df, on="account_id", how="left", suffixes=('', '_history'))

    # Normalize anomaly scores
    min_score = merged_df["anomaly_score"].min()
    max_score = merged_df["anomaly_score"].max()
    merged_df["score"] = (merged_df["anomaly_score"] - min_score) / (max_score - min_score)

    # Initialize Gemini model
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Generate explanations
    explanations = []
    for _, row in tqdm(merged_df.iterrows(), total=merged_df.shape[0]):
        transaction_info = row.drop(labels=[col for col in row.index if col.endswith('_history') or col == "score"]).to_dict()
        history_info = {k: v for k, v in row.items() if k.endswith('_history')}

        prompt = f"""
You are an AI fraud analyst. A fraud detection model flagged the following transaction as fraudulent:

--- Transaction Info ---
{json.dumps(transaction_info, indent=2)}

--- Account History Info ---
{json.dumps(history_info, indent=2)}

Provide a concise explanation (1-2 sentences) of why the model might have flagged this transaction.
"""

        try:
            response = model.generate_content(prompt)
            explanation = response.text.strip()
        except Exception as e:
            explanation = f"Error generating explanation: {e}"

        explanations.append({
            "id": transaction_info.get("transaction_id", ""),
            "timestamp": transaction_info.get("timestamp", ""),
            "amount": transaction_info.get("amount", 0),
            "accountNumber": transaction_info.get("account_number", ""),
            "transactionType": transaction_info.get("transaction_type", ""),
            "score": float(row["score"]),
            "reason": explanation
        })

    # Save explanations to JSON
    with open(output_json_path, "w") as f:
        json.dump(explanations, f, indent=2)

    print(f"✅ Saved {len(explanations)} fraud explanations to '{output_json_path}'")

if __name__ == "__main__":
    generate_fraud_explanations()