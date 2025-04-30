from faker import Faker
import pandas as pd
import numpy as np
import uuid
import random
import os
import csv
import json
from kafka import KafkaProducer

# Initialize Faker and random seeds
fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

# Configuration
NUM_CUSTOMERS = 10
ACCOUNTS_PER_CUSTOMER = 10
NUM_ACCOUNTS = NUM_CUSTOMERS * ACCOUNTS_PER_CUSTOMER
NUM_TRANSACTIONS = 1000
anomaly_probability = 0.05
NUM_ANOMALIES = int(NUM_TRANSACTIONS * anomaly_probability)

blacklisted_merchants = ['Vision Corp', 'Pinnacle Ltd', 'Omega LLC']
blacklisted_locations = ['Lakeview', 'Springfield', 'Newport']
high_risk_countries = ['North Korea', 'Syria', 'Iran']

# Kafka Producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Send existing customer/account data to Kafka
def send_customers_accounts_to_kafka(df_customers, df_accounts):
    for _, row in df_customers.iterrows():
        producer.send('customers_topic', row.to_dict())
    for _, row in df_accounts.iterrows():
        producer.send('accounts_topic', row.to_dict())

# Generate customer and account data
def generate_customers_accounts():
    customers = []
    for _ in range(NUM_CUSTOMERS):
        customer = {
            "customer_id": str(uuid.uuid4()),
            "name": fake.name(),
            "email": fake.email(),
            "phone": fake.phone_number(),
            "address": fake.address().replace('\n', ', '),
            "dob": fake.date_of_birth(minimum_age=18, maximum_age=85).isoformat(),
            "created_at": fake.date_time_this_decade().isoformat()
        }
        customers.append(customer)

    df_customers = pd.DataFrame(customers)

    accounts = []
    for cust in customers:
        for _ in range(ACCOUNTS_PER_CUSTOMER):
            account = {
                "account_id": str(uuid.uuid4()),
                "customer_id": cust["customer_id"],
                "account_type": random.choice(["savings", "checking", "credit"]),
                "account_number": fake.unique.bban(),
                "created_at": fake.date_time_this_decade().isoformat(),
                "balance": round(random.uniform(1000, 100000), 2)
            }
            accounts.append(account)

    df_accounts = pd.DataFrame(accounts)
    df_accounts.set_index("account_id", inplace=True)

    df_customers.to_csv("customers.csv", index=False)
    df_accounts.reset_index(inplace=True)
    df_accounts.to_csv("accounts.csv", index=False)

    # Send new data to Kafka
    send_customers_accounts_to_kafka(df_customers, df_accounts)

    return df_customers, df_accounts.set_index("account_id")

# Load or generate customers/accounts
if os.path.exists("customers.csv") and os.path.exists("accounts.csv"):
    df_customers = pd.read_csv("customers.csv")
    df_accounts = pd.read_csv("accounts.csv")
    df_accounts.set_index("account_id", inplace=True)

    # Always send to Kafka on startup
    send_customers_accounts_to_kafka(df_customers, df_accounts.reset_index())
else:
    df_customers, df_accounts = generate_customers_accounts()

# Transaction generation
transactions = []
account_balances = {}
anomaly_types = [
    "simulated_high_amount",
    "simulated_high_risk_country",
    "simulated_account_draining",
    "simulated_unusual_type",
    "simulated_blacklisted_merchant"
]
anomaly_indices = set(random.sample(range(NUM_TRANSACTIONS), NUM_ANOMALIES))

for i in range(NUM_TRANSACTIONS):
    acct = df_accounts.sample(1)
    acct_id = acct.index[0]
    account_type = acct["account_type"].values[0]

    t_type = random.choice(["purchase", "transfer", "withdrawal", "deposit", "payment"])
    amount = round(random.uniform(5, 2000), 2)
    is_foreign = random.choices([True, False], weights=[0.1, 0.9])[0]
    is_high_risk_country = random.choices([True, False], weights=[0.05, 0.95])[0] if is_foreign else False
    merchant = fake.company()
    location = fake.city()

    if acct_id not in account_balances:
        account_balances[acct_id] = acct["balance"].values[0]

    fraud_reasons = []
    is_fraud = False

    if i in anomaly_indices:
        is_fraud = True
        anomaly_choice = random.choice(anomaly_types)

        if anomaly_choice == "simulated_high_amount":
            amount = round(np.random.uniform(50000, 100000), 2)
            fraud_reasons.append("simulated_high_amount")
        elif anomaly_choice == "simulated_high_risk_country":
            is_foreign = True
            is_high_risk_country = True
            location = random.choice(high_risk_countries)
            fraud_reasons.append("simulated_high_risk_country")
        elif anomaly_choice == "simulated_account_draining":
            t_type = "withdrawal"
            current_balance = account_balances[acct_id]
            amount = round(np.random.uniform(0.9 * current_balance, current_balance), 2)
            fraud_reasons.append("simulated_account_draining")
        elif anomaly_choice == "simulated_unusual_type":
            if account_type == "credit":
                t_type = "ATM Withdrawal"
            else:
                t_type = "Unusual Transaction"
            fraud_reasons.append("simulated_unusual_type")
        elif anomaly_choice == "simulated_blacklisted_merchant":
            merchant = random.choice(blacklisted_merchants)
            fraud_reasons.append("simulated_blacklisted_merchant")

    if t_type in ["deposit", "payment"]:
        opening = account_balances[acct_id]
        closing = opening + amount
    else:
        opening = account_balances[acct_id]
        closing = opening - amount

    transaction = {
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
    }

    transactions.append(transaction)
    account_balances[acct_id] = closing
    producer.send('transactions_topic', transaction)

# Save transactions to CSV
df_transactions = pd.DataFrame(transactions)
df_accounts.reset_index(inplace=True)

transactions_csv = "transactions.csv"
if os.path.exists(transactions_csv):
    df_transactions.to_csv(transactions_csv, mode='a', index=False, header=False)
else:
    df_transactions.to_csv(transactions_csv, index=False)

producer.flush()

# Print summary
print(f"✅ {len(df_transactions)} transactions generated.")
print(f"🚨 {df_transactions['is_fraud'].sum()} anomalous transactions injected.")
