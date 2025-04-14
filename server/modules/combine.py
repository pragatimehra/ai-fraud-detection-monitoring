from pyspark.sql import SparkSession
import os
import shutil

def merge_parquet_to_csv() -> str:
    spark = SparkSession.builder.appName("CombineParquetData").getOrCreate()
    print("🔄 Starting data denormalization...")

    # Load Parquet files
    df_customers = spark.read.parquet("modules/parquet/customers") \
        .withColumnRenamed("created_at", "created_at_cust")
    df_accounts = spark.read.parquet("modules/parquet/accounts") \
        .withColumnRenamed("account_type", "account_type_acct") \
        .withColumnRenamed("created_at", "created_at_acct")
    df_transactions = spark.read.parquet("modules/parquet/transactions") \
        .withColumnRenamed("account_type", "account_type_txn")

    # Join datasets
    df_joined = df_transactions.join(df_accounts, on="account_id", how="inner") \
                               .join(df_customers, on="customer_id", how="inner")
    print(df_joined.count())
    final_columns = [
        "transaction_id", "timestamp", "amount", "transaction_type",
        "merchant", "location", "is_foreign", "is_high_risk_country",
        "opening_balance", "closing_balance",
        "account_id", "account_type_txn", "account_type_acct", "account_number", "balance", "created_at_acct",
        "customer_id", "name", "email", "phone", "address", "dob", "created_at_cust",
        "is_fraud", "fraud_reasons"
    ]

    df_new = df_joined.select(*final_columns).dropDuplicates(["transaction_id"])

    output_dir = "modules/denormalized_transactions"

    # 💣 Clear old data if exists
    if os.path.exists(output_dir):
        print(f"🧹 Clearing old output directory: {output_dir}")
        shutil.rmtree(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # ✅ Save fresh deduplicated data
    df_new.coalesce(1).write.option("header", "true").mode("overwrite").csv(output_dir)
    print(f"✅ New deduplicated dataset saved to '{output_dir}'.")
    
    spark.stop()
    return output_dir

if __name__ == "__main__":
    merge_parquet_to_csv()