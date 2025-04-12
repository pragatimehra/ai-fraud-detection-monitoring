import os
import polars as pl

def enrich_with_historical_features(csv_dir: str):
    # Locate the actual part file inside Spark output directory
    part_file = next((f for f in os.listdir(csv_dir) if f.startswith("part-") and f.endswith(".csv")), None)

    if not part_file:
        raise FileNotFoundError("❌ No part file found in CSV directory.")

    full_csv_path = os.path.join(csv_dir, part_file)
    print(f"🔍 Loading CSV file: {full_csv_path}")
    df = pl.read_csv(full_csv_path)

    # Sort by account_id and timestamp to ensure time order
    df = df.sort(["account_id", "timestamp"])

    # Part A: Account-level Aggregation
    account_agg = (
        df.group_by("account_id")
        .agg([
            pl.col("amount").count().alias("agg_txn_count"),
            pl.col("amount").mean().alias("agg_avg_amount"),
            pl.col("amount").std().alias("agg_std_amount"),
            pl.col("amount").max().alias("agg_max_amount"),
            pl.col("merchant").n_unique().alias("agg_unique_merchants"),
            pl.col("location").n_unique().alias("agg_unique_locations")
        ])
    )

    df = df.join(account_agg, on="account_id", how="left")

    # Part B: Rolling Historical Features
    df = df.with_columns([
        (pl.col("amount").cum_sum().over("account_id") - pl.col("amount")).alias("cumulative_amount"),
        (pl.col("amount").count().over("account_id") - 1).alias("past_txn_count")
    ])

    df = df.with_columns(
        (
            pl.col("cumulative_amount") /
            pl.when(pl.col("past_txn_count") > 0)
              .then(pl.col("past_txn_count"))
              .otherwise(1)
        ).alias("past_avg_amount")
    )

    # Custom Rolling Mode per Account for Merchant and Location
    def get_past_mode(values: list) -> list:
        history = []
        for i in range(len(values)):
            if i == 0:
                history.append("None")
            else:
                freq = {}
                for v in values[:i]:
                    freq[v] = freq.get(v, 0) + 1
                mode_val = max(freq, key=freq.get)
                history.append(mode_val)
        return history

    groups = []
    for group in df.partition_by("account_id", as_dict=False):
        merchants = group["merchant"].to_list()
        locations = group["location"].to_list()
        group = group.with_columns([
            pl.Series("past_common_merchant", get_past_mode(merchants)),
            pl.Series("past_common_location", get_past_mode(locations))
        ])
        groups.append(group)

    df = pl.concat(groups)
    df = df.drop("cumulative_amount")
    df = df.with_columns([
        pl.col("past_avg_amount").fill_null(0),
        pl.col("past_txn_count").fill_null(0),
        pl.col("past_common_merchant").fill_null("None"),
        pl.col("past_common_location").fill_null("None")
    ])

    enriched_output = os.path.join(csv_dir, "denoised_enriched_transactions.csv")
    df.write_csv(enriched_output)
    print(f"✅ Historical features added and saved to '{enriched_output}'.")

if __name__ == "__main__":
    enrich_with_historical_features("modules/denormalized_transactions")