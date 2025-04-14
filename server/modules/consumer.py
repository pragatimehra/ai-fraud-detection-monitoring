from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, DoubleType, BooleanType

# Create Spark session with Kafka packages
spark = SparkSession.builder \
    .appName("KafkaToParquetStreaming") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Define schemas
customer_schema = StructType() \
    .add("customer_id", StringType()) \
    .add("name", StringType()) \
    .add("email", StringType()) \
    .add("phone", StringType()) \
    .add("address", StringType()) \
    .add("dob", StringType()) \
    .add("created_at", StringType())

account_schema = StructType() \
    .add("account_id", StringType()) \
    .add("customer_id", StringType()) \
    .add("account_type", StringType()) \
    .add("account_number", StringType()) \
    .add("created_at", StringType()) \
    .add("balance", DoubleType())

transaction_schema = StructType() \
    .add("transaction_id", StringType()) \
    .add("account_id", StringType()) \
    .add("account_type", StringType()) \
    .add("timestamp", StringType()) \
    .add("amount", DoubleType()) \
    .add("transaction_type", StringType()) \
    .add("merchant", StringType()) \
    .add("location", StringType()) \
    .add("is_foreign", BooleanType()) \
    .add("is_high_risk_country", BooleanType()) \
    .add("opening_balance", DoubleType()) \
    .add("closing_balance", DoubleType()) \
    .add("is_fraud", BooleanType()) \
    .add("fraud_reasons", StringType())

def consume_topic(topic_name, schema, output_path):
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "localhost:9092") \
        .option("subscribe", topic_name) \
        .option("startingOffsets", "earliest") \
        .load()

    value_df = df.selectExpr("CAST(value AS STRING)") \
        .select(from_json(col("value"), schema).alias("data")) \
        .select("data.*")

    query = value_df.writeStream \
        .format("parquet") \
        .option("path", output_path) \
        .option("checkpointLocation", output_path + "_chk") \
        .outputMode("append") \
        .start()

    return query

# Start all streams
customers_query = consume_topic("customers_topic", customer_schema, "parquet/customers")
accounts_query = consume_topic("accounts_topic", account_schema, "parquet/accounts")
transactions_query = consume_topic("transactions_topic", transaction_schema, "parquet/transactions")

# Await termination
customers_query.awaitTermination()
accounts_query.awaitTermination()
transactions_query.awaitTermination()
