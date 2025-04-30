from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import UnknownTopicOrPartitionError

# Connect to Kafka
admin = KafkaAdminClient(bootstrap_servers='localhost:9092')

# List existing topics
existing_topics = admin.list_topics()
print("🧾 Existing Topics:", existing_topics)

# Delete all topics
try:
    admin.delete_topics(topics=existing_topics)
    print("✅ All topics deleted successfully.")
except UnknownTopicOrPartitionError as e:
    print("⚠️ Some topics might not exist:", e)
finally:
    admin.close()