from kafka.admin import KafkaAdminClient

# Initialize Kafka Admin Client
admin = KafkaAdminClient(bootstrap_servers='localhost:9092')

# Fetch and print existing topics
print("🎯 Existing topics:", admin.list_topics())
