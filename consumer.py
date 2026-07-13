from confluent_kafka import Consumer
import json
import requests

conf = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "fraud-scoring-group",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,
}

consumer = Consumer(conf)
consumer.subscribe(["transactions"])

API_URL = "http://127.0.0.1:8000/predict"

print("Listening for transactions... Ctrl+C to stop.")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None:
            continue
        if msg.error():
            print(f"Consumer error: {msg.error()}")
            continue
        
        transaction =json.loads(msg.value().decode("utf-8")) # type: ignore
        response = requests.post(API_URL, json=transaction)
        
        if response.status_code == 200:
            result = response.json()
            print(f"partition {msg.partition()} offset {msg.offset()} "
                  f"-> score={result['anomaly_score']:.4f} tier={result['risk_tier']}")
            consumer.commit(msg) # type: ignore
        else:
            print(f"API error {response.status_code}: {response.text}")
except KeyboardInterrupt:
    print("Stopping consumer.")
finally:
    consumer.close()