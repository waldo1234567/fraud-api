from confluent_kafka import Producer
import json 
import time
import pandas as pd 

conf = {"bootstrap.servers" : "localhost:9092"}
producer = Producer(conf)

def deliver_report(err, msg):
    if err is not None:
        print(f"Delivery failed : {err}")
    else:
        print(f"Delivered to Partition {msg.partition()} at offset {msg.offset()}")

df = pd.read_csv("creditcard.csv")
FEATURES = [c for c in df.columns if c not in ("Time", "Class")]

sample = df.sample(20, random_state=1)

for _,row in sample.iterrows():
    payload = {f : row[f] for f in FEATURES}
    
    producer.produce(
        "transactions",
        value=json.dumps(payload).encode("utf-8"),
        callback=deliver_report
    )
    
    producer.poll(0)
    time.sleep(0.5)
    
producer.flush()
print("Done sending")