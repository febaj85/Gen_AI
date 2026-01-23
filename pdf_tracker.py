import json
import os

TRACKER_FILE = "processed_pdfs.json"

def load_processed():
    if not os.path.exists(TRACKER_FILE):
        return {}
    with open(TRACKER_FILE, "r") as f:
        return json.load(f)

def save_processed(data):
    with open(TRACKER_FILE, "w") as f:
        json.dump(data, f, indent=2)
