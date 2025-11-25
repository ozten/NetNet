from datasets import load_dataset
import os
import json

# Define 50GB in bytes
TARGET_SIZE_BYTES = 50 * 1024 * 1024 * 1024 
save_path = "./fineweb_data.jsonl"

print(f"Streaming FineWeb-Edu (Sample-10BT)... target size: 50GB")

# We use the "sample-10BT" subset which is a high-quality 10 Billion token slice
# streaming=True means we don't download the whole thing at once
dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

current_size = 0
mode = 'w'

# Check if file exists to resume (simple check)
if os.path.exists(save_path):
    print("Found existing file. Overwriting (delete manually to resume/append).")

with open(save_path, mode, encoding='utf-8') as f:
    for i, example in enumerate(dataset):
        text = example['text']
        
        # Create a simple JSON line: {"text": "..."}
        line = json.dumps({"text": text}) + "\n"
        f.write(line)
        
        # Track size
        current_size += len(line.encode('utf-8'))
        
        if i % 10000 == 0:
            gb_size = current_size / (1024 * 1024 * 1024)
            print(f"Downloaded: {gb_size:.2f} GB / 50.00 GB", end='\r')
            
        if current_size >= TARGET_SIZE_BYTES:
            print(f"\nReached target size of 50GB. Stopping.")
            break

print(f"Success! Data saved to {save_path}")