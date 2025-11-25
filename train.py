import os
import glob
import re
from datetime import datetime

import torch
from model import Llama  # <--- This imports your class from model.py
# from datasets import load_from_disk
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm

# --- 1. SETUP DEVICE ---
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = get_device()
print(f"Training on {device}")

# --- 2. HYPERPARAMETERS (125M Config) ---
# 125M is roughly: dim=768, depth=12, heads=12
BATCH_SIZE = 16         # Tune for your GPU Memory
BLOCK_SIZE = 512       # Sequence length (increase to 1024 later)
LEARNING_RATE = 6e-4
MAX_ITERS = 5000        # 5000


# --- 3. LOAD DATA & TOKENIZER ---
print("Loading tokenizer...")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token # Fix for GPT2 tokenizer

print("Loading data...")
# 1. Load the JSONL file
# split="train" gives you the dataset directly, not a dictionary
print("Preparing dataset (this may take time once)...")
dataset = load_dataset("json", data_files="./fineweb_data.jsonl", split="train")
train_data = dataset

# Simple data loader function
def get_batch():
    import random
    
    # 1. Pick random indices
    # We use randint because 'train_data' is now a Dataset object, not a simple list
    random_indices = [random.randint(0, len(train_data) - 1) for _ in range(BATCH_SIZE)]
    
    # 2. Get the actual strings
    # We must access ['text'] to get the string out of the dictionary row
    batch_texts = [train_data[i]['text'] for i in random_indices]
    
    # Debug: Uncomment this if it fails again to see exactly what you are passing
    # print(f"Type check: {type(batch_texts[0])}") 
    
    # 3. Tokenize
    encodings = tokenizer(batch_texts, truncation=True, padding="max_length", 
                          max_length=BLOCK_SIZE+1, return_tensors="pt")
    
    input_ids = encodings['input_ids'].to(device)
    
    x = input_ids[:, :-1]
    y = input_ids[:, 1:]
    
    return x, y

# --- 4. INITIALIZE MODEL ---
print("Initializing Llama model...")
model = Llama(
    vocab_size=50257,  # GPT-2 vocab size
    dim=768,
    depth=12,
    heads=12,
    max_seq_len=BLOCK_SIZE
).to(device)

print(f"Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
# --- 5. RECOVERY LOGIC (### NEW ###) ---
start_iter = 0

# Find all files matching the pattern
checkpoint_files = glob.glob("checkpoint_step_*.pt")
if checkpoint_files:
    # specific regex to find the number in "checkpoint_step_500.pt"
    def extract_step(filename):
        match = re.search(r"checkpoint_step_(\d+).pt", filename)
        return int(match.group(1)) if match else 0

    # Sort files by the step number (highest last)
    latest_checkpoint = max(checkpoint_files, key=extract_step)
    start_iter = extract_step(latest_checkpoint) + 1

    print(f"Found checkpoint: {latest_checkpoint}. Resuming from step {start_iter}...")
    
    # Load the checkpoint
    checkpoint_data = torch.load(latest_checkpoint, map_location=device)
    
    # Check if the file is just weights (old format) or a full dict (new format)
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        # This handles the 'Better' saving format below
        model.load_state_dict(checkpoint_data["model_state_dict"])
        optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        print("Loaded model and optimizer state.")
    else:
        # This handles your CURRENT saving format (just model weights)
        model.load_state_dict(checkpoint_data)
        print("Loaded model weights only (Optimizer reset).")

else:
    print("No checkpoints found. Starting from scratch.")

pbar = tqdm(range(start_iter, MAX_ITERS), desc="Training", initial=start_iter, total=MAX_ITERS)

# --- 5. TRAINING LOOP ---
model.train()
print("Starting training...")

for i in pbar:
    # A. Get Data
    xb, yb = get_batch()
    
    # B. Forward Pass
    logits, loss = model(xb, yb)
    
    # C. Backward Pass
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        # print(f"Step {i}: Loss {loss.item():.4f}")
        pbar.set_description(f"Loss {loss.item():.4f}")

    # Historical Logging
    if i % 100 == 0:
        timestamp = datetime.now().strftime("%H:%M:%S")
        tqdm.write(f"[{timestamp}] Step {i}: Loss {loss.item():.4f}")
    
    if i > 0 and i % 500 == 0:
        print(f"Saving checkpoint at step {i}...")

        # ### UPDATED SAVE LOGIC ###
        # We now save the optimizer too, so future crashes don't lose momentum info
        torch.save({
            'step': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f"checkpoint_step_{i}.pt")

print("Training complete!")
# Add this to the end of train.py
print("Saving model...")
torch.save(model.state_dict(), "model.pt")
print("Model saved to model.pt")