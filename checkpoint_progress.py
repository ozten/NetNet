import torch
import torch.nn.functional as F
import os
import glob
import re
from model import Llama
from transformers import GPT2TokenizerFast

# --- 1. CONFIGURATION ---
# The prompt you want to test across all ages
TEST_PROMPT = "The scientist opened the door and found"

# Must match your train.py architecture exactly!
DIM = 768
DEPTH = 12
HEADS = 12
MAX_SEQ_LEN = 1024  # Or 512, whichever you used in training

# --- 2. SETUP ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on {device}")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Initialize model once (we will just swap weights inside)
model = Llama(
    vocab_size=50257,
    dim=DIM,
    depth=DEPTH,
    heads=HEADS,
    max_seq_len=MAX_SEQ_LEN
).to(device)

model.eval()

# --- 3. HELPER FUNCTIONS ---

def get_step_number(filename):
    """Extracts the number from 'checkpoint_step_500.pt'"""
    match = re.search(r'step_(\d+)', filename)
    return int(match.group(1)) if match else 0

def generate(prompt, max_new_tokens=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context
            idx_cond = input_ids[:, -MAX_SEQ_LEN:]
            
            # Forward
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] 
            
            # Greedy decoding for evolution (temperature=0) often shows 
            # the 'purest' state of the model, but let's use low temp 
            # to keep it stable yet creative.
            logits = logits / 0.6 
            probs = F.softmax(logits, dim=-1)
            # Normal inferencc            
            idx_next = torch.multinomial(probs, num_samples=1)
            # Greedy decoding (known best answer)
            #idx_next = torch.argmax(logits, dim=-1, keepdim=True)            
            input_ids = torch.cat((input_ids, idx_next), dim=1)

    return tokenizer.decode(input_ids[0].cpu().numpy())

# --- 4. THE EVOLUTION LOOP ---

# Find all checkpoint files
files = glob.glob("checkpoint_step_*.pt")

# Sort them numerically (otherwise 1000 comes before 500 in text sorting)
files.sort(key=get_step_number)

if not files:
    print("No checkpoints found! Make sure you have files named 'checkpoint_step_X.pt'")
    exit()

print(f"\nFound {len(files)} checkpoints. Beginning evolution...\n")
print(f"PROMPT: '{TEST_PROMPT}'\n")
print("="*60)

for filepath in files:
    step_count = get_step_number(filepath)
    
    try:
        # 1. Load the file to CPU first to inspect it safely
        checkpoint = torch.load(filepath, map_location=device)
        
        # 2. Check if it's a "Full Checkpoint" (dict with keys like 'model_state_dict')
        if "model_state_dict" in checkpoint:
            # Extract just the weights
            state_dict = checkpoint["model_state_dict"]
        else:
            # It's likely just the raw weights
            state_dict = checkpoint

        # 3. Load into model
        # strict=False allows us to ignore non-weight keys if any sneak in, 
        # but usually cleaning it above is better.
        model.load_state_dict(state_dict)
        
    except Exception as e:
        print(f"Skipping {filepath}: {e}")
        continue

    # Generate
    output = generate(TEST_PROMPT)
    
    # Print Result
    print(f"--- STEP {step_count} ---")
    # We remove the prompt from the output just for cleaner visualization if you want,
    # or keep it. Here I keep it.
    print(output)
    print("="*60)