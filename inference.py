import torch
import torch.nn.functional as F
from model import Llama
from transformers import GPT2TokenizerFast

# --- 1. SETUP ---
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Running on {device}")

# Load Tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# --- 2. INITIALIZE MODEL ---
# MUST MATCH YOUR TRAIN.PY CONFIG EXACTLY
model = Llama(
    vocab_size=50257,
    dim=768,        # Match train.py
    depth=12,       # Match train.py
    heads=12,       # Match train.py
    max_seq_len=1024 
).to(device)

# --- 3. LOAD WEIGHTS ---
print("Loading weights...")
try:
    # map_location ensures it loads to the correct device
    model.load_state_dict(torch.load("model.pt", map_location=device))
    print("Weights loaded successfully!")
except FileNotFoundError:
    print("Error: 'model.pt' not found. Did you finish training?")
    exit()

# Set to Evaluation Mode (Turns off Dropout)
model.eval()

# --- 4. GENERATION FUNCTION ---
def generate(prompt, max_new_tokens=100, temperature=0.8):
    # 1. Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # 2. Generation Loop
    # We use torch.no_grad() because we don't need gradients for inference (saves RAM)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context if it gets too long
            idx_cond = input_ids[:, -1024:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] 
            
            # Apply Temperature (Higher = crazier, Lower = more predictable)
            logits = logits / temperature
            
            # Apply Softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1)

    # 3. Decode back to text
    output_text = tokenizer.decode(input_ids[0].cpu().numpy())
    return output_text

# --- 5. RUN IT ---
while True:
    user_prompt = input("\nType a prompt (or 'q' to quit): ")
    if user_prompt.lower() == 'q':
        break
        
    print("\nGenerating...", end="", flush=True)
    result = generate(user_prompt, max_new_tokens=50, temperature=0.8)
    
    print("\n" + "-"*40)
    print(result)
    print("-"*40)