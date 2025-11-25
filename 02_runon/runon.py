import torch
import torch.nn.functional as F
import sys
import os

# --- PATH SETUP ---
# 1. Get the directory where this script (runon.py) is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Construct the path to the '00_foundation' directory
# (It is one level up '..', then into '00_foundation')
foundation_dir = os.path.join(current_dir, '..', '00_foundation')

# 3. Add '00_foundation' to the system path so Python can find 'model' and 'config' imports
sys.path.append(foundation_dir)

# Now we can import from 00_foundation
from model import Llama
from transformers import GPT2TokenizerFast
import config

# --- 1. SETUP ---
device = config.DEVICE
print(f"Running on {device}")

# Load Tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# --- 2. INITIALIZE MODEL ---
model = Llama(
    vocab_size=config.VOCAB_SIZE,
    dim=config.DIM,
    depth=config.DEPTH,
    heads=config.HEADS,
    max_seq_len=config.MAX_SEQ_LEN 
).to(device)

# --- 3. LOAD WEIGHTS ---
print("Loading weights...")
# 4. Update model loading to look in the foundation directory
model_path = os.path.join(foundation_dir, "model.pt")

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Weights loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: '{model_path}' not found.")
    print("Ensure you have trained the model and the file exists in '00_foundation'.")
    sys.exit()

model.eval()

# --- 4. ITERATIVE GENERATION FUNCTION ---
def generate_chunk(current_context, max_new_tokens=20, temperature=0.7):
    """
    Takes the full text context, generates new tokens, and returns 
    the FULL updated text (old + new).
    """
    # 1. Encode the full context
    input_ids = tokenizer.encode(current_context, return_tensors="pt").to(device)
    
    # 2. Generate new tokens
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context to valid window size (RoPE allows 4096, using 1024 for safety)
            idx_cond = input_ids[:, -1024:]
            
            # Forward pass
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] 
            
            # Sampling
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat((input_ids, idx_next), dim=1)

    # 3. Decode everything back to text
    return tokenizer.decode(input_ids[0].cpu().numpy())

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    initial_prompt = input("\nEnter initial prompt to start the chain: ")
    
    # Configuration
    ITERATIONS = 25
    TOKENS_PER_ITER = 10  # How much to write per step
    
    current_text = initial_prompt
    
    print("\n" + "="*50)
    print(f"Starting long-form generation ({ITERATIONS} iterations)...")
    print("="*50 + "\n")
    
    # Print the start
    print(current_text, end="", flush=True)
    
    for i in range(1, ITERATIONS + 1):
        #print("\nprompt: " + current_text)
        # Generate the next version of the text (contains old + new)
        updated_text = generate_chunk(current_text, max_new_tokens=TOKENS_PER_ITER)

        #print("\nupdated_text: " + updated_text)
        
        # Extract just the new part string for printing
        # (We do this by slicing off the length of the previous text)
        new_content = updated_text[len(current_text):]
        #print("\nnew_content: " + new_content)
        
        # Stream output to console
        print(new_content, end="", flush=True)
        
        # Update context for the next loop
        current_text = new_content
        #print("\n\n=====================")

    print("\n\n" + "="*50)
    print("Generation Complete.")
    print("="*50)