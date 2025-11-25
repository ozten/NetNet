import torch

# --- DEVICE SETUP ---
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()

# --- MODEL ARCHITECTURE ---
# 125M Config: dim=768, depth=12, heads=12
VOCAB_SIZE = 50257
DIM = 768
DEPTH = 12
HEADS = 12
MAX_SEQ_LEN = 512  # Matches training block size
ROPE_SEQ_LEN = 4096  # Matches the checkpoint's RoPE size

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 16
LEARNING_RATE = 6e-4
MAX_ITERS = 35000
