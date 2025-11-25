import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # The weight parameter is learnable
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 1. Calculate the root mean square (RMS)
        # x.pow(2) -> square every element
        # mean(-1) -> average across the hidden dimension
        # rsqrt -> reciprocal square root (1/sqrt)
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        
        # 2. Scale by the learnable weight
        return norm * self.weight

class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        # In Llama, hidden_dim is usually 4 * dim * (2/3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False) # Gate projection
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) # Down projection
        self.w3 = nn.Linear(dim, hidden_dim, bias=False) # Up projection

    def forward(self, x):
        # The magic formula: (Swish(Gate) * Up) -> Down
        # F.silu is the Swish activation
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = config.ROPE_SEQ_LEN, theta: float = 10000.0):
        super().__init__()
        # 1. Calculate frequencies (theta_i)
        # dim must be even
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        
        # 2. Create position indices (0, 1, ..., max_seq_len)
        t = torch.arange(max_seq_len, device=freqs.device, dtype=torch.float32)
        
        # 3. Calculate the outer product to get angles
        freqs = torch.outer(t, freqs).float()  # (seq_len, dim/2)
        
        # 4. Turn into polar coordinates (complex numbers for rotation)
        # We cache this buffer so it's not recomputed every step
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
        self.register_buffer("freqs_cis", freqs_cis)

    def forward(self, x):
        # x shape: (batch, seq_len, n_heads, head_dim)
        # Reshape input into complex pairs: (..., head_dim/2)
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
        # Slice the cached frequencies to match current sequence length
        freqs_cis = self.freqs_cis[:x.shape[1]]
        
        # Reshape freqs for broadcasting: (1, seq_len, 1, head_dim/2)
        freqs_cis = freqs_cis.view(1, x.shape[1], 1, x_complex.shape[-1])
        
        # Apply rotation (multiplication in complex space)
        x_rotated = x_complex * freqs_cis
        
        # Turn back to real numbers and flatten
        return torch.view_as_real(x_rotated).flatten(3)

class LlamaBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        
        self.attention = nn.Linear(dim, dim * 3, bias=False) # Q, K, V combined
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.feed_forward = SwiGLU(dim, hidden_dim)
        
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.rope = RoPE(self.head_dim)

    def forward(self, x):
        B, T, C = x.shape
        
        # 1. Pre-norm
        h = self.norm1(x)
        
        # 2. Self Attention with RoPE
        qkv = self.attention(h)
        q, k, v = qkv.split(C, dim=-1)
        
        # Reshape for heads
        q = q.view(B, T, self.heads, self.head_dim)
        k = k.view(B, T, self.heads, self.head_dim)
        v = v.view(B, T, self.heads, self.head_dim)
        
        # Apply RoPE to Q and K
        q = self.rope(q)
        k = self.rope(k)
        
        # Standard Attention (Flash Attention is auto-enabled in PyTorch 2.0+ if supported)
        # use transpose to get (B, heads, T, head_dim)
        attn_out = F.scaled_dot_product_attention(
            q.transpose(1, 2), 
            k.transpose(1, 2), 
            v.transpose(1, 2), 
            is_causal=True
        )
        
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.out_proj(attn_out) # Residual connection
        
        # 3. Feed Forward
        x = x + self.feed_forward(self.norm2(x))
        return x
# -------------------------------

# --- THE NEW MAIN CLASS ---
class Llama(nn.Module):
    def __init__(self, vocab_size, dim, depth, heads, max_seq_len):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.max_seq_len = max_seq_len
        
        # 1. Token Embeddings (Input)
        self.token_embeddings = nn.Embedding(vocab_size, dim)
        
        # 2. The Stack of Llama Blocks
        self.layers = nn.ModuleList([
            LlamaBlock(dim, heads, hidden_dim=4*dim) 
            for _ in range(depth)
        ])
        
        # 3. Final Norm
        self.norm = RMSNorm(dim)
        
        # 4. Output Head (Project back to vocabulary size)
        self.output = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying (Optional but standard): 
        # Makes the input and output embeddings share weights to save parameters
        self.token_embeddings.weight = self.output.weight

    def forward(self, x, targets=None):
        # x shape: (Batch, Seq_Len)
        
        x = self.token_embeddings(x)
        
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
            
        x = self.norm(x)
        
        # Calculate logits
        logits = self.output(x) # (Batch, Seq_Len, Vocab_Size)
        
        loss = None
        if targets is not None:
            # Flatten the tokens to calculate CrossEntropy
            B, T, C = logits.shape
            # New: Use reshape() which handles non-contiguous memory safely
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss