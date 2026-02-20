import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention Implementation.
    According to docs/architecture.md: d_model=128, n_heads=4.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads # 32, size of one head
        
        # Parameter weight matrices [128, 128]
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output Projection [128, 128]
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): Mask for attention mechanism.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # x: [batch, 64, 128]
        batch_size, seq_len, _ = x.shape
        
        # 1. Projection to Q, K, V -> [batch, 64, 128]
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        
        # 2. Split into heads: [batch, 64, 128] -> [batch, 64, 4, 32] -> [batch, 4, 64, 32]
        q = q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Attention Scores: (Q @ K^T) / sqrt(32) -> [batch, 4, 64, 64]
        # Q: [batch, 4, 64, 32], K^T: [batch, 4, 32, 64]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Masking (prevent model from looking at future tokens during generation)
        if mask is not None:
            # [batch, n_heads, 64, 64] -> [batch, n_heads, 64, 64]
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # [batch, n_heads, 64, 64]
        weights = F.softmax(scores, dim=-1)
        
        # 4. Mix with Values: Weights @ V -> [batch, 4, 64, 32]
        # Weights: [batch, 4, 64, 64], V: [batch, 4, 64, 32]
        attn_out = weights @ v
        
        # 5. Concatenation: [batch, 4, 64, 32] -> [batch, 64, 128]
        # attn_out.transpose(1, 2): [batch, 4, 64, 32] -> [batch, 64, 4, 32]
        # .contiguous().view(): [batch, 64, 4, 32] -> [batch, 64, 128]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # 6. Output Projection: [batch, 64, 128] @ [128, 128] -> [batch, 64, 128]
        return self.W_o(attn_out)

class TransformerBlock(nn.Module):
    """
    One Transformer Block: Attention + LayerNorm + FFN.
    d_model - model dimensionality, 128
    n_heads - number of heads, 4
    d_ff - FFN inner layer dimensionality, 512
    dropout - dropout rate, 0.1
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # FFN: explicit implementation of W1 [128, 512] and W2 [512, 128] matrices
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Self-Attention Block
        # Residual Connection 1: x = x + Attention(LayerNorm(x))
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        
        # 2. Feed-Forward Network Block
        # Residual Connection 2: x = x + FFN(LayerNorm(x))
        x_norm = self.ln2(x)
        
        # Pass through matrices: W1 -> GELU -> W2
        x_ffn = self.W1(x_norm)
        x_ffn = F.gelu(x_ffn)
        x_ffn = self.W2(x_ffn)
        
        x = x + self.dropout(x_ffn)
        return x

class TransformerModel(nn.Module):
    """
    Main Transformer Decoder model (V=4000, T=64, L=4).
    vocab_size: 4000, vocabulary size
    max_seq_len: 64, maximum sequence length (context window)
    d_model: 128, model dimensionality
    n_heads: 4, number of attention heads
    n_layers: 4, number of transformer blocks (layers)
    d_ff: 512, FFN inner dimension
    """
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, n_heads: int, n_layers: int, d_ff: int):
        super().__init__()
        # 1. Embeddings [4000, 128]
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # [1, 64, 128]
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # 2. Stacking: 4 Transformer blocks
        self.transformerBlocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        
        # 3. Final Norm
        self.ln_final = nn.LayerNorm(d_model)
        
        # 4. LM Head [128, 4000]
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # WEIGHT TYING: Tie input embedding and output projection weights
        self.token_emb.weight = self.lm_head.weight
        
        self.max_seq_len = max_seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize weights according to Transformer best practices. """
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
        # If module has pos_emb (like our TransformerModel), initialize it too
        if hasattr(module, 'pos_emb') and module.pos_emb is not None:
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def _create_causal_mask(self, seq_len, device):
        """
        Creates a triangular mask [1, 1, seq_len, seq_len] to prevent 
        the model from looking at future tokens.
        """
        # 1. Square matrix of ones [seq_len, seq_len]
        mask_ones = torch.ones((seq_len, seq_len), device=device)
        
        # 2. Extract lower triangle. [seq_len, seq_len]
        mask_tril = torch.tril(mask_ones)
        
        # 3. Add dimensions for attention compatibility: [1, 1, seq_len, seq_len]
        return mask_tril.view(1, 1, seq_len, seq_len)

    def forward(self, ids):
        # ids: [batch, seq_len]
        # ids represent sequence tokens for each item in the batch
        batch, seq_len = ids.shape
        
        # 1. Tokens -> Embeddings. [batch, seq_len] -> [batch, seq_len, 128]
        tok_emb = self.token_emb(ids)
        
        # 2. Get positional embeddings. [1, 64, 128] -> [1, seq_len, 128]
        # We only take as many positions as tokens in fixed sequence length.
        pos_emb = self.pos_emb[:, :seq_len, :]
        
        # 3. Combine token meaning and its position. [batch, seq_len, 128]
        x = tok_emb + pos_emb
        
        # 4. Create Causal Mask
        mask = self._create_causal_mask(seq_len, ids.device)
        
        # Pass through transformer layers
        # x: [batch, seq_len, 128] -> [batch, seq_len, 128]
        for block in self.transformerBlocks:
            x = block(x, mask)
        
        # Final Layer Normalization
        # x: [batch, seq_len, 128] -> [batch, seq_len, 128]
        x = self.ln_final(x)
        
        # LM Head -> Token logits for each position
        # x: [batch, seq_len, 128] -> [batch, seq_len, 4000]
        logits = self.lm_head(x)
        return logits
