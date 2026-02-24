import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard Sinusoidal Positional Encoding.
    BART can use fixed or learned, we use fixed for zero-parameter efficiency.
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class BARTAttention(nn.Module):
    """
    Standard Multi-Head Attention compatible with BART naming.
    Used for Self-Attention and Cross-Attention.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # BART naming convention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, keys=None, values=None, mask=None):
        batch_size, seq_len, _ = x.shape
        keys = keys if keys is not None else x
        values = values if values is not None else keys
        kv_len = keys.shape[1]

        # Project and split into heads
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(keys).view(batch_size, kv_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(values).view(batch_size, kv_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.out_proj(out)

class BARTEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = BARTAttention(d_model, n_heads, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. Self-Attention (Residual after LN - Post-LN style like BART)
        residual = x
        x = self.self_attn(x, mask=mask)
        x = self.dropout(x)
        x = self.self_attn_layer_norm(residual + x)
        
        # 2. FFN
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.final_layer_norm(residual + x)
        return x

class BARTDecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = BARTAttention(d_model, n_heads, dropout)
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        
        self.encoder_attn = BARTAttention(d_model, n_heads, dropout)
        self.encoder_attn_layer_norm = nn.LayerNorm(d_model)
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, self_mask=None, cross_mask=None):
        # 1. Masked Self-Attention
        residual = x
        x = self.self_attn(x, mask=self_mask)
        x = self.dropout(x)
        x = self.self_attn_layer_norm(residual + x)
        
        # 2. Cross-Attention
        residual = x
        x = self.encoder_attn(x, keys=memory, values=memory, mask=cross_mask)
        x = self.dropout(x)
        x = self.encoder_attn_layer_norm(residual + x)
        
        # 3. FFN
        residual = x
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.final_layer_norm(residual + x)
        return x

class GrammarTransformer(nn.Module):
    """
    v2.0 Encoder-Decoder Transformer (BART-compatible structure).
    Designed for Text-to-Text generation.
    """
    def __init__(self, vocab_size: int, max_seq_len: int, d_model: int, n_heads: int, 
                 n_enc_layers: int, n_dec_layers: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.shared = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len)
        
        # Encoder
        self.encoder_layers = nn.ModuleList([
            BARTEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_enc_layers)
        ])
        
        # Decoder
        self.decoder_layers = nn.ModuleList([
            BARTDecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_dec_layers)
        ])
        
        # Final parameters
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.shared.weight # Weight tying
        
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def encode(self, src_ids, mask=None):
        x = self.pos_encoding(self.shared(src_ids))
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x

    def decode(self, tgt_ids, memory, self_mask=None, cross_mask=None):
        x = self.pos_encoding(self.shared(tgt_ids))
        for layer in self.decoder_layers:
            x = layer(x, memory, self_mask, cross_mask)
        return x

    def forward(self, src_ids, tgt_ids, src_mask=None, tgt_mask=None):
        memory = self.encode(src_ids, src_mask)
        out = self.decode(tgt_ids, memory, tgt_mask, src_mask)
        return self.lm_head(out)

    def generate(self, src_ids, bos_id, eos_id, pad_id=0, max_len=64):
        """
        Autoregressive generation supporting batches and masking.
        """
        device = src_ids.device
        batch_size = src_ids.shape[0]
        
        # 1. Create Source Mask for padding: [batch, 1, 1, src_len]
        src_mask = (src_ids != pad_id).unsqueeze(1).unsqueeze(2)
        
        # 2. Encode once (with mask)
        memory = self.encode(src_ids, src_mask)
        
        # 3. Start decoding
        ys = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
        
        # Keep track of which sequences are finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            tgt_mask = self._create_causal_mask(ys.size(1), device)
            # Use src_mask as cross_mask to ignore PAD tokens in encoder memory
            out = self.decode(ys, memory, tgt_mask, src_mask)
            logits = self.lm_head(out[:, -1, :])
            
            next_word = torch.argmax(logits, dim=-1) # [batch]
            
            # Update ys
            ys = torch.cat([ys, next_word.unsqueeze(1)], dim=1)
            
            # Update status
            unfinished_sequences = unfinished_sequences & (next_word != eos_id).long()
            
            # Stop if all sequences in batch are done
            if unfinished_sequences.max() == 0:
                break
                
        return ys

    def _create_causal_mask(self, seq_len, device):
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)
