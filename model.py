import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 300):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


class TransformerDecoderBlock(nn.Module):
    """Single Transformer decoder block with masked self-attention"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class SerineProteaseTransformer(nn.Module):
    """Autoregressive Transformer for serine protease sequence generation"""
    
    def __init__(
        self,
        vocab_size: int = 21,
        embedding_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dim_feedforward: int = 128,
        max_len: int = 300,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len)
        
        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(embedding_dim, n_heads, dim_feedforward, dropout)
            for _ in range(n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def generate_mask(self, seq_len: int, device: torch.device):
        """Generate causal mask for autoregressive generation"""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x):
        # x: [batch_size, seq_len]
        batch_size, seq_len = x.shape
        
        # Embedding
        x = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # Positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Generate causal mask
        mask = self.generate_mask(seq_len, x.device)
        
        # Pass through decoder blocks
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, mask)
        
        # Project to vocabulary
        output = self.output_proj(x)  # [batch_size, seq_len, vocab_size]
        
        return output
    
    def generate(self, start_token: int = 1, max_length: int = 300, temperature: float = 1.0, device: torch.device = None):
        """Generate a sequence autoregressively"""
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        generated = [start_token]
        
        with torch.no_grad():
            for _ in range(max_length - 1):
                # Create input tensor
                input_seq = torch.tensor([generated], dtype=torch.long, device=device)
                
                # Forward pass
                output = self.forward(input_seq)
                
                # Get logits for last position
                logits = output[0, -1, :] / temperature
                
                # Mask padding token (set to very negative value)
                logits[0] = float('-inf')
                
                # Apply softmax and sample
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Stop if padding token (shouldn't happen with masking, but safety check)
                if next_token == 0:
                    break
                
                generated.append(next_token)
        
        return generated

