import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class TransformerBlock(nn.Module):
    """
    A single transformer block with self-attention and feed-forward layers
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        # Use batch_first=True so tensors are (batch, seq, embed)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        # Self-attention with residual connection and layer normalization
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection and layer normalization
        ff_output = self.ff(x)
        x = self.norm2(x + ff_output)
        return x

class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the token embeddings
    """
    def __init__(self, embed_dim, max_seq_length=1024):
        super(PositionalEncoding, self).__init__()
        
        # Create positional encodings
        pe = torch.zeros(max_seq_length, embed_dim)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (not a parameter, but part of the module)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # Add positional encoding to the input
        return x + self.pe[:, :x.size(1), :]

class LLM(nn.Module):
    """
    A simple language model based on the transformer architecture
    """
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, ff_dim=2048, 
                 num_layers=6, max_seq_length=1024, dropout=0.1):
        super(LLM, self).__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        
        # Stack of transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        
        # Final layer for token prediction
        self.output_layer = nn.Linear(embed_dim, vocab_size)
        # Tie output weights to input embeddings (common trick to improve perplexity)
        self.output_layer.weight = self.token_embedding.weight
        
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, key_padding_mask: Optional[torch.Tensor] = None):
        # x shape: (batch, seq)
        x = self.token_embedding(x)                  # (batch, seq, embed)
        x = self.positional_encoding(x)              # add sinusoidal positions
        x = self.dropout(x)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        
        # Project to vocabulary size
        x = self.output_layer(x)                     # (batch, seq, vocab)
        return x
    
    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # attn_mask shape for batch_first MHA: (seq, seq) or (batch*num_heads, seq, seq)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def _apply_repetition_penalty(self, logits: torch.Tensor, generated_ids: torch.Tensor, penalty: float) -> torch.Tensor:
        if penalty == 1.0 or generated_ids.numel() == 0:
            return logits
        # Penalize tokens that already appeared
        unique_tokens = generated_ids.unique()
        logits[:, unique_tokens] /= penalty
        return logits

    def _top_k_top_p_filtering(self, logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
        # top-k
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k)
            min_keep = values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_keep, torch.full_like(logits, float('-inf')), logits)
        # top-p (nucleus)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_mask = cumulative_probs > top_p
            # Ensure at least one token is kept
            sorted_mask[:, 0] = False
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
            logits = torch.where(mask, torch.full_like(logits, float('-inf')), logits)
        return logits

    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate text using the model with sampling strategies.
        Args:
            input_ids: (batch=1, seq)
            max_length: maximum number of tokens to append
            temperature: softmax temperature (>0)
            top_k: keep only top_k tokens per step (0 disables)
            top_p: nucleus sampling cumulative probability (1.0 disables)
            repetition_penalty: >1.0 discourages repeats (1.0 disables)
            eos_token_id: stop early when generated
        """
        self.eval()
        device = input_ids.device
        with torch.no_grad():
            for _ in range(max_length):
                seq_len = input_ids.size(1)
                attn_mask = self._build_causal_mask(seq_len, device)
                logits = self(input_ids, attn_mask=attn_mask)[:, -1, :]
                logits = logits / max(temperature, 1e-6)
                logits = self._apply_repetition_penalty(logits, input_ids, repetition_penalty)
                logits = self._top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break
        return input_ids