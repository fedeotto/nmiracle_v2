import torch.nn as nn
from nmiracle_v2.models.components.spectrum_encoder import PositionalEncoding
import torch
import math
import torch.nn.functional as F

class TransformerModel(nn.Module):
    """
    NMR2Struct-style Transformer for structure generation
    """
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation='relu',
        layer_norm_eps=1e-05,
        batch_first=True,
        norm_first=False,
        vocab_size=100,  # SMILES vocabulary size
        pad_token=0,
    ):
        super().__init__()
        
        # Core transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )
        
        # Target embedding and output projection
        self.tgt_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Token indices
        self.pad_token = pad_token
        self.d_model = d_model
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
    def _generate_square_subsequent_mask(self, size, device):
        """Generate a mask for decoder self-attention"""
        mask = torch.triu(torch.ones(size, size, device=device) * float('-inf'), diagonal=1)
        return mask

    def initialize_weights(self) -> None:
        """Initializes network weights
        Non-1D parameters are initialized using Xavier initialization
        """
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt=None, src_key_padding_mask=None):
        """
        Forward pass through the transformer
        
        Args:
            src: Encoder input [batch_size, src_len, d_model]
            tgt: Decoder input [batch_size, tgt_len] (optional)
            src_key_padding_mask: Mask for padding in source [batch_size, src_len]
        """
        batch_size = src.size(0)
        
        # Apply positional encoding to source
        # src = self.pos_encoder(src)
        
        # For inference with no target provided
        if tgt is None:
            tgt = torch.zeros(batch_size, 1, dtype=torch.long, device=src.device) + self.pad_token + 1  # Start token
        
        # Create target mask for causal attention
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)
        
        # Get target padding mask
        tgt_key_padding_mask = (tgt == self.pad_token)
        
        # Embed target
        tgt_embedded = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.pos_encoder(tgt_embedded, None)

        # Memory mask should typically match source padding mask
        memory_key_padding_mask = src_key_padding_mask  # Add this line

        # Automatically detect device type and disable autocast
        device_type = 'cuda' if tgt_embedded.is_cuda else 'cpu'
        # Force FP32 computation by disabling autocast
        with torch.autocast(device_type=device_type, enabled=False):
            src = src.to(torch.float32)  # Ensure float32 for transformer encoder
            tgt_embedded = tgt_embedded.to(torch.float32)  # Ensure float32 for transformer decoder
            
            output = self.transformer(
                src,
                tgt_embedded,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits