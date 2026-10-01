import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import torch
import torch.nn as nn
from torch import Tensor

class SpectrumEncoder(nn.Module):
    """
    Encodes spectral data (IR or NMR) with peak-focused processing
    """
    def __init__(
        self, 
        d_model=768, 
        nhead=8, 
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        spectrum_type="ir",
        pool_size_1 = 12,
        out_channels_1=64,
        kernel_size_1=5,
        pool_size_2=20,
        out_channels_2=128,
        kernel_size_2=9,
        cnmr_binary=True,
        cnmr_binary_bins=80,
        add_pos_encode=True
    ):
        super().__init__()
        self.spectrum_type    = spectrum_type
        self.d_model          = d_model
        self.cnmr_binary      = cnmr_binary
        self.cnmr_binary_bins = cnmr_binary_bins
        self.add_pos_encode   = add_pos_encode

        if spectrum_type == "ir" or spectrum_type == 'hnmr':
            # Integrated feature extraction for IR
            layers = []
            
            # First convolutional layer to extract local patterns
            layers.append(nn.Conv1d(1, out_channels_1, kernel_size_1, stride=1, padding='valid'))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool_size_1))
            
            # Second convolutional layer for higher-level features
            layers.append(nn.Conv1d(out_channels_1, out_channels_2, kernel_size_2, stride=1, padding='valid'))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool_size_2))
            
            self.feature_extractor = nn.Sequential(*layers)
            self.post_conv_transform = nn.Linear(out_channels_2, self.d_model)

            # Positional encoding (added "locally" for each spectra.)
            if self.add_pos_encode:
                # self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)
                self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=0.1)

        elif spectrum_type == 'cnmr':
            if self.cnmr_binary:
                # Binary tokenization for CNMR
                self.feature_extractor = nn.Embedding(self.cnmr_binary_bins + 1, self.d_model, padding_idx=0)

                # No need for post_conv_proj for embeddings
                self.post_conv_transform = nn.Identity()
            else:
                # Convolutional approach for continuous CNMR
                layers = []
                
                # First layer to capture peak patterns
                layers.append(nn.Conv1d(1, out_channels_1, kernel_size_1, stride=1, padding='valid'))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool1d(pool_size_1))
                
                # Second layer
                layers.append(nn.Conv1d(out_channels_1, out_channels_2, kernel_size_2, stride=1, padding='valid'))
                layers.append(nn.ReLU())
                layers.append(nn.MaxPool1d(pool_size_2))
                
                self.feature_extractor = nn.Sequential(*layers)
                self.post_conv_transform = nn.Linear(out_channels_2, self.d_model)

                # Positional encoding (matches the ir/hnmr branch above -- forward()
                # calls self.pos_encoder unconditionally for this non-binary conv path)
                if self.add_pos_encode:
                    self.pos_encoder = LearnablePositionalEncoding(d_model, dropout=0.1)
        else:
            raise ValueError(f"Unsupported spectrum type: {spectrum_type}")
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True)
        
        self.transformer = nn.TransformerEncoder(encoder_layer, 
                                                 num_layers=num_layers,
                                                 enable_nested_tensor=True)

    def _embed_cnmr(self, cnmr):
        assert(cnmr.shape[-1] == self.cnmr_binary_bins)
        if cnmr.ndim == 3:
            cnmr = cnmr.squeeze(1)
        padder_idx = self.cnmr_binary_bins * 2
        indices = torch.arange(1, self.cnmr_binary_bins + 1)
        indices = indices.to(cnmr.dtype).to(cnmr.device)

        cnmr = cnmr * indices
        cnmr[cnmr == 0] = padder_idx
        cnmr = torch.sort(cnmr).values
        cnmr[cnmr == padder_idx] = 0
        return cnmr.long()
    
    def forward(self, x):
        """
        Process spectral data with appropriate feature extraction
        
        Args:
            x: Input tensor [batch_size, sequence_length]
            
        Returns:
            features: Processed features [batch_size, seq_len, d_model]
            mask: Attention mask for padding (only for binary CNMR)
        """
        # Handle binary CNMR specially
        if self.spectrum_type == 'cnmr' and self.cnmr_binary:
            # Tokenize the binary CNMR data
            tokens = self._embed_cnmr(x)
            
            # Create mask for transformer (True values will be ignored)
            mask = (tokens == 0).bool().to(x.device)
            
            x = self.feature_extractor(tokens)  # [batch_size, seq_len, d_model]

            # Apply transformer with mask - disable autocast just for this operation
            device_type = 'cuda' if x.is_cuda else 'cpu'

            with torch.autocast(device_type=device_type, enabled=False):
                x = self.transformer(x, src_key_padding_mask=mask)   

            return x, mask
            
        else:
            # Standard convolutional processing for IR/HNMR/continuous CNMR
            
            # Add channel dimension
            x = x.unsqueeze(1)  # [batch_size, 1, sequence_length]
            
            # Apply integrated feature extraction (including pooling if configured)
            x = self.feature_extractor(x)  # [batch_size, channels, seq_len]
            
            # Reshape for projection
            x = x.transpose(1, 2)  # [batch_size, seq_len, channels]
            x = self.post_conv_transform(x)  # [batch_size, seq_len, d_model]
            
            # Add positional encoding if needed
            if self.add_pos_encode:
                x = self.pos_encoder(x, None)
            
            # Apply transformer encoder
            x = self.transformer(x)
            
            # Create empty mask (no padding)
            mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device)
            
            return x, mask
    
    
class MultiSpectralEncoder(nn.Module):
    """
    Encoder that can handle multiple types of spectral data
    """
    def __init__(
        self,
        d_model=128,
        n_hnmr_features=10000,  # Default HNMR features
        n_cnmr_features=10000,  # Default CNMR features
        n_ir_features=1800,  # Default IR features
        cnmr_binary=True,
        cnmr_binary_bins=80,
        pool_variation="max",
        pool_size_1 = 12,
        out_channels_1=64,
        kernel_size_1=5,
        pool_size_2=20,
        out_channels_2=128,
        kernel_size_2=9,
        add_pos_encode=True,
        nhead=8,
        fusion_layers=2,
        spectrum_encoder_layers=2,

        fusion_scheme='transformer',
        shared_encoder=False,
        use_ir=True,
        use_hnmr=True,
        use_cnmr=True,
        #Adding new CNMR processing
    ):
        super().__init__()
        self.d_model      = d_model
        self.use_ir       = use_ir
        self.use_hnmr     = use_hnmr
        self.use_cnmr     = use_cnmr
        self.shared_encoder= shared_encoder
        self.fusion_scheme= fusion_scheme

        self.n_ir_features = n_ir_features
        self.n_hnmr_features = n_hnmr_features
        self.n_cnmr_features = n_cnmr_features
        
        # Add CNMR-specific parameters
        self.cnmr_binary = cnmr_binary
        self.cnmr_binary_bins = cnmr_binary_bins
        self.pool_variation = pool_variation

        # Initialize pooling parameter
        self.pool_size_1 = pool_size_1
        self.pool_size_2 = pool_size_2
        self.kernel_size_1 = kernel_size_1
        self.kernel_size_2 = kernel_size_2

        # Calculate total spectral features based on enabled spectra
        total_spectral_features = 0
        if use_ir:
            total_spectral_features += self.n_ir_features
        if use_hnmr:
            total_spectral_features += self.n_hnmr_features

        self.n_spectral_features = total_spectral_features
        self.n_Cfeatures = cnmr_binary_bins if cnmr_binary else n_cnmr_features

        # Calculate actual modalities that will be used
        active_modalities = int(use_ir) + int(use_hnmr) + int(use_cnmr)        
        # Only need fusion if we have more than one modality actually used
        self.needs_fusion = active_modalities > 1

        # Calculate final sequence length after convolutions
        if use_hnmr or use_ir:
            self.h_spectrum_final_seq_len = self._compute_final_seq_len(
                self.n_spectral_features,
                [(kernel_size_1, pool_size_1, pool_variation), 
                 (kernel_size_2, pool_size_2, pool_variation)]
            )
        else:
            self.h_spectrum_final_seq_len = 0

        if self.use_ir:
            self.ir_encoder = SpectrumEncoder(
                d_model=d_model,
                nhead=nhead,
                pool_size_1=pool_size_1,
                out_channels_1=out_channels_1,
                kernel_size_1=kernel_size_1,
                pool_size_2=pool_size_2,
                out_channels_2=out_channels_2,
                kernel_size_2=kernel_size_2,
                add_pos_encode=add_pos_encode,
                spectrum_type="ir",
                num_layers=spectrum_encoder_layers
            )
        
        if self.use_hnmr:
            self.hnmr_encoder = SpectrumEncoder(
                d_model=d_model,
                nhead=nhead,
                pool_size_1=pool_size_1,
                out_channels_1=out_channels_1,
                kernel_size_1=kernel_size_1,
                pool_size_2=pool_size_2,
                out_channels_2=out_channels_2,
                kernel_size_2=kernel_size_2,
                add_pos_encode=add_pos_encode,
                spectrum_type="hnmr",
                num_layers=spectrum_encoder_layers,
            )
        
        if self.use_cnmr:
            # Use the unified SpectrumEncoder for both binary and continuous CNMR
            self.cnmr_encoder = SpectrumEncoder(
                d_model=d_model,
                nhead=nhead,
                pool_size_1=pool_size_1,
                out_channels_1=out_channels_1,
                kernel_size_1=kernel_size_1,
                pool_size_2=pool_size_2,
                out_channels_2=out_channels_2,
                kernel_size_2=kernel_size_2,
                add_pos_encode=add_pos_encode,
                num_layers=spectrum_encoder_layers,
                spectrum_type="cnmr",
                cnmr_binary=cnmr_binary,
                cnmr_binary_bins=cnmr_binary_bins
            )

        # Only initialize fusion transformer if we have more than one modality
        if self.needs_fusion and self.fusion_scheme == 'transformer':
            fusion_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                norm_first=False
            )
            self.fusion_transformer = nn.TransformerEncoder(fusion_layer,
                                                            num_layers=fusion_layers,
                                                            enable_nested_tensor=True)

            # Add a global positional encoder
            if add_pos_encode:
                self.global_pos_encoder = LearnablePositionalEncoding(d_model, dropout=0.1)

        # Add positional encoding
        self.add_pos_encode = add_pos_encode
        print(f"MultiSpectralEncoder initialized with:")
        print(f"- Total spectral features: {self.n_spectral_features}")
        print(f"- Using HNMR: {self.use_hnmr} (length: {self.n_hnmr_features})")
        print(f"- Using CNMR: {self.use_cnmr} ({'binary' if self.cnmr_binary else 'continuous'}, length: {self.n_Cfeatures})")
        print(f"- Using IR: {self.use_ir} (length: {self.n_ir_features})")
        if use_hnmr or use_ir:
            print(f"- Final sequence length after convolutions: {self.h_spectrum_final_seq_len}")

    def _separate_spectra_components(self, x):
        """
        Separate the concatenated spectra into different components.
        Similar to NMR2Struct's approach.
        
        Args:
            x: Concatenated spectra tensor [batch_size, seq_len] or [batch_size, 1, seq_len]
        
        Returns:
            ir_data, hnmr_data, cnmr_data: Separated spectral components
        """
        # Ensure x has the right shape
        if len(x.shape) == 3:
            # [batch_size, 1, seq_len]
            x = x.squeeze(1)
        
        # Initialize empty tensors for each component
        ir_data = None
        hnmr_data = None
        cnmr_data = None
        
        # Process in the expected order: IR, HNMR, CNMR
        offset = 0
        
        # Extract IR data
        if self.use_ir:
            ir_len = self.n_ir_features if hasattr(self, 'n_ir_features') else 1800
            ir_data = x[:, offset:offset+ir_len]
            offset += ir_len
        
        # Extract HNMR data
        if self.use_hnmr:
            hnmr_len = self.n_hnmr_features if hasattr(self, 'n_hnmr_features') else 10000
            hnmr_data = x[:, offset:offset+hnmr_len]
            offset += hnmr_len
        
        # Extract CNMR data
        if self.use_cnmr:
            if self.cnmr_binary:
                cnmr_len = self.cnmr_binary_bins
            else:
                cnmr_len = self.n_cnmr_features if hasattr(self, 'n_cnmr_features') else 10000
            cnmr_data = x[:, offset:offset+cnmr_len]
        
        return ir_data, hnmr_data, cnmr_data


    def forward(self, spectra=None):
        """Process multiple types of spectral data and fuse them"""
        features = []
        mask_parts = []  # For collecting attention masks

        # If concatenated spectra is provided, separate it into components
        if spectra is not None:
            ir_data, hnmr_data, cnmr_data = self._separate_spectra_components(spectra)

        # Original implementation: complete separate encoders
        # Process IR data if available
        if self.use_ir and ir_data is not None:
            ir_features, ir_mask = self.ir_encoder(ir_data)
            features.append(ir_features)
            mask_parts.append(ir_mask)

        # Process HNMR data if available
        if self.use_hnmr and hnmr_data is not None:
            hnmr_features, hnmr_mask = self.hnmr_encoder(hnmr_data)
            features.append(hnmr_features)
            mask_parts.append(hnmr_mask)

        # Process CNMR data if available
        if self.use_cnmr and cnmr_data is not None:
            # No need for conditional logic - the encoder handles both approaches
            cnmr_features, cnmr_mask = self.cnmr_encoder(cnmr_data)
            features.append(cnmr_features)
            mask_parts.append(cnmr_mask)
            
        # Handle empty features case
        if not features:
            raise ValueError("No valid features to process - check spectra_types and input data")

        # For both single and multiple modalities, proceed with consistent processing
        if len(features) == 1:
            # Single modality case - still apply pooling for consistency
            fused_features = features[0]
            fused_mask = mask_parts[0] if mask_parts else None
            
            return fused_features, fused_mask
        
        if self.needs_fusion:
            # Concatenate features and masks
            fused_features = torch.cat(features, dim=1)
            fused_mask = torch.cat(mask_parts, dim=1)
            
            if self.add_pos_encode:
                fused_features = self.global_pos_encoder(fused_features, None)
                
            if self.fusion_scheme == 'transformer':
                # Apply transformer with mask - disable autocast just for this operation
                device_type = 'cuda' if fused_features.is_cuda else 'cpu'

                with torch.autocast(device_type=device_type, enabled=False):
                    fused_features = self.fusion_transformer(
                        fused_features, 
                        src_key_padding_mask=fused_mask)
                
        return fused_features, fused_mask


    def _calculate_dim_after_conv(self, L_in, kernel, padding, dilation=1, stride=1):
        """Calculate output dimension after convolution"""
        if padding == 'valid':
            padding = 0
        numerator = L_in + (2 * padding) - (dilation * (kernel - 1)) - 1
        return math.floor((numerator/stride) + 1)
    
    def _calculate_dim_after_pool(self, pool_variation, L_in, kernel, padding=0, dilation=1, stride=None):
        """Calculate output dimension after pooling"""
        if stride is None:
            stride = kernel
            
        if pool_variation == 'max':
            numerator = L_in + (2 * padding) - (dilation * (kernel - 1)) - 1
            return math.floor((numerator/stride) + 1)
        elif pool_variation == 'avg':
            numerator = L_in + (2 * padding) - kernel
            return math.floor((numerator/stride) + 1)
    
    def _compute_final_seq_len(self, L_in, block_args):
        """Compute final sequence length after all conv+pool blocks"""
        curr_L = L_in
        for kernel, pool, pool_variation in block_args:
            curr_L = self._calculate_dim_after_conv(curr_L, kernel, 'valid')
            curr_L = self._calculate_dim_after_pool(pool_variation, curr_L, pool)
        return curr_L

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Make positions directly learnable
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, mean=0, std=0.02)  # Initialize with small values
    
    def forward(self, x, ind=None):
        if ind is not None:
            # Use provided indices to select positions
            added_pe = self.pe[torch.arange(1).reshape(-1, 1), ind, :]
            x = x + added_pe
        else:
            # Use sequential positions
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
class PositionalEncoding(nn.Module):
    """ Positional encoding with option of selecting specific indices to add """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 30000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor, ind: Tensor) -> Tensor:
        '''
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
            ind: Tensor, shape [batch_size, seq_len] or NoneType
        '''
        #Select and expand the PE to be the right shape first
        if ind is not None:
            added_pe = self.pe[torch.arange(1).reshape(-1, 1), ind, :]
            x = x + added_pe
        else:
            x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
    
# Why Use a Transformer for Tokenized CNMR?
# Using a transformer on the embedded tokens has several advantages:

# Capturing Peak Relationships: The transformer can learn relationships between different CNMR peaks, which often appear in characteristic patterns

# Handling Variable-Length Sequences: Transformers with masking naturally handle the variable number of peaks in different molecules

# Consistency with NMR2Struct: This approach follows the NMR2Struct pipeline, which processes tokenized CNMR through their transformer

# Full Architecture Comparison
# Processing Step	Continuous IR/HNMR	Binary CNMR
# Feature Extraction	Convolutional layers	Token embedding
# Dimensionality Reduction	MaxPooling	Tokenization
# Positional Information	Positional encoding	Inherent in tokens
# Contextual Processing	Transformer encoder	Transformer encoder
# Mask Handling	None (dense)	Padding mask