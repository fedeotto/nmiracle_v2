import torch
import torch.nn as nn
import math
import numpy as np
from torch import Tensor

class ConvolutionalSpectrumEncoder(nn.Module):
    """
    Implements NMR2Struct's approach of concatenating spectra and processing with CNNs
    """
    def __init__(
        self,
        d_model=768,
        n_hnmr_features=10000,
        n_cnmr_features=10000,
        n_ir_features=1800,
        cnmr_binary=True,
        cnmr_binary_bins=80,
        pool_variation="max",
        pool_size_1=12,
        out_channels_1=64,
        kernel_size_1=5,
        pool_size_2=20,
        out_channels_2=128,
        kernel_size_2=9,
        add_pos_encode=True,
        use_ir=False,
        use_hnmr=True,
        use_cnmr=True,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Store all the spectral configuration options
        self.use_hnmr = use_hnmr
        self.use_cnmr = use_cnmr
        self.use_ir = use_ir
        self.cnmr_binary = cnmr_binary
        self.cnmr_binary_bins = cnmr_binary_bins
        
        # Store lengths of each spectrum type
        self.hnmr_len = n_hnmr_features
        self.cnmr_len = n_cnmr_features
        self.ir_len = n_ir_features
        
        # Calculate total spectral features based on enabled spectra
        total_spectral_features = 0
        if use_ir:
            total_spectral_features += self.ir_len
        if use_hnmr:
            total_spectral_features += self.hnmr_len
        
        self.n_spectral_features = total_spectral_features
        self.n_Cfeatures = cnmr_binary_bins if cnmr_binary else n_cnmr_features
        
        # CNMR embedding (for binary CNMR features)
        if use_cnmr:
            self.c_embed = nn.Embedding(self.n_Cfeatures + 1, self.d_model, padding_idx=0)
        
        # CNN layers for HNMR/IR processing
        if use_hnmr or use_ir:
            # Pooling layers
            if pool_variation == 'max':
                self.pool1 = nn.MaxPool1d(pool_size_1)
                self.pool2 = nn.MaxPool1d(pool_size_2)
            else:
                self.pool1 = nn.AvgPool1d(pool_size_1)
                self.pool2 = nn.AvgPool1d(pool_size_2)
            
            # Convolutional layers
            self.conv1 = nn.Conv1d(1, out_channels_1, kernel_size_1, stride=1, padding='valid')
            self.conv2 = nn.Conv1d(out_channels_1, out_channels_2, kernel_size_2, stride=1, padding='valid')
            self.relu = nn.ReLU()
            self.post_conv_transform = nn.Linear(out_channels_2, self.d_model)
        
        # Calculate final sequence length after convolutions
        if use_hnmr or use_ir:
            self.h_spectrum_final_seq_len = self._compute_final_seq_len(
                self.n_spectral_features,
                [(kernel_size_1, pool_size_1, pool_variation), 
                 (kernel_size_2, pool_size_2, pool_variation)]
            )
        else:
            self.h_spectrum_final_seq_len = 0
                    
        # Add positional encoding flag
        self.add_pos_encoder = add_pos_encode
        
        # Positional encoding (can be added here or in the transformer)
        if self.add_pos_encoder:
            self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)
        
        print(f"ConvolutionalSpectrumEncoder initialized with:")
        print(f"- Total spectral features: {self.n_spectral_features}")
        print(f"- Using HNMR: {self.use_hnmr} (length: {self.hnmr_len})")
        print(f"- Using CNMR: {self.use_cnmr} ({'binary' if self.cnmr_binary else 'continuous'}, length: {self.n_Cfeatures})")
        print(f"- Using IR: {self.use_ir} (length: {self.ir_len})")
        if use_hnmr or use_ir:
            print(f"- Final sequence length after convolutions: {self.h_spectrum_final_seq_len}")
    
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
    
    def _separate_spectra_components(self, x):
        """
        Separate the input tensor into different spectral components
        (excluding formula, which is passed separately)
        """
        if len(x.shape) == 2:
            x = torch.unsqueeze(x, 1)
            
        # Process spectra in expected order
        offset = 0
        
        # These go to convolutional pathway (IR and HNMR)
        spectral_components = []
        if self.use_ir:
            spectral_components.append(x[:, :, offset:offset+self.ir_len])
            offset += self.ir_len
        if self.use_hnmr:
            spectral_components.append(x[:, :, offset:offset+self.hnmr_len])
            offset += self.hnmr_len
            
        # Combine components for convolutional path
        if spectral_components:
            spectral_x = torch.cat(spectral_components, dim=2)
        else:
            spectral_x = torch.zeros(x.shape[0], x.shape[1], 0).to(x.device)
        
        # CNMR goes to embedding pathway
        if self.use_cnmr:
            cnmr_x = x[:, :, offset:offset+self.n_Cfeatures]
        else:
            cnmr_x = torch.zeros(x.shape[0], x.shape[1], 0).to(x.device)
        
        # Return spectral components and CNMR (no formula)
        
        return spectral_x, cnmr_x
    
    def _embed_cnmr(self, cnmr):
        """Embeds the binary tensor into a continuous space
        Convert to 1-indexed indices, pad with 0
        """
        assert(cnmr.shape[-1] == self.n_Cfeatures)
        if cnmr.ndim == 3:
            cnmr = cnmr.squeeze(1)
        padder_idx = self.n_Cfeatures * 2
        indices = torch.arange(0, self.n_Cfeatures) + 1
        indices = indices.to(cnmr.dtype).to(cnmr.device)
        cnmr = cnmr * indices
        cnmr[cnmr == 0] = padder_idx
        cnmr = torch.sort(cnmr).values
        cnmr[cnmr == padder_idx] = 0
        return self.c_embed(cnmr.long())
    
    def _embed_spectra(self, spectra):
        """Process spectra through CNN layers"""
        assert spectra.ndim == 3
        if spectra.shape[2] == 0:
            return torch.tensor([]).to(spectra.device)
        # Apply CNN processing
        spectra = self.conv1(spectra)
        spectra = self.relu(spectra)
        spectra = self.pool1(spectra)
        spectra = self.conv2(spectra)
        spectra = self.relu(spectra)
        spectra = self.pool2(spectra)  # (batch_size, channels, seq_len)
        
        # Transpose for sequence processing
        spectra = torch.transpose(spectra, 1, 2)  # (batch_size, seq_len, channels)
        
        # Project to model dimension
        return self.post_conv_transform(spectra)
    
    def _process_formula(self, formula):
        """
        Process molecular formula features
        
        Args:
            formula: Formula tensor [batch_size, 1, features] or [batch_size, features]
        """

        # Ensure formula has the right shape
        if len(formula.shape) == 3:
            formula = formula.squeeze(1)
        
        # Process through MLP
        processed = self.formula_processor(formula)
        
        # Add sequence dimension if needed
        if len(processed.shape) == 2:
            processed = processed.unsqueeze(1)
        
        return processed
    
    def forward(self, x):
        """
        Process spectral input with optional formula
        
        Args:
            x: Input tensor containing concatenated spectra 
            [batch_size, seq_len] or [batch_size, 1, seq_len]
            formula: Optional tensor with molecular formula features
                    [batch_size, max_atomic_number]
        """
        # Separate spectral components (excluding formula)
        spectra, cnmr = self._separate_spectra_components(x)
        
        # Process each component
        output_components = []
        mask_parts = []
        
        # Process HNMR/IR through CNN path
        if (self.use_hnmr or self.use_ir) and spectra.shape[2] > 0:
            spectra_embed = self._embed_spectra(spectra)

            # For HNMR and IR - generally treat as dense features
            spectral_mask = torch.zeros(spectra.shape[0], 
                                    spectra_embed.shape[1]).bool().to(spectra.device)
        
            output_components.append(spectra_embed)
            mask_parts.append(spectral_mask)

        # Process CNMR through embedding path
        if self.use_cnmr and cnmr.shape[2] > 0:
            cnmr_embed = self._embed_cnmr(cnmr)
            output_components.append(cnmr_embed)

            if self.cnmr_binary:
                cnmr_mask = (cnmr_embed == 0).any(dim=-1).bool().to(cnmr_embed.device)
            else:
                # For continuous, threshold and mask
                cnmr_mask = (cnmr_embed < self.eps).any(dim=-1).bool().to(cnmr_embed.device)
            mask_parts.append(cnmr_mask)
    
        # Combine all components
        if not output_components:
            raise ValueError("No spectral components were processed")
            
        # Concatenate along sequence dimension
        embedded = torch.cat(output_components, dim=1)

        # Combine all masks
        if mask_parts:
            src_key_pad_mask = torch.cat(mask_parts, dim=1)
        else:
            src_key_pad_mask = torch.zeros(x.shape[0], embedded.shape[1]).bool().to(x.device)
        
        embedded = embedded * math.sqrt(self.d_model)  # Scale by sqrt(d_model)
        # Apply positional encoding if enabled
        if self.add_pos_encoder:
            embedded = self.pos_encoder(embedded, None)
            
        return embedded, src_key_pad_mask


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