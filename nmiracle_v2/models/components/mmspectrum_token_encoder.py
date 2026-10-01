import torch
import torch.nn as nn

from nmiracle_v2.models.components.multispectra_encoder import LearnablePositionalEncoding


class MMSpectrumTokenEncoder(nn.Module):
    """
    Embedding-lookup encoder over tokenized spectral sequences produced by
    data/alberts_baseline_tokenize.py. Reuses the same nn.TransformerEncoder
    building block already used for the binary-CNMR token pathway in
    SpectrumEncoder -- only the front-end tokenization differs, not the
    transformer itself. Returns (features, mask) with the same contract as
    MultiSpectralEncoder / ConvolutionalSpectrumEncoder, so it's a drop-in
    alternative spectrum_encoder.
    """

    def __init__(
        self,
        vocab_size,
        pad_id=1,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        add_pos_encode=True,
        max_len=512,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.add_pos_encode = add_pos_encode
        self.num_layers = num_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

        if self.add_pos_encode:
            # max_len must be >= the padded token length (alberts_baseline_max_len
            # etc.) -- LearnablePositionalEncoding's own default (500) silently
            # clips via slicing rather than erroring, so a mismatch here shows up
            # as a broadcast RuntimeError deep in forward(), not a clear message.
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        # num_layers == 0 makes this a pure source *embedding* (lookup + positional
        # encoding), leaving all encoding to the downstream structure model's
        # encoder. This is how we mirror OpenNMT/Alberts' single 4-encoder/4-decoder
        # seq2seq: token embed here -> structure_model's 4 encoder layers -> 4
        # decoder layers, instead of stacking two encoders back-to-back.
        # (A literal 0-layer nn.TransformerEncoder raises IndexError in torch 2.11,
        # so the transformer is simply not built / not applied when num_layers == 0.)
        if num_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers, enable_nested_tensor=True
            )
        else:
            self.transformer = None

    def forward(self, token_ids):
        """
        Args:
            token_ids: [batch_size, seq_len] long tensor of vocab ids

        Returns:
            features: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len] bool, True where padded
        """
        mask = token_ids.eq(self.pad_id)
        x = self.token_embedding(token_ids)

        if self.add_pos_encode:
            x = self.pos_encoder(x, None)

        if self.transformer is not None:
            device_type = 'cuda' if x.is_cuda else 'cpu'
            with torch.autocast(device_type=device_type, enabled=False):
                x = self.transformer(x, src_key_padding_mask=mask)

        return x, mask
