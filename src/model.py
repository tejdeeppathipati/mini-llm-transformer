# --- model.py ---
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)  # Shape: [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)  # even dimensions → sin
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dimensions → cos

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)  # Not learnable, just stored

    def forward(self, x):
        """
        x shape: [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        # Add positional encodings
        x = x + self.pe[:, :seq_len].to(x.device)
        return x

class TransformerChatbot(nn.Module):
    """
    A Transformer-based seq2seq model that:
      - Expects precomputed embeddings for src (e.g. from ELECTRA) 
      - Still has an embedding + positional encoding for tgt tokens
    """
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 dim_feedforward=512,
                 dropout=0.1,
                 max_len=128,
                 vocab_size=30522):
        super().__init__()

        self.d_model = d_model

        # We'll only embed the decoder side (tgt) here.
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)

        # Positional encodings for both src & tgt
        self.pos_encoder_src = PositionalEncoding(d_model, max_len)
        self.pos_encoder_tgt = PositionalEncoding(d_model, max_len)

        # Transformer encoder/decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                src_embeddings,
                tgt_ids,
                src_mask=None,
                tgt_mask=None,
                src_padding_mask=None,
                tgt_padding_mask=None):
        """
        src_embeddings: [B, src_len, d_model], from ELECTRA or another model.
        tgt_ids: [B, tgt_len]
        """
        if src_embeddings.size(-1) != self.d_model:
            raise ValueError(f"Expected src_embeddings of dimension {self.d_model}, got {src_embeddings.size(-1)}")

        # 1) We do NOT embed src here — user already supplied src_embeddings.
        src = self.pos_encoder_src(src_embeddings)  # shape: [B, src_len, d_model]

        # 2) Embed + position-encode the target tokens
        tgt = self.tgt_embedding(tgt_ids)                     # [B, tgt_len, d_model]
        tgt = self.pos_encoder_tgt(tgt)                       # apply pos encoding

        # 3) Pass through Transformer
        memory = self.encoder(src, src_key_padding_mask=src_padding_mask)
        output = self.decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask
        )
        # 4) Final linear layer → vocab logits
        return self.fc_out(output)

def generate_square_subsequent_mask(sz):
    """
    Creates a causal mask for the target sequence to ensure auto-regressive decoding.
    shape: [sz, sz], upper-triangular with -inf above the main diagonal
    """
    return torch.triu(torch.ones((sz, sz)) * float('-inf'), diagonal=1)
