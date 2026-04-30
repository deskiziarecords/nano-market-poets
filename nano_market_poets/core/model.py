import torch
import torch.nn as nn
import torch.nn.functional as F

class NanoMold(nn.Module):
    def __init__(self, vocab_size=7, d_model=128, n_heads=4, n_layers=4, max_seq_len=64):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Transformer Decoder for Causal Prediction
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, 
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, idx):
        # idx: [Batch, SeqLen]
        B, T = idx.size()
        assert T <= self.pos_emb.size(1)
        
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        
        # Causal Mask
        mask = nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        # For TransformerDecoder, mask is passed as tgt_mask
        x = self.transformer(x, x, tgt_mask=mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        return logits[:, -1, :] # Prediction for NEXT token
