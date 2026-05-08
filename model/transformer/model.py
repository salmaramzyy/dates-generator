import torch
import torch.nn as nn
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from tokenizer import DATE_SEQ_LEN, VOCAB_SIZE, MIN_DECADE, MAX_DECADE

NUM_DAYS    = 7
NUM_MONTHS  = 12
NUM_LEAPS   = 2
NUM_DECADES = MAX_DECADE - MIN_DECADE + 1
EMBED_DIM   = 16
COND_DIM    = EMBED_DIM * 4
D_MODEL     = 128
NHEAD       = 4
NUM_LAYERS  = 3
DIM_FF      = 256
DROPOUT     = 0.1


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 50) -> None:
        super().__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerDateGenerator(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.day_emb    = nn.Embedding(NUM_DAYS,    EMBED_DIM)
        self.month_emb  = nn.Embedding(NUM_MONTHS,  EMBED_DIM)
        self.leap_emb   = nn.Embedding(NUM_LEAPS,   EMBED_DIM)
        self.decade_emb = nn.Embedding(NUM_DECADES, EMBED_DIM)

        self.cond_proj  = nn.Linear(COND_DIM, D_MODEL)
        self.token_emb  = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_enc    = PositionalEncoding(D_MODEL)

        decoder_layer   = nn.TransformerDecoderLayer(
            d_model=D_MODEL, nhead=NHEAD,
            dim_feedforward=DIM_FF, dropout=DROPOUT, batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=NUM_LAYERS)
        self.fc_out = nn.Linear(D_MODEL, VOCAB_SIZE)

    def encode_conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        d   = self.day_emb(conditions[:, 0])
        m   = self.month_emb(conditions[:, 1])
        l   = self.leap_emb(conditions[:, 2])
        dec = self.decade_emb(conditions[:, 3])
        cond_vec = torch.cat([d, m, l, dec], dim=-1)
        return self.cond_proj(cond_vec).unsqueeze(1)

    def forward(self, conditions: torch.Tensor,
                date_tokens: torch.Tensor) -> torch.Tensor:
        memory   = self.encode_conditions(conditions)
        sos      = torch.full((date_tokens.size(0), 1), 12,
                              dtype=torch.long, device=date_tokens.device)
        dec_input = torch.cat([sos, date_tokens[:, :-1]], dim=1)
        tgt       = self.pos_enc(self.token_emb(dec_input))
        tgt_mask  = nn.Transformer.generate_square_subsequent_mask(
            tgt.size(1), device=tgt.device)
        out       = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        return self.fc_out(out)

    @torch.no_grad()
    def generate(self, conditions: torch.Tensor) -> list:
        from tokenizer import DateTokenizer, SOS_TOKEN, EOS_TOKEN, DATE_SEQ_LEN
        self.eval()
        tok    = DateTokenizer()
        B      = conditions.size(0)
        device = conditions.device
        memory = self.encode_conditions(conditions)
        token  = torch.full((B, 1), SOS_TOKEN, dtype=torch.long, device=device)
        generated = [[] for _ in range(B)]

        for _ in range(DATE_SEQ_LEN):
            tgt      = self.pos_enc(self.token_emb(token))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tgt.size(1), device=device)
            out      = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
            logits   = self.fc_out(out[:, -1, :])
            next_tok = logits.argmax(dim=-1, keepdim=True)
            token    = torch.cat([token, next_tok], dim=1)
            for i in range(B):
                generated[i].append(next_tok[i].item())

        return [tok.decode_date(seq) for seq in generated]