import torch
import torch.nn as nn
from typing import List

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from tokenizer import (VOCAB_SIZE, SOS_TOKEN, EOS_TOKEN,
                       PAD_TOKEN, DATE_SEQ_LEN,
                       DAY_TOKENS, MONTH_TOKENS, LEAP_TOKENS, MIN_DECADE, MAX_DECADE)


class LSTMDateGenerator(nn.Module):

    def __init__(self, day_vocab: int = 7, month_vocab: int = 12,
                 leap_vocab: int = 2, decade_vocab: int = 41,
                 embed_dim: int = 32, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()

        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers

        self.day_emb    = nn.Embedding(day_vocab,    embed_dim)
        self.month_emb  = nn.Embedding(month_vocab,  embed_dim)
        self.leap_emb   = nn.Embedding(leap_vocab,   embed_dim)
        self.decade_emb = nn.Embedding(decade_vocab, embed_dim)

        self.cond_proj = nn.Linear(embed_dim * 4, hidden_dim)

        self.token_emb = nn.Embedding(VOCAB_SIZE, embed_dim)

        self.lstm = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout
        )

        self.fc_out = nn.Linear(hidden_dim, VOCAB_SIZE)

    def encode_conditions(self, conditions: torch.Tensor) -> tuple:
        day_e    = self.day_emb(conditions[:, 0])
        month_e  = self.month_emb(conditions[:, 1])
        leap_e   = self.leap_emb(conditions[:, 2])
        decade_e = self.decade_emb(conditions[:, 3])

        cond_vec = torch.cat([day_e, month_e, leap_e, decade_e], dim=-1)
        h0       = torch.tanh(self.cond_proj(cond_vec))
        h0       = h0.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0       = torch.zeros_like(h0)
        return h0, c0

    def forward(self, conditions: torch.Tensor,
                date_tokens: torch.Tensor) -> torch.Tensor:
        h, c = self.encode_conditions(conditions)

        sos = torch.full((date_tokens.size(0), 1), SOS_TOKEN,
                         dtype=torch.long, device=date_tokens.device)
        dec_input = torch.cat([sos, date_tokens[:, :-1]], dim=1)

        emb, _ = self.lstm(self.token_emb(dec_input), (h, c))
        logits  = self.fc_out(emb)
        return logits

    @torch.no_grad()
    def generate(self, conditions: torch.Tensor) -> List[str]:
        self.eval()
        from tokenizer import DateTokenizer
        tok = DateTokenizer()

        h, c   = self.encode_conditions(conditions)
        device = conditions.device
        B      = conditions.size(0)

        token = torch.full((B, 1), SOS_TOKEN, dtype=torch.long, device=device)
        generated = [[] for _ in range(B)]

        for _ in range(DATE_SEQ_LEN):
            emb, (h, c) = self.lstm(self.token_emb(token), (h, c))
            logits      = self.fc_out(emb.squeeze(1))
            token       = logits.argmax(dim=-1, keepdim=True)
            for i in range(B):
                generated[i].append(token[i].item())

        return [tok.decode_date(seq) for seq in generated]