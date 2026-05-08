import torch
import torch.nn as nn
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))
from tokenizer import DATE_SEQ_LEN, VOCAB_SIZE, MIN_DECADE, MAX_DECADE

NUM_DAYS    = 7
NUM_MONTHS  = 12
NUM_LEAPS   = 2
NUM_DECADES = MAX_DECADE - MIN_DECADE + 1
EMBED_DIM   = 16
COND_DIM    = EMBED_DIM * 4
NOISE_DIM   = 64
FLAT_DATE   = DATE_SEQ_LEN * VOCAB_SIZE


class Generator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.day_emb    = nn.Embedding(NUM_DAYS,    EMBED_DIM)
        self.month_emb  = nn.Embedding(NUM_MONTHS,  EMBED_DIM)
        self.leap_emb   = nn.Embedding(NUM_LEAPS,   EMBED_DIM)
        self.decade_emb = nn.Embedding(NUM_DECADES, EMBED_DIM)

        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM + COND_DIM, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, FLAT_DATE),
        )

    def encode_conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        d   = self.day_emb(conditions[:, 0])
        m   = self.month_emb(conditions[:, 1])
        l   = self.leap_emb(conditions[:, 2])
        dec = self.decade_emb(conditions[:, 3])
        return torch.cat([d, m, l, dec], dim=-1)

    def forward(self, noise: torch.Tensor,
                conditions: torch.Tensor) -> torch.Tensor:
        cond = self.encode_conditions(conditions)
        x    = torch.cat([noise, cond], dim=-1)
        out  = self.net(x)
        return out.view(-1, DATE_SEQ_LEN, VOCAB_SIZE)


class Discriminator(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.day_emb    = nn.Embedding(NUM_DAYS,    EMBED_DIM)
        self.month_emb  = nn.Embedding(NUM_MONTHS,  EMBED_DIM)
        self.leap_emb   = nn.Embedding(NUM_LEAPS,   EMBED_DIM)
        self.decade_emb = nn.Embedding(NUM_DECADES, EMBED_DIM)

        self.net = nn.Sequential(
            nn.Linear(FLAT_DATE + COND_DIM, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def encode_conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        d   = self.day_emb(conditions[:, 0])
        m   = self.month_emb(conditions[:, 1])
        l   = self.leap_emb(conditions[:, 2])
        dec = self.decade_emb(conditions[:, 3])
        return torch.cat([d, m, l, dec], dim=-1)

    def forward(self, date_onehot: torch.Tensor,
                conditions: torch.Tensor) -> torch.Tensor:
        cond = self.encode_conditions(conditions)
        x    = torch.cat([date_onehot.view(date_onehot.size(0), -1), cond], dim=-1)
        return self.net(x)