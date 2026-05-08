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
FLAT_DATE   = DATE_SEQ_LEN * VOCAB_SIZE
LATENT_DIM  = 64


class VAEDateGenerator(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.day_emb    = nn.Embedding(NUM_DAYS,    EMBED_DIM)
        self.month_emb  = nn.Embedding(NUM_MONTHS,  EMBED_DIM)
        self.leap_emb   = nn.Embedding(NUM_LEAPS,   EMBED_DIM)
        self.decade_emb = nn.Embedding(NUM_DECADES, EMBED_DIM)

        self.encoder = nn.Sequential(
            nn.Linear(FLAT_DATE + COND_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.fc_mu      = nn.Linear(256, LATENT_DIM)
        self.fc_log_var = nn.Linear(256, LATENT_DIM)

        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM + COND_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, FLAT_DATE),
        )

    def encode_conditions(self, conditions: torch.Tensor) -> torch.Tensor:
        d   = self.day_emb(conditions[:, 0])
        m   = self.month_emb(conditions[:, 1])
        l   = self.leap_emb(conditions[:, 2])
        dec = self.decade_emb(conditions[:, 3])
        return torch.cat([d, m, l, dec], dim=-1)

    def encode(self, date_onehot: torch.Tensor,
               cond: torch.Tensor) -> tuple:
        x       = torch.cat([date_onehot.view(date_onehot.size(0), -1), cond], dim=-1)
        h       = self.encoder(x)
        mu      = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(self, mu: torch.Tensor,
                       log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor,
               cond: torch.Tensor) -> torch.Tensor:
        x   = torch.cat([z, cond], dim=-1)
        out = self.decoder(x)
        return out.view(-1, DATE_SEQ_LEN, VOCAB_SIZE)

    def forward(self, date_tokens: torch.Tensor,
                conditions: torch.Tensor) -> tuple:
        import torch.nn.functional as F
        cond       = self.encode_conditions(conditions)
        onehot     = F.one_hot(date_tokens, num_classes=VOCAB_SIZE).float()
        mu, log_var = self.encode(onehot, cond)
        z          = self.reparameterize(mu, log_var)
        recon      = self.decode(z, cond)
        return recon, mu, log_var

    @torch.no_grad()
    def generate(self, conditions: torch.Tensor) -> list:
        import torch.nn.functional as F
        from tokenizer import DateTokenizer
        self.eval()
        tok  = DateTokenizer()
        cond = self.encode_conditions(conditions)
        z    = torch.randn(conditions.size(0), LATENT_DIM, device=conditions.device)
        out  = self.decode(z, cond)
        tokens = out.argmax(dim=-1)
        return [tok.decode_date(tokens[i].tolist()) for i in range(tokens.size(0))]