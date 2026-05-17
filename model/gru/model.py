import torch
import torch.nn as nn


class GRUGenerator(nn.Module):

    def __init__(
        self,
        vocab_size=100,
        embedding_dim=128,
        hidden_dim=256
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        self.gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )

    def forward(self, x):

        x = self.embedding(x)

        output, _ = self.gru(x)

        output = self.fc(output)

        return output

    def generate(self, x):

        output = self.forward(x)

        output = torch.argmax(output, dim=-1)

        return output