# model/shared/dataset.py

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple
import random

from tokenizer import DateTokenizer


class DatesDataset(Dataset):
    """Loads data.txt and returns (conditions_tensor, date_tokens_tensor)."""

    def __init__(self, filepath: str, seed: int = 42) -> None:
        self.tokenizer = DateTokenizer()
        self.samples: List[Tuple[Tuple, List[int]]] = []

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                conditions, date_tokens = self.tokenizer.parse_line(line)
                self.samples.append((conditions, date_tokens))

        random.seed(seed)
        random.shuffle(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        conditions, date_tokens = self.samples[idx]

        cond_tensor = torch.tensor(conditions, dtype=torch.long)

        date_tensor = torch.tensor(date_tokens, dtype=torch.long)

        return cond_tensor, date_tensor


def get_dataloaders(
    filepath: str,
    batch_size: int = 256,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    dataset = DatesDataset(filepath, seed=seed)
    total = len(dataset)

    train_size = int(total * train_ratio)
    val_size   = int(total * val_ratio)
    test_size  = total - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    print(f"Total samples : {total:,}")
    print(f"Train samples : {train_size:,}")
    print(f"Val samples   : {val_size:,}")
    print(f"Test samples  : {test_size:,}")

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        filepath="data/data.txt"
    )

    # Peek at one batch
    cond_batch, date_batch = next(iter(train_loader))
    print("\nConditions batch shape:", cond_batch.shape)   
    print("Date batch shape      :", date_batch.shape)     
    print("\nSample condition:", cond_batch[0])
    print("Sample date     :", date_batch[0])