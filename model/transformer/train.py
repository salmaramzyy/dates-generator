import torch
import torch.nn as nn
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from dataset import get_dataloaders
from utils import condition_accuracy, plot_losses, save_metrics
from tokenizer import DateTokenizer
from model import TransformerDateGenerator

DEVICE     = torch.device("cpu")
EPOCHS     = 20
BATCH_SIZE = 256
LR         = 1e-3
SEED       = 42

torch.manual_seed(SEED)

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
LOGS_DIR    = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


def evaluate(model: TransformerDateGenerator, loader,
             criterion: nn.CrossEntropyLoss) -> float:
    model.eval()
    total = 0.0
    with torch.no_grad():
        for cond, date_tokens in loader:
            cond        = cond.to(DEVICE)
            date_tokens = date_tokens.to(DEVICE)
            logits      = model(cond, date_tokens)
            loss        = criterion(logits.reshape(-1, logits.size(-1)),
                                    date_tokens.reshape(-1))
            total += loss.item()
    return total / len(loader)


def train() -> None:
    train_loader, val_loader, test_loader = get_dataloaders(
        filepath=os.path.join(os.path.dirname(__file__), "../../data/data.txt"),
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    model     = TransformerDateGenerator().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for cond, date_tokens in train_loader:
            cond        = cond.to(DEVICE)
            date_tokens = date_tokens.to(DEVICE)

            optimizer.zero_grad()
            logits = model(cond, date_tokens)
            loss   = criterion(logits.reshape(-1, logits.size(-1)),
                               date_tokens.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        val_loss   = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(WEIGHTS_DIR, "best.pt"))
            print(f"  ✓ Best model saved")

    plot_losses(train_losses, val_losses, "Transformer Loss", os.path.join(LOGS_DIR, "loss.png"))

    model.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "best.pt")))
    model.eval()

    tok = DateTokenizer()
    all_preds, all_conds = [], []

    with torch.no_grad():
        for cond_b, _ in test_loader:
            cond_b = cond_b.to(DEVICE)
            preds  = model.generate(cond_b)
            all_preds.extend(preds)
            all_conds.extend([tuple(c.tolist()) for c in cond_b])

    metrics = condition_accuracy(all_preds, all_conds)

    print("\n── Transformer Test Metrics ──")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.2f}%")

    save_metrics(metrics, os.path.join(LOGS_DIR, "test_metrics.txt"))

    with open(os.path.join(LOGS_DIR, "sample_predictions.txt"), "w") as f:
        for i in range(min(20, len(all_preds))):
            f.write(f"{all_preds[i]}\n")

    print(f"\nLogs saved to {LOGS_DIR}")


if __name__ == "__main__":
    train()