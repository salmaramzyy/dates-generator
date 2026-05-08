import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared'))

from dataset import get_dataloaders
from utils import condition_accuracy, plot_losses, save_metrics
from tokenizer import DateTokenizer, VOCAB_SIZE, DATE_SEQ_LEN
from model import Generator, Discriminator, NOISE_DIM

DEVICE     = torch.device("cpu")
EPOCHS     = 30
BATCH_SIZE = 256
LR         = 2e-4
SEED       = 42

torch.manual_seed(SEED)

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
LOGS_DIR    = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


def tokens_to_onehot(tokens: torch.Tensor) -> torch.Tensor:
    return F.one_hot(tokens, num_classes=VOCAB_SIZE).float()


def onehot_to_tokens(onehot: torch.Tensor) -> torch.Tensor:
    return onehot.argmax(dim=-1)


def train() -> None:
    train_loader, val_loader, test_loader = get_dataloaders(
        filepath=os.path.join(os.path.dirname(__file__), "../../data/data.txt"),
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

    criterion = nn.BCELoss()

    g_losses, d_losses = [], []
    best_all_pass = -1.0

    for epoch in range(1, EPOCHS + 1):
        G.train()
        D.train()
        total_g = total_d = 0.0

        for cond, date_tokens in train_loader:
            cond        = cond.to(DEVICE)
            date_tokens = date_tokens.to(DEVICE)
            B           = cond.size(0)

            real_onehot = tokens_to_onehot(date_tokens)
            real_labels = torch.ones(B, 1, device=DEVICE)
            fake_labels = torch.zeros(B, 1, device=DEVICE)

            noise     = torch.randn(B, NOISE_DIM, device=DEVICE)
            fake_out  = G(noise, cond)
            fake_soft = F.softmax(fake_out, dim=-1)

            d_real = D(real_onehot, cond)
            d_fake = D(fake_soft.detach(), cond)
            loss_D = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            noise    = torch.randn(B, NOISE_DIM, device=DEVICE)
            fake_out = G(noise, cond)
            fake_soft = F.softmax(fake_out, dim=-1)
            d_fake_g  = D(fake_soft, cond)
            loss_G    = criterion(d_fake_g, real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            total_g += loss_G.item()
            total_d += loss_D.item()

        avg_g = total_g / len(train_loader)
        avg_d = total_d / len(train_loader)
        g_losses.append(avg_g)
        d_losses.append(avg_d)

        print(f"Epoch {epoch:02d}/{EPOCHS} | G Loss: {avg_g:.4f} | D Loss: {avg_d:.4f}")

        if epoch % 5 == 0:
            G.eval()
            all_preds, all_conds = [], []
            tok = DateTokenizer()

            with torch.no_grad():
                for cond_b, _ in test_loader:
                    cond_b = cond_b.to(DEVICE)
                    noise  = torch.randn(cond_b.size(0), NOISE_DIM, device=DEVICE)
                    out    = G(noise, cond_b)
                    tokens = onehot_to_tokens(F.softmax(out, dim=-1))
                    for i in range(tokens.size(0)):
                        all_preds.append(tok.decode_date(tokens[i].tolist()))
                    all_conds.extend([tuple(c.tolist()) for c in cond_b])

            metrics  = condition_accuracy(all_preds, all_conds)
            all_pass = metrics["all_pass"]
            print(f"  all_pass: {all_pass:.2f}%")

            if all_pass > best_all_pass:
                best_all_pass = all_pass
                torch.save(G.state_dict(), os.path.join(WEIGHTS_DIR, "best_G.pt"))
                torch.save(D.state_dict(), os.path.join(WEIGHTS_DIR, "best_D.pt"))
                print(f"  ✓ Best model saved")

    plot_losses(g_losses, d_losses, "GAN Loss", os.path.join(LOGS_DIR, "loss.png"))

    G.load_state_dict(torch.load(os.path.join(WEIGHTS_DIR, "best_G.pt")))
    G.eval()

    tok = DateTokenizer()
    all_preds, all_conds = [], []

    with torch.no_grad():
        for cond_b, _ in test_loader:
            cond_b = cond_b.to(DEVICE)
            noise  = torch.randn(cond_b.size(0), NOISE_DIM, device=DEVICE)
            out    = G(noise, cond_b)
            tokens = onehot_to_tokens(F.softmax(out, dim=-1))
            for i in range(tokens.size(0)):
                all_preds.append(tok.decode_date(tokens[i].tolist()))
            all_conds.extend([tuple(c.tolist()) for c in cond_b])

    metrics = condition_accuracy(all_preds, all_conds)

    print("\n── GAN Test Metrics ──")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.2f}%")

    save_metrics(metrics, os.path.join(LOGS_DIR, "test_metrics.txt"))

    with open(os.path.join(LOGS_DIR, "sample_predictions.txt"), "w") as f:
        for i in range(min(20, len(all_preds))):
            f.write(f"{all_preds[i]}\n")

    print(f"\nLogs saved to {LOGS_DIR}")


if __name__ == "__main__":
    train()