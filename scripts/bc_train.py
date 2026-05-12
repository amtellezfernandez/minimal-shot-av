"""Train BC agents: Token-BC (discrete) and Continuous-BC (regression).

Both use the same 10-feature geometric state and the same MLP depth.
The only difference is the output head and loss function.

Token-BC   (Agent B): cross-entropy over 9 tokens — same vocabulary as Spotlight Reflex
Continuous-BC (Agent A): MSE regression over (speed_scale, lat_offset_m) — raw continuous

Architecture: 3-layer MLP, 256 hidden units, GELU, dropout 0.15, LayerNorm.
This is deliberately over-parameterised for the input size to give learning every chance.

Output:
  artifacts/bc_models/token_bc.pt
  artifacts/bc_models/continuous_bc.pt
  artifacts/bc_models/training_log.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_DIR = ROOT / "artifacts" / "bc_data"
OUT_DIR  = ROOT / "artifacts" / "bc_models"

N_FEATURES = 10
N_TOKENS   = 9
HIDDEN     = 256
DROPOUT    = 0.15
LR         = 3e-4
EPOCHS     = 80
BATCH      = 512
VAL_FRAC   = 0.15
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


class GeomMLP(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(N_FEATURES),
            nn.Linear(N_FEATURES, HIDDEN), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN),     nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN // 2), nn.GELU(),
            nn.Linear(HIDDEN // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _load_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features    = np.load(DATA_DIR / "features.npy")
    token_idx   = np.load(DATA_DIR / "token_idx.npy")
    speed_scale = np.load(DATA_DIR / "speed_scale.npy")
    lat_offset  = np.load(DATA_DIR / "lat_offset.npy")
    feat_mean   = np.load(DATA_DIR / "feat_mean.npy")
    feat_std    = np.load(DATA_DIR / "feat_std.npy")
    return features, token_idx, speed_scale, lat_offset, feat_mean, feat_std


def train_token_bc(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> tuple[GeomMLP, list[dict]]:
    dataset = TensorDataset(features, labels)
    n_val   = int(len(dataset) * VAL_FRAC)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    model = GeomMLP(N_TOKENS).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    w = class_weights.to(DEVICE) if class_weights is not None else None

    log = []
    print(f"  Token-BC  — {n_train:,} train, {n_val:,} val, device={DEVICE}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = F.cross_entropy(model(xb), yb, weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0; correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                val_loss += F.cross_entropy(logits, yb, weight=w).item() * len(xb)
                correct  += (logits.argmax(1) == yb).sum().item()
        val_loss /= n_val
        val_acc   = correct / n_val

        sched.step()
        entry = {"epoch": epoch, "train_loss": round(train_loss, 4),
                 "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4)}
        log.append(entry)
        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.3f}")

    return model, log


def train_continuous_bc(
    features: torch.Tensor,
    targets: torch.Tensor,     # (speed_scale, lat_offset_m) normalised
    target_std: torch.Tensor,
) -> tuple[GeomMLP, list[dict]]:
    dataset = TensorDataset(features, targets)
    n_val   = int(len(dataset) * VAL_FRAC)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    model = GeomMLP(2).to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    log = []
    print(f"  Continuous-BC — {n_train:,} train, {n_val:,} val, device={DEVICE}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += F.mse_loss(model(xb), yb).item() * len(xb)
        val_loss /= n_val
        sched.step()

        entry = {"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5)}
        log.append(entry)
        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}  train={train_loss:.5f}  val={val_loss:.5f}")

    return model, log


def main() -> None:
    print("Loading data…")
    features, token_idx, speed_scale, lat_offset, feat_mean, feat_std = _load_data()
    n = len(features)
    print(f"  {n:,} transitions, {N_FEATURES} features, {N_TOKENS} tokens")

    # Normalise features
    feat_t = torch.from_numpy((features - feat_mean) / feat_std).float()

    # Continuous targets: normalise each output independently
    cont_raw = np.stack([speed_scale, lat_offset], axis=1).astype(np.float32)
    cont_mean = cont_raw.mean(0)
    cont_std  = cont_raw.std(0) + 1e-6
    cont_t    = torch.from_numpy((cont_raw - cont_mean) / cont_std).float()

    tok_t = torch.from_numpy(token_idx).long()

    # Inverse-frequency class weights so rare tokens (stop, evasive, crawl) are not overwhelmed
    counts = np.bincount(token_idx, minlength=N_TOKENS).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)  # avoid div-by-zero for absent tokens
    class_weights = torch.from_numpy(1.0 / counts * counts.mean()).float()
    print(f"  Class weights: {class_weights.numpy().round(3)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nTraining Token-BC (Agent B)…")
    tok_model, tok_log = train_token_bc(feat_t, tok_t, class_weights)
    torch.save({
        "state_dict": tok_model.state_dict(),
        "feat_mean": feat_mean.tolist(), "feat_std": feat_std.tolist(),
        "n_features": N_FEATURES, "n_tokens": N_TOKENS,
    }, OUT_DIR / "token_bc.pt")

    print("\nTraining Continuous-BC (Agent A)…")
    cont_model, cont_log = train_continuous_bc(feat_t, cont_t, torch.from_numpy(cont_std).float())
    torch.save({
        "state_dict": cont_model.state_dict(),
        "feat_mean": feat_mean.tolist(), "feat_std": feat_std.tolist(),
        "cont_mean": cont_mean.tolist(), "cont_std": cont_std.tolist(),
        "n_features": N_FEATURES,
    }, OUT_DIR / "continuous_bc.pt")

    training_log = {"token_bc": tok_log, "continuous_bc": cont_log}
    (OUT_DIR / "training_log.json").write_text(json.dumps(training_log, indent=2))
    print(f"\nModels saved to {OUT_DIR}")
    print(f"  token_bc final val_acc: {tok_log[-1]['val_acc']:.3f}")
    print(f"  continuous_bc final val_loss: {cont_log[-1]['val_loss']:.5f}")


if __name__ == "__main__":
    main()
