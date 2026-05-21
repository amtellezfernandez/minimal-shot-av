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
  artifacts/bc_models/token_rnn_bc.pt
  artifacts/bc_models/token_transformer_bc.pt
  artifacts/bc_models/token_dagger_bc.pt   (if artifacts/bc_dagger_data exists)
  artifacts/bc_models/training_log.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DATA_DIR = ROOT / "artifacts" / "bc_data"
DAGGER_DATA_DIR = ROOT / "artifacts" / "bc_dagger_data"
OUT_DIR  = ROOT / "artifacts" / "bc_models"

N_FEATURES = 10
TOKEN_ORDER = (
    "stop",
    "crawl",
    "maintain",
    "slow_yield",
    "nudge_left",
    "nudge_right",
    "evasive_left",
    "evasive_right",
    "lane_recover",
)
N_TOKENS   = len(TOKEN_ORDER)
HIDDEN     = 256
RNN_HIDDEN = 160
TRANSFORMER_HIDDEN = 192
TRANSFORMER_HEADS = 4
TRANSFORMER_LAYERS = 2
HISTORY_LEN = 8
DROPOUT    = 0.15
LR         = 3e-4
EPOCHS     = 80
RNN_EPOCHS = 60
DAGGER_EPOCHS = 60
BATCH      = 512
VAL_FRAC   = 0.15
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument(
        "--dagger-data-dir",
        type=Path,
        action="append",
        default=None,
        help="Repeat to merge one or more DAgger relabel datasets into Token-DAgger training.",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dagger-model-name", type=str, default="token_dagger_bc.pt")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--train-mode", choices=("all", "dagger-only"), default="all")
    parser.add_argument(
        "--class-weight-mode",
        choices=("inverse_freq", "none"),
        default="inverse_freq",
        help="Token loss weighting. Use 'none' to test natural-frequency BC/DAgger training.",
    )
    parser.add_argument(
        "--source-weights",
        type=str,
        default="",
        help=(
            "Comma-separated DAgger source weights, e.g. '0:1,1:1,2:0.5,3:0.25'. "
            "Source 0 is expert BC data; DAgger pools are 1..N in --dagger-data-dir order."
        ),
    )
    return parser.parse_args()


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


class GeomGRUTokenBC(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_tokens: int = N_TOKENS,
        hidden: int = RNN_HIDDEN,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, n_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        _, h = self.gru(x)
        return self.head(h[-1])


class GeomTransformerTokenBC(nn.Module):
    def __init__(
        self,
        n_features: int = N_FEATURES,
        n_tokens: int = N_TOKENS,
        hidden: int = TRANSFORMER_HIDDEN,
        n_heads: int = TRANSFORMER_HEADS,
        n_layers: int = TRANSFORMER_LAYERS,
        history_len: int = HISTORY_LEN,
    ) -> None:
        super().__init__()
        self.history_len = history_len
        self.input_norm = nn.LayerNorm(n_features)
        self.input_proj = nn.Linear(n_features, hidden)
        self.pos_embed = nn.Parameter(torch.zeros(1, history_len, hidden))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 2,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden, n_tokens),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.history_len:
            raise ValueError(f"expected history length {self.history_len}, got {x.shape[1]}")
        x = self.input_norm(x)
        x = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        x = self.encoder(x)
        return self.head(x[:, -1, :])


def _load_data(data_dir: Path) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    features    = np.load(data_dir / "features.npy")
    token_idx   = np.load(data_dir / "token_idx.npy")
    _validate_token_contract(data_dir, token_idx)
    speed_scale = np.load(data_dir / "speed_scale.npy")
    lat_offset  = np.load(data_dir / "lat_offset.npy")
    feat_mean   = np.load(data_dir / "feat_mean.npy")
    feat_std    = np.load(data_dir / "feat_std.npy")
    rollout_id  = _load_sequence_array(data_dir / "rollout_id.npy", length=len(features))
    step_idx    = _load_sequence_array(
        data_dir / "step_idx.npy",
        length=len(features),
        default=np.arange(len(features)),
    )
    return features, token_idx, speed_scale, lat_offset, feat_mean, feat_std, rollout_id, step_idx


def _validate_token_contract(data_dir: Path, token_idx: np.ndarray) -> None:
    if token_idx.size:
        min_idx = int(token_idx.min())
        max_idx = int(token_idx.max())
        if min_idx < 0 or max_idx >= N_TOKENS:
            raise ValueError(f"{data_dir} token_idx contains out-of-range ids [{min_idx}, {max_idx}]")
    meta_path = data_dir / "meta.json"
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_names = meta.get("token_names")
    if token_names is None:
        return
    if list(token_names) != list(TOKEN_ORDER):
        raise ValueError(
            f"{data_dir} token_names do not match training contract: "
            f"expected {list(TOKEN_ORDER)}, got {list(token_names)}"
        )


def _load_sequence_array(path: Path, *, length: int, default: np.ndarray | None = None) -> np.ndarray:
    if path.is_file():
        return np.load(path).astype(np.int64)
    if default is not None:
        return default.astype(np.int64)
    return np.zeros(length, dtype=np.int64)


def build_history_windows(
    features: np.ndarray,
    rollout_id: np.ndarray,
    *,
    history_len: int = HISTORY_LEN,
) -> np.ndarray:
    """Left-pad per-rollout feature histories for recurrent BC."""
    if features.ndim != 2:
        raise ValueError(f"features must be rank-2, got shape {features.shape}")
    if rollout_id.shape[0] != features.shape[0]:
        raise ValueError("rollout_id length must match feature rows")
    if history_len <= 0:
        raise ValueError("history_len must be positive")

    windows = np.empty((len(features), history_len, features.shape[1]), dtype=np.float32)
    start = 0
    n = len(features)
    while start < n:
        rid = rollout_id[start]
        end = start + 1
        while end < n and rollout_id[end] == rid:
            end += 1
        first = features[start]
        for row in range(start, end):
            history_start = max(start, row - history_len + 1)
            history = features[history_start : row + 1]
            pad_count = history_len - len(history)
            if pad_count:
                windows[row, :pad_count] = first
                windows[row, pad_count:] = history
            else:
                windows[row] = history
        start = end
    return windows


def train_token_bc(
    features: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    *,
    source_ids: torch.Tensor | None = None,
    source_weights: dict[int, float] | None = None,
    epochs: int = EPOCHS,
    label: str = "Token-BC",
    device: str = DEVICE,
) -> tuple[GeomMLP, list[dict]]:
    if source_ids is not None and len(source_ids) != len(labels):
        raise ValueError("source_ids length must match labels")
    if source_ids is None:
        dataset = TensorDataset(features, labels)
    else:
        dataset = TensorDataset(features, labels, source_ids)
    n_val   = int(len(dataset) * VAL_FRAC)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    model = GeomMLP(N_TOKENS).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = class_weights.to(device) if class_weights is not None else None
    source_weight_tensor: torch.Tensor | None = None
    if source_weights:
        max_source = max(max(source_weights), int(source_ids.max().item()) if source_ids is not None else 0)
        weights = torch.ones(max_source + 1, dtype=torch.float32)
        for source_id, weight in source_weights.items():
            if source_id < 0:
                raise ValueError("source weight ids must be non-negative")
            if weight < 0.0:
                raise ValueError("source weights must be non-negative")
            weights[source_id] = float(weight)
        source_weight_tensor = weights.to(device)

    log = []
    print(f"  {label}  — {n_train:,} train, {n_val:,} val, device={device}")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            if len(batch) == 2:
                xb, yb = batch
                sb = None
            else:
                xb, yb, sb = batch
            xb, yb = xb.to(device), yb.to(device)
            sb = sb.to(device) if sb is not None else None
            loss = _classification_loss(model(xb), yb, class_weights=w, source_ids=sb, source_weights=source_weight_tensor)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0; correct = 0
        with torch.no_grad():
            for batch in val_dl:
                if len(batch) == 2:
                    xb, yb = batch
                    sb = None
                else:
                    xb, yb, sb = batch
                xb, yb = xb.to(device), yb.to(device)
                sb = sb.to(device) if sb is not None else None
                logits = model(xb)
                val_loss += _classification_loss(
                    logits,
                    yb,
                    class_weights=w,
                    source_ids=sb,
                    source_weights=source_weight_tensor,
                ).item() * len(xb)
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


def _classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    source_ids: torch.Tensor | None = None,
    source_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    per_sample = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    if source_ids is not None and source_weights is not None:
        if int(source_ids.max().item()) >= len(source_weights):
            raise ValueError("source_ids contain an id without a configured source weight")
        per_sample = per_sample * source_weights[source_ids]
    return per_sample.mean()


def train_token_rnn_bc(
    history_features: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    *,
    device: str = DEVICE,
) -> tuple[GeomGRUTokenBC, list[dict]]:
    dataset = TensorDataset(history_features, labels)
    n_val = int(len(dataset) * VAL_FRAC)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    model = GeomGRUTokenBC().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=RNN_EPOCHS)
    w = class_weights.to(device) if class_weights is not None else None

    log = []
    print(f"  Token-RNN-BC — {n_train:,} train, {n_val:,} val, device={device}, history={HISTORY_LEN}")
    for epoch in range(1, RNN_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb, weight=w)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0; correct = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += F.cross_entropy(logits, yb, weight=w).item() * len(xb)
                correct += (logits.argmax(1) == yb).sum().item()
        val_loss /= n_val
        val_acc = correct / n_val
        sched.step()

        entry = {"epoch": epoch, "train_loss": round(train_loss, 4),
                 "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4)}
        log.append(entry)
        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.3f}")

    return model, log


def train_token_transformer_bc(
    history_features: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
    *,
    source_ids: torch.Tensor | None = None,
    source_weights: dict[int, float] | None = None,
    epochs: int = RNN_EPOCHS,
    label: str = "Token-Transformer-BC",
    device: str = DEVICE,
) -> tuple[GeomTransformerTokenBC, list[dict]]:
    if source_ids is not None and len(source_ids) != len(labels):
        raise ValueError("source_ids length must match labels")
    if source_ids is None:
        dataset = TensorDataset(history_features, labels)
    else:
        dataset = TensorDataset(history_features, labels, source_ids)
    n_val = int(len(dataset) * VAL_FRAC)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    model = GeomTransformerTokenBC(history_len=history_features.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = class_weights.to(device) if class_weights is not None else None
    source_weight_tensor: torch.Tensor | None = None
    if source_weights:
        max_source = max(max(source_weights), int(source_ids.max().item()) if source_ids is not None else 0)
        weights = torch.ones(max_source + 1, dtype=torch.float32)
        for source_id, weight in source_weights.items():
            if source_id < 0:
                raise ValueError("source weight ids must be non-negative")
            if weight < 0.0:
                raise ValueError("source weights must be non-negative")
            weights[source_id] = float(weight)
        source_weight_tensor = weights.to(device)

    log = []
    print(f"  {label} — {n_train:,} train, {n_val:,} val, device={device}, history={history_features.shape[1]}")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            if len(batch) == 2:
                xb, yb = batch
                sb = None
            else:
                xb, yb, sb = batch
            xb, yb = xb.to(device), yb.to(device)
            sb = sb.to(device) if sb is not None else None
            loss = _classification_loss(model(xb), yb, class_weights=w, source_ids=sb, source_weights=source_weight_tensor)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for batch in val_dl:
                if len(batch) == 2:
                    xb, yb = batch
                    sb = None
                else:
                    xb, yb, sb = batch
                xb, yb = xb.to(device), yb.to(device)
                sb = sb.to(device) if sb is not None else None
                logits = model(xb)
                val_loss += _classification_loss(
                    logits,
                    yb,
                    class_weights=w,
                    source_ids=sb,
                    source_weights=source_weight_tensor,
                ).item() * len(xb)
                correct += (logits.argmax(1) == yb).sum().item()
        val_loss /= n_val
        val_acc = correct / n_val
        sched.step()

        entry = {"epoch": epoch, "train_loss": round(train_loss, 4), "val_loss": round(val_loss, 4), "val_acc": round(val_acc, 4)}
        log.append(entry)
        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.3f}")

    return model, log


def train_continuous_bc(
    features: torch.Tensor,
    targets: torch.Tensor,     # (speed_scale, lat_offset_m) normalised
    target_std: torch.Tensor,
    *,
    device: str = DEVICE,
) -> tuple[GeomMLP, list[dict]]:
    dataset = TensorDataset(features, targets)
    n_val   = int(len(dataset) * VAL_FRAC)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    model = GeomMLP(2).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    log = []
    print(f"  Continuous-BC — {n_train:,} train, {n_val:,} val, device={device}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += F.mse_loss(model(xb), yb).item() * len(xb)
        val_loss /= n_val
        sched.step()

        entry = {"epoch": epoch, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5)}
        log.append(entry)
        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}  train={train_loss:.5f}  val={val_loss:.5f}")

    return model, log


def _token_checkpoint_payload(
    *,
    model: nn.Module,
    feat_mean: np.ndarray,
    feat_std: np.ndarray,
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "state_dict": model.state_dict(),
        "feat_mean": feat_mean.tolist(),
        "feat_std": feat_std.tolist(),
        "n_features": N_FEATURES,
        "n_tokens": N_TOKENS,
        "token_names": list(TOKEN_ORDER),
    }
    payload.update(extra)
    return payload


def _load_dagger_merge(
    base_features: np.ndarray,
    base_token_idx: np.ndarray,
    base_rollout_id: np.ndarray,
    dagger_dirs: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, int | str]]]:
    merged_features = [base_features]
    merged_token_idx = [base_token_idx]
    merged_source_ids = [np.zeros(len(base_features), dtype=np.int64)]
    merged_rollout_ids = [base_rollout_id.astype(np.int64)]
    summaries: list[dict[str, int | str]] = []
    rollout_offset = int(base_rollout_id.max()) + 1 if len(base_rollout_id) else 0
    for source_id, dagger_dir in enumerate(dagger_dirs, start=1):
        dagger_features = np.load(dagger_dir / "features.npy").astype(np.float32)
        dagger_token_idx = np.load(dagger_dir / "token_idx.npy").astype(np.int64)
        _validate_token_contract(dagger_dir, dagger_token_idx)
        dagger_rollout_id = _load_sequence_array(
            dagger_dir / "rollout_id.npy",
            length=len(dagger_features),
            default=np.arange(len(dagger_features)),
        )
        dagger_rollout_id = dagger_rollout_id.astype(np.int64) + rollout_offset
        rollout_offset = int(dagger_rollout_id.max()) + 1 if len(dagger_rollout_id) else rollout_offset
        source_policy = dagger_dir.name
        meta_path = dagger_dir / "meta.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            source_policy = str(meta.get("source_policy", source_policy))
        summaries.append(
            {
                "data_dir": str(dagger_dir),
                "source_policy": source_policy,
                "source_id": source_id,
                "transitions": int(len(dagger_features)),
            }
        )
        merged_features.append(dagger_features)
        merged_token_idx.append(dagger_token_idx)
        merged_source_ids.append(np.full(len(dagger_features), source_id, dtype=np.int64))
        merged_rollout_ids.append(dagger_rollout_id)
    return (
        np.concatenate(merged_features, axis=0).astype(np.float32),
        np.concatenate(merged_token_idx, axis=0).astype(np.int64),
        np.concatenate(merged_source_ids, axis=0).astype(np.int64),
        np.concatenate(merged_rollout_ids, axis=0).astype(np.int64),
        summaries,
    )


def _parse_source_weights(spec: str) -> dict[int, float]:
    if not spec.strip():
        return {}
    weights: dict[int, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"invalid source weight {item!r}; expected id:weight")
        source_id_str, weight_str = item.split(":", maxsplit=1)
        source_id = int(source_id_str)
        weight = float(weight_str)
        if source_id < 0:
            raise ValueError("source ids must be non-negative")
        if weight < 0.0:
            raise ValueError("source weights must be non-negative")
        weights[source_id] = weight
    return weights


def _inverse_frequency_weights(token_idx: np.ndarray, *, enabled: bool) -> torch.Tensor | None:
    if not enabled:
        return None
    counts = np.bincount(token_idx, minlength=N_TOKENS).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)
    return torch.from_numpy(1.0 / counts * counts.mean()).float()


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for BC training, but torch.cuda.is_available() is False")
    device = "cuda" if args.device in {"cuda", "auto"} and torch.cuda.is_available() else "cpu"
    source_weights = _parse_source_weights(args.source_weights)
    print("Loading data…")
    features, token_idx, speed_scale, lat_offset, feat_mean, feat_std, rollout_id, step_idx = _load_data(args.data_dir)
    del step_idx
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

    class_weights = _inverse_frequency_weights(token_idx, enabled=args.class_weight_mode == "inverse_freq")
    if class_weights is None:
        print("  Class weights: disabled")
    else:
        print(f"  Class weights: {class_weights.numpy().round(3)}")
    if source_weights:
        print(f"  Source weights: {source_weights}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tok_log: list[dict] | None = None
    cont_log: list[dict] | None = None
    rnn_log: list[dict] | None = None
    transformer_log: list[dict] | None = None

    if args.train_mode == "all":
        print("\nTraining Token-BC (Agent B)…")
        tok_model, tok_log = train_token_bc(feat_t, tok_t, class_weights, device=device)
        torch.save(
            _token_checkpoint_payload(model=tok_model, feat_mean=feat_mean, feat_std=feat_std),
            args.out_dir / "token_bc.pt",
        )

        print("\nTraining Continuous-BC (Agent A)…")
        cont_model, cont_log = train_continuous_bc(feat_t, cont_t, torch.from_numpy(cont_std).float(), device=device)
        torch.save({
            "state_dict": cont_model.state_dict(),
            "feat_mean": feat_mean.tolist(), "feat_std": feat_std.tolist(),
            "cont_mean": cont_mean.tolist(), "cont_std": cont_std.tolist(),
            "n_features": N_FEATURES,
        }, args.out_dir / "continuous_bc.pt")

        print("\nTraining Token-RNN-BC (history baseline)…")
        hist_np = build_history_windows((features - feat_mean) / feat_std, rollout_id, history_len=HISTORY_LEN)
        hist_t = torch.from_numpy(hist_np).float()
        rnn_model, rnn_log = train_token_rnn_bc(hist_t, tok_t, class_weights, device=device)
        torch.save(
            _token_checkpoint_payload(
                model=rnn_model,
                feat_mean=feat_mean,
                feat_std=feat_std,
                history_len=HISTORY_LEN,
                rnn_hidden=RNN_HIDDEN,
            ),
            args.out_dir / "token_rnn_bc.pt",
        )

        print("\nTraining Token-Transformer-BC (history baseline)…")
        transformer_model, transformer_log = train_token_transformer_bc(hist_t, tok_t, class_weights, device=device)
        torch.save(
            _token_checkpoint_payload(
                model=transformer_model,
                feat_mean=feat_mean,
                feat_std=feat_std,
                history_len=HISTORY_LEN,
                transformer_hidden=TRANSFORMER_HIDDEN,
                transformer_heads=TRANSFORMER_HEADS,
                transformer_layers=TRANSFORMER_LAYERS,
            ),
            args.out_dir / "token_transformer_bc.pt",
        )

    dagger_log: list[dict] | None = None
    dagger_transformer_log: list[dict] | None = None
    dagger_dirs = args.dagger_data_dir or ([DAGGER_DATA_DIR] if (DAGGER_DATA_DIR / "features.npy").is_file() else [])
    if dagger_dirs:
        print(f"\nTraining Token-DAgger-BC from {len(dagger_dirs)} relabel pool(s)…")
        merged_features, merged_token_idx, merged_source_ids, merged_rollout_id, dagger_sources = _load_dagger_merge(
            features,
            token_idx,
            rollout_id,
            dagger_dirs,
        )
        dagger_feat_mean = merged_features.mean(axis=0)
        dagger_feat_std = merged_features.std(axis=0) + 1e-6
        merged_feat_t = torch.from_numpy((merged_features - dagger_feat_mean) / dagger_feat_std).float()
        merged_tok_t = torch.from_numpy(merged_token_idx).long()
        merged_source_t = torch.from_numpy(merged_source_ids).long()
        merged_class_weights = _inverse_frequency_weights(
            merged_token_idx,
            enabled=args.class_weight_mode == "inverse_freq",
        )
        dagger_model, dagger_log = train_token_bc(
            merged_feat_t,
            merged_tok_t,
            merged_class_weights,
            source_ids=merged_source_t,
            source_weights=source_weights,
            epochs=DAGGER_EPOCHS,
            label="Token-DAgger-BC",
            device=device,
        )
        torch.save(
            _token_checkpoint_payload(
                model=dagger_model,
                feat_mean=dagger_feat_mean,
                feat_std=dagger_feat_std,
                base_transitions=int(len(features)),
                dagger_transitions=int(len(merged_token_idx) - len(features)),
                dagger_source_count=len(dagger_dirs),
                dagger_sources=dagger_sources,
                class_weight_mode=args.class_weight_mode,
                source_weights={str(key): value for key, value in sorted(source_weights.items())},
            ),
            args.out_dir / args.dagger_model_name,
        )

        merged_hist_np = build_history_windows(
            (merged_features - dagger_feat_mean) / dagger_feat_std,
            merged_rollout_id,
            history_len=HISTORY_LEN,
        )
        merged_hist_t = torch.from_numpy(merged_hist_np).float()
        transformer_dagger_model, dagger_transformer_log = train_token_transformer_bc(
            merged_hist_t,
            merged_tok_t,
            merged_class_weights,
            source_ids=merged_source_t,
            source_weights=source_weights,
            epochs=DAGGER_EPOCHS,
            label="Token-Transformer-DAgger-BC",
            device=device,
        )
        torch.save(
            _token_checkpoint_payload(
                model=transformer_dagger_model,
                feat_mean=dagger_feat_mean,
                feat_std=dagger_feat_std,
                history_len=HISTORY_LEN,
                transformer_hidden=TRANSFORMER_HIDDEN,
                transformer_heads=TRANSFORMER_HEADS,
                transformer_layers=TRANSFORMER_LAYERS,
                base_transitions=int(len(features)),
                dagger_transitions=int(len(merged_token_idx) - len(features)),
                dagger_source_count=len(dagger_dirs),
                dagger_sources=dagger_sources,
                class_weight_mode=args.class_weight_mode,
                source_weights={str(key): value for key, value in sorted(source_weights.items())},
            ),
            args.out_dir / "token_transformer_dagger_bc.pt",
        )
    else:
        print(f"\nSkipping Token-DAgger-BC: {DAGGER_DATA_DIR / 'features.npy'} not found")

    training_log = {}
    training_log["config"] = {
        "class_weight_mode": args.class_weight_mode,
        "source_weights": {str(key): value for key, value in sorted(source_weights.items())},
        "token_names": list(TOKEN_ORDER),
    }
    if tok_log is not None:
        training_log["token_bc"] = tok_log
    if cont_log is not None:
        training_log["continuous_bc"] = cont_log
    if rnn_log is not None:
        training_log["token_rnn_bc"] = rnn_log
    if transformer_log is not None:
        training_log["token_transformer_bc"] = transformer_log
    if dagger_log is not None:
        training_log["token_dagger_bc"] = dagger_log
    if dagger_transformer_log is not None:
        training_log["token_transformer_dagger_bc"] = dagger_transformer_log
    (args.out_dir / "training_log.json").write_text(json.dumps(training_log, indent=2))
    print(f"\nModels saved to {args.out_dir}")
    if tok_log is not None:
        print(f"  token_bc final val_acc: {tok_log[-1]['val_acc']:.3f}")
    if rnn_log is not None:
        print(f"  token_rnn_bc final val_acc: {rnn_log[-1]['val_acc']:.3f}")
    if transformer_log is not None:
        print(f"  token_transformer_bc final val_acc: {transformer_log[-1]['val_acc']:.3f}")
    if dagger_log is not None:
        print(f"  token_dagger_bc final val_acc: {dagger_log[-1]['val_acc']:.3f}")
    if dagger_transformer_log is not None:
        print(f"  token_transformer_dagger_bc final val_acc: {dagger_transformer_log[-1]['val_acc']:.3f}")
    if cont_log is not None:
        print(f"  continuous_bc final val_loss: {cont_log[-1]['val_loss']:.5f}")


if __name__ == "__main__":
    main()
