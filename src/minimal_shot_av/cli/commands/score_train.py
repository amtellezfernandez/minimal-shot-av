"""Train a trajectory-informed maneuver scorer f(state, trajectory, interaction) -> utility.

The scorer consumes the 10-D geometric state plus per-candidate future trajectories
encoded as relative (x, y, heading) waypoints. It also computes explicit geometric
invariants such as segment lengths, heading deltas, curvature, lateral acceleration,
and jerk-like finite differences to bake in a physical inductive bias.

Output:
  artifacts/score_models/trajectory_scorer.pt
  artifacts/score_models/training_log.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

N_FEATURES = 10
N_TOKENS = 9
STATE_HIDDEN = 64
TRAJ_HIDDEN = 96
FUSION_HIDDEN = 96
INTERACTION_HIDDEN = 48
DROPOUT = 0.10
LR = 3e-4
EPOCHS = 50
BATCH = 256
VAL_FRAC = 0.15
WEIGHT_DECAY = 1e-4


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        default=None,
        help="One or more scorer data directories. Repeat the flag to merge datasets.",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "score_models")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    return parser.parse_args()


def _load_array(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path)


def _load_meta(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return payload


def load_score_datasets(
    data_dirs: list[Path],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not data_dirs:
        raise ValueError("at least one --data-dir is required")

    metas: list[dict[str, Any]] = []
    state_features_parts: list[np.ndarray] = []
    candidate_traj_parts: list[np.ndarray] = []
    candidate_interaction_parts: list[np.ndarray] = []
    expert_idx_parts: list[np.ndarray] = []
    rollout_id_parts: list[np.ndarray] = []
    feat_mean_parts: list[np.ndarray] = []
    feat_std_parts: list[np.ndarray] = []
    rollout_offset = 0
    reference_traj_shape: tuple[int, ...] | None = None
    reference_interaction_shape: tuple[int, ...] | None = None
    reference_horizon_s: float | None = None

    for data_dir in data_dirs:
        meta = _load_meta(data_dir / "meta.json")
        state_features = _load_array(data_dir / "state_features.npy").astype(np.float32)
        candidate_traj = _load_array(data_dir / "candidate_traj.npy").astype(np.float32)
        interaction_path = data_dir / "candidate_interaction.npy"
        if interaction_path.is_file():
            candidate_interaction = _load_array(interaction_path).astype(np.float32)
        else:
            candidate_interaction = np.zeros((candidate_traj.shape[0], candidate_traj.shape[1], 0), dtype=np.float32)
        expert_idx = _load_array(data_dir / "expert_idx.npy").astype(np.int64)
        rollout_id = _load_array(data_dir / "rollout_id.npy").astype(np.int64)
        feat_mean = _load_array(data_dir / "feat_mean.npy").astype(np.float32)
        feat_std = _load_array(data_dir / "feat_std.npy").astype(np.float32)

        if reference_traj_shape is None:
            reference_traj_shape = tuple(candidate_traj.shape[1:])
            reference_interaction_shape = tuple(candidate_interaction.shape[1:])
            reference_horizon_s = float(meta.get("trajectory_horizon_seconds", 2.0))
        else:
            if tuple(candidate_traj.shape[1:]) != reference_traj_shape:
                raise ValueError(f"trajectory shape mismatch for {data_dir}: {candidate_traj.shape[1:]} vs {reference_traj_shape}")
            if tuple(candidate_interaction.shape[1:]) != reference_interaction_shape:
                raise ValueError(
                    f"interaction shape mismatch for {data_dir}: {candidate_interaction.shape[1:]} vs {reference_interaction_shape}"
                )
            if abs(float(meta.get("trajectory_horizon_seconds", 2.0)) - float(reference_horizon_s)) > 1e-6:
                raise ValueError(f"trajectory horizon mismatch for {data_dir}")

        metas.append(meta)
        state_features_parts.append(state_features)
        candidate_traj_parts.append(candidate_traj)
        candidate_interaction_parts.append(candidate_interaction)
        expert_idx_parts.append(expert_idx)
        rollout_id_parts.append(rollout_id + rollout_offset)
        feat_mean_parts.append(feat_mean)
        feat_std_parts.append(feat_std)
        rollout_offset += int(rollout_id.max()) + 1 if rollout_id.size else 0

    merged_state = np.concatenate(state_features_parts, axis=0)
    merged_traj = np.concatenate(candidate_traj_parts, axis=0)
    merged_interaction = np.concatenate(candidate_interaction_parts, axis=0)
    merged_expert_idx = np.concatenate(expert_idx_parts, axis=0)
    merged_rollout_id = np.concatenate(rollout_id_parts, axis=0)
    merged_feat_mean = merged_state.mean(axis=0).astype(np.float32)
    merged_feat_std = (merged_state.std(axis=0) + 1e-6).astype(np.float32)

    merged_meta = {
        "schema": "score_data_merged_v1" if len(metas) > 1 else metas[0]["schema"],
        "source_count": len(metas),
        "sources": [
            {
                "data_dir": str(data_dir),
                "meta": meta,
            }
            for data_dir, meta in zip(data_dirs, metas)
        ],
        "trajectory_point_horizon": metas[0]["trajectory_point_horizon"],
        "trajectory_horizon_seconds": metas[0].get("trajectory_horizon_seconds", 2.0),
        "trajectory_channels": list(metas[0]["trajectory_channels"]),
        "interaction_shape": list(merged_interaction.shape[1:]),
        "interaction_feature_names": list(metas[0].get("interaction_feature_names", [])),
    }
    return merged_meta, merged_state, merged_traj, merged_interaction, merged_expert_idx, merged_rollout_id, merged_feat_mean, merged_feat_std


def split_rollouts(rollout_id: np.ndarray, *, val_frac: float = VAL_FRAC, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    unique = np.unique(rollout_id)
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_frac)))
    val_ids = set(int(v) for v in shuffled[:n_val])
    train_mask = np.array([int(rid) not in val_ids for rid in rollout_id], dtype=bool)
    val_mask = ~train_mask
    return train_mask, val_mask


def _wrap_angle_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


class KinematicTrajectoryEncoder(nn.Module):
    def __init__(self, point_horizon: int, traj_dim: int = 3) -> None:
        super().__init__()
        self.point_horizon = point_horizon
        self.traj_dim = traj_dim
        raw_dim = point_horizon * traj_dim
        segment_count = max(point_horizon - 1, 1)
        curvature_count = max(point_horizon - 2, 1)
        invariant_dim = (
            segment_count +      # ds
            segment_count +      # dtheta
            curvature_count +    # curvature
            curvature_count +    # lateral acceleration proxy
            max(curvature_count - 1, 1) +  # curvature delta
            max(segment_count - 1, 1) +    # longitudinal jerk proxy
            4                    # path summary features
        )
        self.net = nn.Sequential(
            nn.Linear(raw_dim + invariant_dim, TRAJ_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(TRAJ_HIDDEN, TRAJ_HIDDEN),
            nn.GELU(),
        )

    def forward(self, traj: torch.Tensor, speed_mps: torch.Tensor, horizon_seconds: float) -> torch.Tensor:
        invariants = self._compute_invariants(traj, speed_mps, horizon_seconds)
        flat_traj = traj.reshape(traj.shape[0], -1)
        x = torch.cat([flat_traj, invariants], dim=-1)
        return self.net(x)

    def _compute_invariants(self, traj: torch.Tensor, speed_mps: torch.Tensor, horizon_seconds: float) -> torch.Tensor:
        eps = 1e-6
        dx = traj[:, 1:, 0] - traj[:, :-1, 0]
        dy = traj[:, 1:, 1] - traj[:, :-1, 1]
        dtheta = _wrap_angle_torch(traj[:, 1:, 2] - traj[:, :-1, 2])
        ds = torch.sqrt(dx.pow(2) + dy.pow(2) + eps)

        if dtheta.shape[1] > 1:
            curvature = dtheta[:, 1:] / (ds[:, 1:] + eps)
        else:
            curvature = dtheta / (ds + eps)
        speed = speed_mps.unsqueeze(1)
        lateral_accel = speed.pow(2) * curvature

        if curvature.shape[1] > 1:
            curvature_delta = curvature[:, 1:] - curvature[:, :-1]
        else:
            curvature_delta = curvature

        dt = max(horizon_seconds / max(self.point_horizon, 1), 1e-3)
        segment_speed = ds / dt
        if segment_speed.shape[1] > 1:
            jerk_proxy = (segment_speed[:, 1:] - segment_speed[:, :-1]) / dt
        else:
            jerk_proxy = segment_speed / dt

        path_length = ds.sum(dim=1, keepdim=True)
        final_offset = torch.sqrt(traj[:, -1, 0].pow(2) + traj[:, -1, 1].pow(2) + eps).unsqueeze(1)
        final_heading = traj[:, -1, 2].unsqueeze(1)
        heading_span = _wrap_angle_torch(traj[:, -1, 2] - traj[:, 0, 2]).unsqueeze(1)

        return torch.cat(
            [
                ds,
                dtheta,
                curvature,
                lateral_accel,
                curvature_delta,
                jerk_proxy,
                path_length,
                final_offset,
                final_heading,
                heading_span,
            ],
            dim=1,
        )


class InteractionFeatureEncoder(nn.Module):
    def __init__(self, interaction_dim: int) -> None:
        super().__init__()
        self.interaction_dim = interaction_dim
        if interaction_dim <= 0:
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.LayerNorm(interaction_dim),
                nn.Linear(interaction_dim, INTERACTION_HIDDEN),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(INTERACTION_HIDDEN, INTERACTION_HIDDEN),
                nn.GELU(),
            )

    def forward(self, interaction: torch.Tensor) -> torch.Tensor:
        if self.interaction_dim <= 0 or self.net is None:
            return interaction.new_zeros((interaction.shape[0], 0))
        return self.net(interaction)


class TrajectoryScorer(nn.Module):
    def __init__(self, point_horizon: int, interaction_dim: int = 0) -> None:
        super().__init__()
        self.interaction_dim = interaction_dim
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(N_FEATURES),
            nn.Linear(N_FEATURES, STATE_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(STATE_HIDDEN, STATE_HIDDEN),
            nn.GELU(),
        )
        self.traj_encoder = KinematicTrajectoryEncoder(point_horizon=point_horizon)
        self.interaction_encoder = InteractionFeatureEncoder(interaction_dim)
        self.fusion = nn.Sequential(
            nn.Linear(STATE_HIDDEN + TRAJ_HIDDEN + (INTERACTION_HIDDEN if interaction_dim > 0 else 0), FUSION_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(FUSION_HIDDEN, 1),
        )
        self.log_temperature = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        state: torch.Tensor,
        trajectories: torch.Tensor,
        horizon_seconds: float,
        interaction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, n_tokens, _, _ = trajectories.shape
        state_enc = self.state_encoder(state).unsqueeze(1).expand(-1, n_tokens, -1)
        traj_flat = trajectories.reshape(batch * n_tokens, trajectories.shape[2], trajectories.shape[3])
        speed_mps = state[:, 6].unsqueeze(1).expand(-1, n_tokens).reshape(batch * n_tokens)
        traj_enc = self.traj_encoder(traj_flat, speed_mps, horizon_seconds).reshape(batch, n_tokens, -1)
        parts = [state_enc, traj_enc]
        if self.interaction_dim > 0:
            if interaction is None:
                interaction = trajectories.new_zeros((batch, n_tokens, self.interaction_dim))
            interaction_flat = interaction.reshape(batch * n_tokens, interaction.shape[2])
            interaction_enc = self.interaction_encoder(interaction_flat).reshape(batch, n_tokens, -1)
            parts.append(interaction_enc)
        fused = torch.cat(parts, dim=-1)
        scores = self.fusion(fused.reshape(batch * n_tokens, -1)).reshape(batch, n_tokens)
        temperature = self.log_temperature.exp().clamp(min=0.05, max=10.0)
        return scores / temperature


def train_model(
    model: TrajectoryScorer,
    train_dl: DataLoader,
    val_dl: DataLoader,
    *,
    epochs: int,
    horizon_seconds: float,
    device: str,
) -> list[dict[str, float]]:
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    log: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_n = 0
        for state, traj, interaction, target in train_dl:
            state = state.to(device)
            traj = traj.to(device)
            interaction = interaction.to(device)
            target = target.to(device)
            logits = model(state, traj, horizon_seconds, interaction)
            loss = F.cross_entropy(logits, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_n = len(state)
            train_loss += loss.item() * batch_n
            train_correct += (logits.argmax(dim=1) == target).sum().item()
            train_n += batch_n
        train_loss /= max(train_n, 1)
        train_acc = train_correct / max(train_n, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_n = 0
        with torch.no_grad():
            for state, traj, interaction, target in val_dl:
                state = state.to(device)
                traj = traj.to(device)
                interaction = interaction.to(device)
                target = target.to(device)
                logits = model(state, traj, horizon_seconds, interaction)
                loss = F.cross_entropy(logits, target)
                batch_n = len(state)
                val_loss += loss.item() * batch_n
                val_correct += (logits.argmax(dim=1) == target).sum().item()
                val_n += batch_n
        val_loss /= max(val_n, 1)
        val_acc = val_correct / max(val_n, 1)
        sched.step()

        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 5),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 5),
            "temperature": round(float(model.log_temperature.exp().detach().cpu()), 5),
        }
        log.append(entry)
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"    epoch {epoch:3d} train={train_loss:.5f} acc={train_acc:.3f} "
                f"val={val_loss:.5f} acc={val_acc:.3f} temp={entry['temperature']:.3f}"
            )
    return log


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for scorer training, but torch.cuda.is_available() is False")
    device = "cuda" if args.device in {"cuda", "auto"} and torch.cuda.is_available() else "cpu"
    data_dirs = args.data_dir or [ROOT / "artifacts" / "score_data_smoke"]
    meta, state_features, candidate_traj, candidate_interaction, expert_idx, rollout_id, feat_mean, feat_std = load_score_datasets(data_dirs)

    train_mask, val_mask = split_rollouts(rollout_id)
    point_horizon = int(meta["trajectory_point_horizon"])
    horizon_seconds = float(meta.get("trajectory_horizon_seconds", 2.0))
    interaction_dim = int(candidate_interaction.shape[2])

    norm_state = (state_features - feat_mean) / feat_std
    train_ds = TensorDataset(
        torch.from_numpy(norm_state[train_mask]).float(),
        torch.from_numpy(candidate_traj[train_mask]).float(),
        torch.from_numpy(candidate_interaction[train_mask]).float(),
        torch.from_numpy(expert_idx[train_mask]).long(),
    )
    val_ds = TensorDataset(
        torch.from_numpy(norm_state[val_mask]).float(),
        torch.from_numpy(candidate_traj[val_mask]).float(),
        torch.from_numpy(candidate_interaction[val_mask]).float(),
        torch.from_numpy(expert_idx[val_mask]).long(),
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Loading scorer data from {', '.join(str(path) for path in data_dirs)}")
    print(f"  transitions={len(state_features):,} train={train_mask.sum():,} val={val_mask.sum():,}")
    print(f"  candidate_traj shape={candidate_traj.shape} candidate_interaction shape={candidate_interaction.shape} device={device}")

    model = TrajectoryScorer(point_horizon=point_horizon, interaction_dim=interaction_dim).to(device)
    log = train_model(model, train_dl, val_dl, epochs=args.epochs, horizon_seconds=horizon_seconds, device=device)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "feat_mean": feat_mean.tolist(),
            "feat_std": feat_std.tolist(),
            "point_horizon": point_horizon,
            "horizon_seconds": horizon_seconds,
            "n_features": N_FEATURES,
            "n_tokens": N_TOKENS,
            "state_hidden": STATE_HIDDEN,
            "traj_hidden": TRAJ_HIDDEN,
            "interaction_hidden": INTERACTION_HIDDEN,
            "interaction_dim": interaction_dim,
            "fusion_hidden": FUSION_HIDDEN,
            "temperature": float(model.log_temperature.exp().detach().cpu()),
            "source_meta": meta,
            "device": device,
        },
        args.output_dir / "trajectory_scorer.pt",
    )
    (args.output_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    print(f"\nSaved scorer model to {args.output_dir}")
    print(f"  final val_acc={log[-1]['val_acc']:.3f}")


if __name__ == "__main__":
    main()
