# NAVSIM v2 mapped smoke g2 K=8 seed0

Scope: mapped `navhard_two_stage` smoke group `g2`, 48 scenes total: 4 original NAVSIM v2 tokens plus 44 reactive synthetic scenarios. This is larger than `g1` but still a smoke-scale artifact.

## Candidate-bank audit

| Metric | Value |
|---|---:|
| Scenes | 48 |
| Candidates | 8 |
| Official top-1 EPDMS | 0.045811 |
| Official best fixed candidate | 2 |
| Official best fixed EPDMS | 0.418909 |
| Token-oracle proxy | 0.614206 |

## Ridge probe diagnostic

Probe scores below use per-token proxy rows, not official combined EPDMS.

| Probe | Features | Top-1 proxy | Oracle proxy | LOSO proxy | Gap closed |
|---|---:|---:|---:|---:|---:|
| Zero-perception | 35 | 0.545394 | 0.614206 | 0.515941 | -42.8% |
| Full probe | 37 | 0.545394 | 0.614206 | 0.518862 | -38.6% |

Readout: g2 confirms real K=8 selection headroom in the candidate bank, but the current low-capacity LOSO ridge probe does not recover it. This pushes the paper away from claiming a universal zero-perception linear selector and toward a cleaner audit claim: OneVL generates recoverable alternatives, but selection requires either more training data, richer supervision, or a non-linear/contextual selector.
