# nuPlan Selected-Token Realized Clearance Audit

- Scene count: `1000`
- Proxy-safe selected count: `774`
- Proxy-safe + realized safe count: `373`
- Proxy-safe + realized near/collision count: `401`
- Proxy-safe + realized missing count: `0`
- Proxy-safe realized near/collision rate: `0.518`
- Proxy-safe realized near/collision 95% CI: `[0.483, 0.553]`
- Realized clearance source: `log_replay`
- Replay-oracle diagnostics are secondary analyses; they do not redefine the primary failure rungs.

## Failure Table

| Failure rung | Count | Rate |
|---|---:|---:|
| no actor/state visibility | 0 | 0.000 |
| no safe token existed | 95 | 0.095 |
| safe token existed but selector missed it | 131 | 0.131 |
| proxy-safe but replay-infeasible | 401 | 0.401 |
| metric/spec ambiguity | 0 | 0.000 |

## Replay Oracle Diagnostics

| Diagnostic | Count | Rate |
|---|---:|---:|
| selected_replay_safe | 373 | 0.373 |
| selected_failed_no_replay_safe_alternative | 281 | 0.281 |
| selected_failed_with_replay_safe_alternative | 346 | 0.346 |
| selected_replay_missing | 0 | 0.000 |

## Primary Rung By Replay Oracle Diagnostic

| Failure rung | Replay oracle diagnostic | Count |
|---|---|---:|
| no safe token existed | selected_failed_no_replay_safe_alternative | 95 |
| proxy-safe but replay-infeasible | selected_failed_no_replay_safe_alternative | 105 |
| proxy-safe but replay-infeasible | selected_failed_with_replay_safe_alternative | 296 |
| resolved safe | selected_replay_safe | 373 |
| safe token existed but selector missed it | selected_failed_no_replay_safe_alternative | 81 |
| safe token existed but selector missed it | selected_failed_with_replay_safe_alternative | 50 |

## Proxy-Safe Replay Failure Split

| Type | Count | Rate |
|---|---:|---:|
| no_replay_safe_alternative | 105 | 0.262 |
| replay_safe_alternative_exists | 296 | 0.738 |

## Token Failure Table

| Token | Proxy-safe selected | Realized near/collision | Failure rate |
|---|---:|---:|---:|
| evasive_left | 99 | 70 | 0.707 |
| evasive_right | 100 | 65 | 0.650 |
| lane_recover | 95 | 38 | 0.400 |
| maintain | 274 | 122 | 0.445 |
| nudge_left | 11 | 11 | 1.000 |
| nudge_right | 73 | 50 | 0.685 |
| slow_yield | 122 | 45 | 0.369 |

## Threshold Sensitivity

| Threshold | Proxy-safe + realized near/collision | Rate |
|---:|---:|---:|
| 0.5 m | 254 | 0.328 |
| 1.0 m | 401 | 0.518 |
| 1.5 m | 612 | 0.791 |
| 2.0 m | 704 | 0.910 |

## Horizon Sensitivity

| Horizon | Proxy-safe selected failures | Proxy-safe stop failures | No replay-safe token |
|---:|---:|---:|---:|
| 1.0 s | 107 | 0 | 123 |
| 2.0 s | 234 | 0 | 193 |
| 3.0 s | 267 | 0 | 246 |
| 4.0 s | 401 | 0 | 281 |
| 5.0 s | n/a | n/a | n/a |

## Proxy-Safe Replay Failure Onset

| Failure onset | Count |
|---|---:|
| immediate_<=1s | 107 |
| mid_1to2s | 127 |
| mid_2to3s | 33 |
| late_3to4s | 134 |

## No-Safe-Token Causes

| Cause | Count |
|---|---:|
| stop still unsafe | 95 |

## Replay Oracle Cases

| Case | Count |
|---|---:|
| proxy-safe token exists and replay-safe token exists | 719 |
| proxy-safe token exists but no replay-safe token exists | 186 |
| selected token fails but another replay-safe token exists | 346 |
| no proxy-safe token but replay-safe token exists | 0 |

## Replay Oracle Misses

| Selected token | Count | Most common replay-safe oracle token | Most common oracle count |
|---|---:|---|---:|
| evasive_left | 63 | stop | 31 |
| evasive_right | 34 | stop | 32 |
| lane_recover | 33 | evasive_right | 20 |
| maintain | 121 | stop | 54 |
| nudge_left | 23 | stop | 13 |
| nudge_right | 52 | stop | 28 |
| slow_yield | 20 | crawl | 17 |

## Failed Stop Causes

| Cause | Count |
|---|---:|

## Stop Vs Evasive Right

- Failed stop scenes: `0`
- `evasive_right` available: `0`
- `evasive_right` would avoid failure: `0`
