# nuPlan Selected-Token Realized Clearance Audit

- Scene count: `1000`
- Proxy-safe selected count: `732`
- Proxy-safe + realized safe count: `324`
- Proxy-safe + realized near/collision count: `408`
- Proxy-safe + realized missing count: `0`
- Proxy-safe realized near/collision rate: `0.557`
- Proxy-safe realized near/collision 95% CI: `[0.521, 0.593]`
- Realized clearance source: `log_replay`
- Replay-oracle diagnostics are secondary analyses; they do not redefine the primary failure rungs.

## Failure Table

| Failure rung | Count | Rate |
|---|---:|---:|
| no actor/state visibility | 0 | 0.000 |
| no safe token existed | 63 | 0.063 |
| safe token existed but selector missed it | 205 | 0.205 |
| proxy-safe but replay-infeasible | 408 | 0.408 |
| metric/spec ambiguity | 0 | 0.000 |

## Replay Oracle Diagnostics

| Diagnostic | Count | Rate |
|---|---:|---:|
| selected_replay_safe | 324 | 0.324 |
| selected_failed_no_replay_safe_alternative | 223 | 0.223 |
| selected_failed_with_replay_safe_alternative | 453 | 0.453 |
| selected_replay_missing | 0 | 0.000 |

## Primary Rung By Replay Oracle Diagnostic

| Failure rung | Replay oracle diagnostic | Count |
|---|---|---:|
| no safe token existed | selected_failed_no_replay_safe_alternative | 63 |
| proxy-safe but replay-infeasible | selected_failed_no_replay_safe_alternative | 82 |
| proxy-safe but replay-infeasible | selected_failed_with_replay_safe_alternative | 326 |
| resolved safe | selected_replay_safe | 324 |
| safe token existed but selector missed it | selected_failed_no_replay_safe_alternative | 78 |
| safe token existed but selector missed it | selected_failed_with_replay_safe_alternative | 127 |

## Proxy-Safe Replay Failure Split

| Type | Count | Rate |
|---|---:|---:|
| no_replay_safe_alternative | 82 | 0.201 |
| replay_safe_alternative_exists | 326 | 0.799 |

## Token Failure Table

| Token | Proxy-safe selected | Realized near/collision | Failure rate |
|---|---:|---:|---:|
| evasive_left | 94 | 66 | 0.702 |
| evasive_right | 92 | 59 | 0.641 |
| fast_left | 151 | 61 | 0.404 |
| fast_right | 181 | 73 | 0.403 |
| lane_recover | 22 | 22 | 1.000 |
| maintain | 84 | 46 | 0.548 |
| nudge_left | 5 | 5 | 1.000 |
| nudge_right | 42 | 42 | 1.000 |
| slow_yield | 29 | 23 | 0.793 |
| wide_left | 17 | 2 | 0.118 |
| wide_right | 15 | 9 | 0.600 |

## Threshold Sensitivity

| Threshold | Proxy-safe + realized near/collision | Rate |
|---:|---:|---:|
| 0.5 m | 240 | 0.328 |
| 1.0 m | 408 | 0.557 |
| 1.5 m | 546 | 0.746 |
| 2.0 m | 617 | 0.843 |

## Horizon Sensitivity

| Horizon | Proxy-safe selected failures | Proxy-safe stop failures | No replay-safe token |
|---:|---:|---:|---:|
| 1.0 s | 142 | 0 | 50 |
| 2.0 s | 267 | 0 | 115 |
| 3.0 s | 285 | 0 | 192 |
| 4.0 s | 408 | 0 | 223 |
| 5.0 s | n/a | n/a | n/a |

## Proxy-Safe Replay Failure Onset

| Failure onset | Count |
|---|---:|
| immediate_<=1s | 142 |
| mid_1to2s | 125 |
| mid_2to3s | 18 |
| late_3to4s | 123 |

## No-Safe-Token Causes

| Cause | Count |
|---|---:|
| stop still unsafe | 63 |

## Replay Oracle Cases

| Case | Count |
|---|---:|
| proxy-safe token exists and replay-safe token exists | 777 |
| proxy-safe token exists but no replay-safe token exists | 160 |
| selected token fails but another replay-safe token exists | 453 |
| no proxy-safe token but replay-safe token exists | 0 |

## Replay Oracle Misses

| Selected token | Count | Most common replay-safe oracle token | Most common oracle count |
|---|---:|---|---:|
| evasive_left | 59 | wide_left_crawl | 56 |
| evasive_right | 30 | reverse_creep | 17 |
| fast_left | 105 | wide_left | 23 |
| fast_right | 130 | wide_left_crawl | 30 |
| lane_recover | 17 | evasive_right | 17 |
| maintain | 44 | wide_right_crawl | 16 |
| nudge_left | 5 | wide_left_crawl | 5 |
| nudge_right | 42 | stop | 23 |
| slow_yield | 15 | crawl | 6 |
| wide_left | 2 | wide_left_crawl | 2 |
| wide_right | 4 | stop | 4 |

## Failed Stop Causes

| Cause | Count |
|---|---:|

## Stop Vs Evasive Right

- Failed stop scenes: `0`
- `evasive_right` available: `0`
- `evasive_right` would avoid failure: `0`
