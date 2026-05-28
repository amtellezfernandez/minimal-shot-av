# Exploratory Meta-Calibration Diagnosis

This diagnosis explains the bad exploratory run with `meta_fail_penalty=30`. That setting
does not beat fixed `lambda40` because its training objective is utility-heavy and usually
labels low lambda as optimal. The corrected exploratory setting is
`meta_fail_penalty=250`, which restores overperformance versus `lambda40` on replay risk.

## Per-Split Comparison

| Seed | Holdout scenes | lambda40 fail | lambda80 fail | meta fail | meta worse than lambda40 | meta better than lambda40 | mean progress delta vs lambda40 | mean ADE delta vs lambda40 | meta choice hist | train target hist |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 300 | 0.313 | 0.163 | 0.347 | 10 | 0 | 0.599 | -0.191 | `{"0.0": 30, "20.0": 107, "40.0": 85, "80.0": 78}` | `{"0.0": 559, "20.0": 105, "40.0": 32, "80.0": 4}` |
| 1 | 155 | 0.265 | 0.252 | 0.297 | 6 | 1 | 1.937 | -0.253 | `{"0.0": 28, "20.0": 80, "40.0": 31, "80.0": 16}` | `{"0.0": 666, "20.0": 146, "40.0": 21, "80.0": 12}` |
| 2 | 300 | 0.503 | 0.490 | 0.513 | 4 | 1 | 1.103 | -0.220 | `{"0.0": 121, "20.0": 80, "40.0": 65, "80.0": 34}` | `{"0.0": 520, "20.0": 136, "40.0": 40, "80.0": 4}` |
| 3 | 300 | 0.637 | 0.627 | 0.717 | 24 | 0 | 3.431 | -1.335 | `{"0.0": 155, "20.0": 124, "40.0": 21}` | `{"0.0": 518, "20.0": 166, "40.0": 14, "80.0": 2}` |
| 4 | 155 | 0.303 | 0.271 | 0.303 | 0 | 0 | 0.136 | -0.038 | `{"0.0": 24, "20.0": 7, "40.0": 79, "80.0": 45}` | `{"0.0": 632, "20.0": 158, "40.0": 45, "80.0": 10}` |

## Aggregate

- Holdout chosen lambda histogram: `{"0.0": 358, "20.0": 398, "40.0": 281, "80.0": 173}`
- Train objective target histogram: `{"0.0": 2895, "20.0": 711, "40.0": 152, "80.0": 32}`

## Main failure mechanism

- The train objective target is dominated by `lambda0`/`lambda20`. That follows from the scalar objective: `-30 * replay_failure + progress - 2 * ADE` lets progress and imitation compensate for replay failures.
- Fixed `lambda40` is a replay-risk operating point. Meta-value is trained to optimize mixed utility. Therefore it can improve progress/ADE while underperforming `lambda40` on replay-infeasible rate.
- If the goal is to beat `lambda40` on replay risk, the meta target must be safety-dominant.
  The fail-penalty sweep shows `meta_fail_penalty=250` is sufficient on the current
  1k-scene artifact.

## Common losing patterns

- Seed 0 worse patterns: `[{"lambda": "20.0", "meta_token": "maintain", "lambda40_token": "evasive_left", "count": 8}, {"lambda": "20.0", "meta_token": "maintain", "lambda40_token": "crawl", "count": 2}]`
- Seed 1 worse patterns: `[{"lambda": "20.0", "meta_token": "slow_yield", "lambda40_token": "crawl", "count": 5}, {"lambda": "20.0", "meta_token": "nudge_left", "lambda40_token": "slow_yield", "count": 1}]`
- Seed 2 worse patterns: `[{"lambda": "20.0", "meta_token": "lane_recover", "lambda40_token": "stop", "count": 2}, {"lambda": "80.0", "meta_token": "slow_yield", "lambda40_token": "nudge_left", "count": 1}, {"lambda": "20.0", "meta_token": "slow_yield", "lambda40_token": "crawl", "count": 1}]`
- Seed 3 worse patterns: `[{"lambda": "20.0", "meta_token": "lane_recover", "lambda40_token": "evasive_right", "count": 8}, {"lambda": "20.0", "meta_token": "maintain", "lambda40_token": "evasive_right", "count": 6}, {"lambda": "0.0", "meta_token": "lane_recover", "lambda40_token": "evasive_right", "count": 5}, {"lambda": "0.0", "meta_token": "maintain", "lambda40_token": "nudge_right", "count": 2}, {"lambda": "20.0", "meta_token": "nudge_right", "lambda40_token": "slow_yield", "count": 2}, {"lambda": "0.0", "meta_token": "lane_recover", "lambda40_token": "nudge_right", "count": 1}]`
- Seed 4 worse patterns: `[]`
