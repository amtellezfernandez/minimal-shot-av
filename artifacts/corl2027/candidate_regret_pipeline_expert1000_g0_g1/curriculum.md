# Candidate Regret Curriculum

- G0 report: `artifacts/corl2027/candidate_regret_pipeline_expert1000_g0_g1/g0_analysis.json`
- G1 report: `artifacts/corl2027/candidate_regret_pipeline_expert1000_g0_g1/g1_analysis.json`
- Claim boundary: Curriculum for generator bootstrapping. Safety and utility phases have in-bank teacher trajectories; generation-gap scenes are expansion requests because no candidate in K is safe.

## Phase Summary

| Phase | Count | Mean regret | Mean weight | Main histogram |
|---|---:|---:|---:|---|
| safety_recovery | 346 | 226.103 | 2.806 | `{"crawl": 92, "evasive_left": 13, "evasive_right": 64, "lane_recover": 3, "nudge_left": 4, "nudge_right": 28, "slow_yield": 96, "stop": 46}` |
| utility_recovery | 228 | 5.696 | 0.767 | `{"crawl": 1, "evasive_left": 7, "evasive_right": 17, "lane_recover": 15, "maintain": 135, "nudge_left": 3, "nudge_right": 47, "slow_yield": 3}` |
| candidate_expansion | 281 | n/a | n/a | `{"crawl": 20, "evasive_left": 10, "evasive_right": 53, "lane_recover": 18, "maintain": 134, "nudge_right": 5, "slow_yield": 22, "stop": 19}` |

## Training Use

- Phase 1 trains the generator toward replay-safe teachers where top-1 failed.
- Phase 2 trains utility recovery after safety regret is mostly removed.
- Candidate-expansion requests require new samples or a richer generator; no in-bank teacher exists.

