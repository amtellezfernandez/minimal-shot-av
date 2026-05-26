# Candidate Regret Pipeline

- Output dir: `artifacts/corl2027/candidate_regret_pipeline_expert1000_g0_g1`
- Lifecycle available: `True`
- Curriculum available: `True`
- Claim boundary: Pipeline is analysis-complete only when every candidate stage has evaluator metrics. Generation-ready-only dumps are preserved but not used for regret claims.

| Stage | Readiness | Scenes | Candidates | Gen gap | Safety regret | Utility regret | Top-1 fail |
|---|---|---:|---:|---:|---:|---:|---:|
| g0 | analysis_ready | 1000 | 9000 | 281 | 346 | 44 | 0.627 |
| g1 | analysis_ready | 1000 | 9000 | 281 | 13 | 228 | 0.294 |

