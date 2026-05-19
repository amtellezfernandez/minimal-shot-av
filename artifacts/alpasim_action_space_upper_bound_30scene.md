# AlpaSim action-space upper-bound audit

This audit reconstructs the token candidate set and the selector-free direct grid on the baseline collision timeline after injecting the world-frame oracle actor proxy. It is an open-loop proxy upper bound, not a counterfactual AlpaSim physics replay.

## Summary

- Baseline collision scenes audited: `18`.
- Baseline first-impact components: `{'collision_front': 3, 'collision_lateral': 1, 'collision_rear': 14}`.
- Frame status counts: `{'ok': 1117}`.
- Actionable window: at least `5` frames (`0.5s`) before impact.
- Proxy no-overlap threshold: `0.00m`; reported margin threshold: `0.55m`.
- Token candidate set proxy-safe actionable frames: `986/1027` across `18` scenes.
- Token selected proxy-safe actionable frames: `889` across `18` scenes.
- Token selected margin-safe actionable frames: `850` across `18` scenes.
- Direct grid proxy-safe actionable frames: `962/1027` across `18` scenes.
- Direct grid cost-selected proxy-safe actionable frames: `962` across `18` scenes.
- Direct grid cost-selected margin-safe actionable frames: `920` across `18` scenes.
- Direct grid cost-missed proxy-safe actionable frames: `0` across `0` scenes.
- Direct grid cost-missed margin-safe actionable frames: `0` across `0` scenes.
- Bucket counts: `{'direct_grid_cost_selects_proxy_safe_action_before_collision': 18}`.

## Interpretation

If the direct grid contains proxy-safe actions but its cost rejects them, the selector-free planner is likely under-tuned or incorrectly weighted. If the direct grid cost selects proxy-safe actions before impact but the actual closed-loop run still collides, the likely bottleneck moves to controller execution, receding-horizon dynamics, or simulator mismatch. If neither token candidates nor the direct grid contain proxy-safe actions, the available action space is too weak for this collision surface.

## Per-scene buckets

| Scene | Component | Bucket | Token safe frames | Token selected safe | Direct safe frames | Direct selected safe | Direct cost misses | First direct safe lead |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 001_clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 25 | 9 | 23 | 23 | 0 | 32 |
| 002_clipgt-804afc4a-fd1e-4f58-bd39-a4c486a916e5 | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 36 | 35 | 36 | 36 | 0 | 42 |
| 004_clipgt-9b3509c2-a5fa-42c0-b0e5-feb617e9bf4e | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 44 | 43 | 44 | 44 | 0 | 50 |
| 005_clipgt-90d1908c-9fdc-40ea-a5a1-351240aa323e | collision_front | direct_grid_cost_selects_proxy_safe_action_before_collision | 22 | 16 | 25 | 25 | 0 | 29 |
| 007_clipgt-8aaa1135-573f-4f6c-8ff8-7ac6bd6ea5d6 | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 32 | 31 | 29 | 29 | 0 | 39 |
| 008_clipgt-9f8acd00-e8f5-467a-867d-b205f8d4d75d | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 47 | 46 | 47 | 47 | 0 | 53 |
| 009_clipgt-97a2e621-a27b-4b87-9a58-b6b740fc1b4e | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 44 | 37 | 39 | 39 | 0 | 48 |
| 011_clipgt-9b6a51bd-cb54-4a68-8ed8-0e33a504c67e | collision_front | direct_grid_cost_selects_proxy_safe_action_before_collision | 18 | 15 | 18 | 18 | 0 | 28 |
| 013_clipgt-89f2eeb8-f4e4-42a1-8e07-d60003d223dc | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 88 | 87 | 88 | 88 | 0 | 93 |
| 014_clipgt-7c4f23b7-be14-4505-b332-6d1ceb36cd75 | collision_front | direct_grid_cost_selects_proxy_safe_action_before_collision | 23 | 17 | 22 | 22 | 0 | 27 |
| 018_clipgt-825dc886-0491-482a-abe3-e1743f5972b8 | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 36 | 36 | 35 | 35 | 0 | 44 |
| 019_clipgt-994977b5-4db6-43eb-950a-c0faaccc930b | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 34 | 32 | 36 | 36 | 0 | 43 |
| 020_clipgt-8578f2c7-70a0-4877-980a-7ba65bcded30 | collision_lateral | direct_grid_cost_selects_proxy_safe_action_before_collision | 168 | 160 | 166 | 166 | 0 | 172 |
| 021_clipgt-7eaac028-ff62-4f54-9cca-52c8f49ba87e | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 43 | 42 | 39 | 39 | 0 | 49 |
| 022_clipgt-82da2d46-4ec7-4f8c-8f54-e6febb1c303b | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 27 | 7 | 27 | 27 | 0 | 31 |
| 023_clipgt-89c2e887-cffa-4b7c-86cf-80b89be1cb06 | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 135 | 135 | 135 | 135 | 0 | 141 |
| 025_clipgt-7c05aee0-88f1-4338-a48b-5cbe24d5b888 | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 116 | 102 | 109 | 109 | 0 | 122 |
| 030_clipgt-98dd56b2-3e60-49ec-9366-70b610dd93cc | collision_rear | direct_grid_cost_selects_proxy_safe_action_before_collision | 48 | 39 | 44 | 44 | 0 | 52 |

## Impact-frame snapshot

| Scene | Lead | Token selected safe | Token safe count | Direct selected safe | Direct safe count | Direct selected clearance | Direct best clearance |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| 001_clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb | 0 | False | 0 | False | 0 | -0.5802 | -0.5802 |
| 002_clipgt-804afc4a-fd1e-4f58-bd39-a4c486a916e5 | 0 | False | 0 | False | 0 | -1.4351 | -1.4351 |
| 004_clipgt-9b3509c2-a5fa-42c0-b0e5-feb617e9bf4e | 0 | False | 0 | False | 0 | -1.3181 | -1.3181 |
| 005_clipgt-90d1908c-9fdc-40ea-a5a1-351240aa323e | 0 | True | 6 | True | 21 | 1.7335 | 1.7927 |
| 007_clipgt-8aaa1135-573f-4f6c-8ff8-7ac6bd6ea5d6 | 0 | False | 0 | False | 0 | -1.1449 | -1.1449 |
| 008_clipgt-9f8acd00-e8f5-467a-867d-b205f8d4d75d | 0 | False | 0 | False | 0 | -1.4723 | -1.4723 |
| 009_clipgt-97a2e621-a27b-4b87-9a58-b6b740fc1b4e | 0 | False | 0 | False | 0 | -0.8071 | -0.8071 |
| 011_clipgt-9b6a51bd-cb54-4a68-8ed8-0e33a504c67e | 0 | False | 3 | True | 8 | 1.4585 | 1.4585 |
| 013_clipgt-89f2eeb8-f4e4-42a1-8e07-d60003d223dc | 0 | False | 0 | False | 0 | -0.4539 | -0.4539 |
| 014_clipgt-7c4f23b7-be14-4505-b332-6d1ceb36cd75 | 0 | False | 1 | True | 7 | 1.3627 | 1.3627 |
| 018_clipgt-825dc886-0491-482a-abe3-e1743f5972b8 | 0 | False | 0 | False | 0 | -1.0122 | -1.0122 |
| 019_clipgt-994977b5-4db6-43eb-950a-c0faaccc930b | 0 | False | 0 | False | 0 | -1.2484 | -1.2484 |
| 020_clipgt-8578f2c7-70a0-4877-980a-7ba65bcded30 | 0 | True | 9 | True | 21 | 0.5811 | 0.5859 |
| 021_clipgt-7eaac028-ff62-4f54-9cca-52c8f49ba87e | 0 | False | 0 | False | 0 | -0.7012 | -0.7012 |
| 022_clipgt-82da2d46-4ec7-4f8c-8f54-e6febb1c303b | 0 | False | 1 | True | 1 | 0.033 | 0.033 |
| 023_clipgt-89c2e887-cffa-4b7c-86cf-80b89be1cb06 | 0 | False | 0 | False | 0 | -1.3119 | -1.3119 |
| 025_clipgt-7c05aee0-88f1-4338-a48b-5cbe24d5b888 | 0 | False | 0 | False | 0 | -1.1112 | -1.1112 |
| 030_clipgt-98dd56b2-3e60-49ec-9366-70b610dd93cc | 0 | False | 0 | False | 0 | -0.9761 | -0.9761 |
