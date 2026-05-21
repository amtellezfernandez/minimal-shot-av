# AlpaSim 30-scene candidate-counterfactual audit

This audit injects the world-frame oracle actor proxy into the baseline timeline and reconstructs selector-side candidate feasibility. It does not replay candidates through the AlpaSim controller or traffic simulator.

## Summary

- Baseline collision scenes audited: `18`.
- Same-scene oracle collisions among those scenes: `18`.
- Baseline first-impact components: `{'collision_front': 3, 'collision_lateral': 1, 'collision_rear': 14}`.
- Frame status counts: `{'ok': 108}`.
- Actionable window: at least `5` frames (`0.5s`) before first impact.
- Axis-proxy actionable safe frames: `52/54`.
- Actor-axis actionable safe frames: `25/54`.
- Actor-axis safe frame selected by baseline policy: `20`.
- Actor-axis safe frame missed by baseline policy: `5`.
- Axis-proxy safe at impact: `5/18`.
- Actor-axis safe at impact: `0/18`.
- Bucket counts: `{'actor_axis_safe_candidate_missed_by_baseline_selector': 2, 'baseline_selected_actor_axis_safe_candidate_but_collision_persisted': 12, 'mixed_actor_axis_safe_candidate_intermittently_missed': 2, 'no_actor_axis_safe_candidate_actionable': 2}`.

## Interpretation

The ordinary axis scorer often still marks candidates safe after actor injection, which explains why actor visibility alone did not remove rear collisions. The stricter actor-axis test is more diagnostic: when it finds no actionable safe candidate, the existing candidate set/controller envelope is the likely bottleneck; when it finds a safe candidate that the baseline-selected token violates, the result is only a selector-side miss until a controller replay confirms that candidate actually avoids impact.

## Per-scene buckets

| Scene | Component | Bucket | Actor-axis safe frames | Missed safe frames | Impact actor-safe | Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| 001_clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb | collision_rear | no_actor_axis_safe_candidate_actionable | 0 | 0 | False | no actor-axis-safe candidate in audited frames |
| 002_clipgt-804afc4a-fd1e-4f58-bd39-a4c486a916e5 | collision_rear | baseline_selected_actor_axis_safe_candidate_but_collision_persisted | 2 | 0 | False | earliest actor-safe lead 20f; best safe evasive_left; selected violation None |
| 004_clipgt-9b3509c2-a5fa-42c0-b0e5-feb617e9bf4e | collision_rear | baseline_selected_actor_axis_safe_candidate_but_collision_persisted | 2 | 0 | False | earliest actor-safe lead 20f; best safe maintain; selected violation None |
| 005_clipgt-90d1908c-9fdc-40ea-a5a1-351240aa323e | collision_front | mixed_actor_axis_safe_candidate_intermittently_missed | 3 | 2 | False | earliest actor-safe lead 20f; best safe nudge_left; selected violation None |
| 007_clipgt-8aaa1135-573f-4f6c-8ff8-7ac6bd6ea5d6 | collision_rear | baseline_selected_actor_axis_safe_candidate_but_collision_persisted | 1 | 0 | False | earliest actor-safe lead 20f; best safe evasive_left; selected violation None |
| 008_clipgt-9f8acd00-e8f5-467a-867d-b205f8d4d75d | collision_rear | baseline_selected_actor_axis_safe_candidate_but_collision_persisted | 2 | 0 | False | earliest actor-safe lead 20f; best safe evasive_left; selected violation None |
| 009_clipgt-97a2e621-a27b-4b87-9a58-b6b740fc1b4e | collision_rear | baseline_selected_actor_axis_safe_candidate_but_collision_persisted | 1 | 0 | False | earliest actor-safe lead 20f; best safe nudge_left; selected violation None |
| 011_clipgt-9b6a51bd-cb54-4a68-8ed8-0e33a504c67e | collision_front | baseline_selected_actor_axis_safe_candidate_but_collision_persisted | 1 | 0 | False | earliest actor-safe lead 20f; best safe maintain; selected violation None |
| 013_clipgt-89f2eeb8-f4e4-42a1-8e07-d60003d223dc | collision_rear | baseline_selected_actor_axis_safe_candidate_but_collision_persisted | 2 | 0 | False | earliest actor-safe lead 20f; best safe lane_recover; selected violation None |
| 014_clipgt-7c4f23b7-be14-4505-b332-6d1ceb36cd75 | collision_front | baseline_selected_actor_axis_safe_candidate_but_collision_persisted | 1 | 0 | False | earliest actor-safe lead 20f; best safe slow_yield; selected violation None |
| 018_clipgt-825dc886-0491-482a-abe3-e1743f5972b8 | collision_rear | baseline_selected_actor_axis_safe_candidate_but_collision_persisted | 1 | 0 | False | earliest actor-safe lead 20f; best safe nudge_right; selected violation None |
| 019_clipgt-994977b5-4db6-43eb-950a-c0faaccc930b | collision_rear | actor_axis_safe_candidate_missed_by_baseline_selector | 1 | 1 | False | earliest actor-safe lead 20f; best safe nudge_left; selected violation actor_horizon_clearance |
| 020_clipgt-8578f2c7-70a0-4877-980a-7ba65bcded30 | collision_lateral | mixed_actor_axis_safe_candidate_intermittently_missed | 3 | 1 | False | earliest actor-safe lead 20f; best safe maintain; selected violation None |
| 021_clipgt-7eaac028-ff62-4f54-9cca-52c8f49ba87e | collision_rear | baseline_selected_actor_axis_safe_candidate_but_collision_persisted | 1 | 0 | False | earliest actor-safe lead 20f; best safe maintain; selected violation None |
| 022_clipgt-82da2d46-4ec7-4f8c-8f54-e6febb1c303b | collision_rear | no_actor_axis_safe_candidate_actionable | 0 | 0 | False | no actor-axis-safe candidate in audited frames |
| 023_clipgt-89c2e887-cffa-4b7c-86cf-80b89be1cb06 | collision_rear | baseline_selected_actor_axis_safe_candidate_but_collision_persisted | 2 | 0 | False | earliest actor-safe lead 20f; best safe slow_yield; selected violation None |
| 025_clipgt-7c05aee0-88f1-4338-a48b-5cbe24d5b888 | collision_rear | actor_axis_safe_candidate_missed_by_baseline_selector | 1 | 1 | False | earliest actor-safe lead 20f; best safe nudge_left; selected violation actor_horizon_clearance |
| 030_clipgt-98dd56b2-3e60-49ec-9366-70b610dd93cc | collision_rear | baseline_selected_actor_axis_safe_candidate_but_collision_persisted | 1 | 0 | False | earliest actor-safe lead 20f; best safe nudge_left; selected violation None |

## Frame Detail

### 001_clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | slow_yield | maintain,lane_recover,slow_yield,crawl | -- | actor_horizon_clearance | None | ok |
| 10 | evasive_right | evasive_left,nudge_left,slow_yield,lane_recover | -- | actor_horizon_clearance | None | ok |
| 5 | evasive_right | nudge_right,evasive_right,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 2 | maintain | nudge_left,evasive_left,maintain,lane_recover | -- | actor_action_clearance | None | ok |
| 1 | maintain | evasive_left,nudge_left,crawl,maintain | -- | actor_action_clearance | None | ok |
| 0 | maintain | evasive_left,nudge_left,crawl,maintain | -- | actor_action_clearance | None | ok |

### 002_clipgt-804afc4a-fd1e-4f58-bd39-a4c486a916e5

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | crawl | evasive_left,slow_yield,lane_recover,nudge_left | evasive_left,slow_yield,lane_recover,nudge_left | None | evasive_left | ok |
| 10 | evasive_left | maintain,lane_recover,nudge_left,nudge_right | maintain,lane_recover,nudge_left,nudge_right | None | maintain | ok |
| 5 | evasive_right | nudge_left,evasive_left,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 2 | evasive_right | nudge_left,evasive_left,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 1 | evasive_right | nudge_left,evasive_left,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 0 | evasive_right | -- | -- | actor_action_clearance | None | ok |

### 004_clipgt-9b3509c2-a5fa-42c0-b0e5-feb617e9bf4e

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | evasive_right | maintain,lane_recover,nudge_left,nudge_right | maintain,lane_recover,nudge_left,nudge_right | None | maintain | ok |
| 10 | evasive_right | nudge_right,evasive_right,lane_recover,slow_yield | nudge_right,lane_recover,slow_yield,crawl | None | nudge_right | ok |
| 5 | evasive_right | nudge_right,lane_recover,slow_yield,crawl | -- | actor_action_clearance | None | ok |
| 2 | maintain | nudge_left,evasive_left,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 1 | maintain | nudge_left,evasive_left,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 0 | maintain | -- | -- | actor_action_clearance | None | ok |

### 005_clipgt-90d1908c-9fdc-40ea-a5a1-351240aa323e

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | slow_yield | nudge_left,maintain,evasive_left,lane_recover | nudge_left,maintain,evasive_left,lane_recover | None | nudge_left | ok |
| 10 | evasive_right | -- | stop | actor_action_clearance | stop | ok |
| 5 | evasive_left | -- | stop | actor_action_clearance | stop | ok |
| 2 | evasive_right | nudge_right,evasive_right,lane_recover,slow_yield | evasive_right,lane_recover,slow_yield,maintain | None | evasive_right | ok |
| 1 | evasive_right | evasive_right,slow_yield,nudge_right,crawl | -- | lane_margin | None | ok |
| 0 | evasive_right | evasive_right,nudge_right,lane_recover,slow_yield | -- | lane_margin | None | ok |

### 007_clipgt-8aaa1135-573f-4f6c-8ff8-7ac6bd6ea5d6

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | crawl | slow_yield,crawl | evasive_left,nudge_left,slow_yield,lane_recover | None | evasive_left | ok |
| 10 | evasive_right | nudge_left,maintain,evasive_left,lane_recover | -- | actor_horizon_clearance | None | ok |
| 5 | evasive_right | nudge_left,evasive_left,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 2 | evasive_right | nudge_left,evasive_left,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 1 | evasive_right | nudge_left,evasive_left,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 0 | evasive_right | -- | -- | actor_action_clearance | None | ok |

### 008_clipgt-9f8acd00-e8f5-467a-867d-b205f8d4d75d

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | evasive_right | nudge_left,evasive_left,lane_recover,slow_yield | evasive_left,nudge_left,lane_recover,slow_yield | None | evasive_left | ok |
| 10 | evasive_right | maintain,lane_recover,slow_yield,nudge_left | maintain,lane_recover,slow_yield,nudge_left | None | maintain | ok |
| 5 | maintain | nudge_left,evasive_left,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 2 | maintain | nudge_right,evasive_right,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 1 | maintain | nudge_right,evasive_right,maintain,lane_recover | -- | actor_action_clearance | None | ok |
| 0 | lane_recover | -- | -- | actor_action_clearance | None | ok |

### 009_clipgt-97a2e621-a27b-4b87-9a58-b6b740fc1b4e

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | evasive_right | nudge_left,evasive_left,lane_recover,slow_yield | nudge_left,evasive_left,lane_recover,slow_yield | None | nudge_left | ok |
| 10 | maintain | nudge_left,evasive_left,lane_recover,slow_yield | -- | actor_horizon_clearance | None | ok |
| 5 | maintain | nudge_left,evasive_left,lane_recover,maintain | -- | actor_action_clearance | None | ok |
| 2 | maintain | nudge_right,evasive_right,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 1 | maintain | nudge_right,evasive_right,maintain,lane_recover | -- | actor_action_clearance | None | ok |
| 0 | lane_recover | -- | -- | actor_action_clearance | None | ok |

### 011_clipgt-9b6a51bd-cb54-4a68-8ed8-0e33a504c67e

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | slow_yield | maintain,lane_recover,slow_yield,crawl | maintain,lane_recover,slow_yield,crawl | None | maintain | ok |
| 10 | evasive_right | slow_yield,lane_recover,nudge_left,crawl | -- | actor_action_clearance | None | ok |
| 5 | maintain | evasive_right | -- | actor_action_clearance | None | ok |
| 2 | maintain | evasive_right | -- | actor_action_clearance | None | ok |
| 1 | maintain | evasive_right | -- | actor_action_clearance | None | ok |
| 0 | lane_recover | evasive_right | -- | actor_horizon_clearance | None | ok |

### 013_clipgt-89f2eeb8-f4e4-42a1-8e07-d60003d223dc

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | slow_yield | maintain,lane_recover,slow_yield,crawl | lane_recover,slow_yield,crawl,nudge_right | None | lane_recover | ok |
| 10 | slow_yield | maintain,lane_recover,slow_yield,crawl | lane_recover,slow_yield,crawl,nudge_right | None | lane_recover | ok |
| 5 | slow_yield | maintain,lane_recover,slow_yield,crawl | -- | actor_action_clearance | None | ok |
| 2 | slow_yield | nudge_right,evasive_right,maintain,lane_recover | -- | actor_action_clearance | None | ok |
| 1 | slow_yield | nudge_right,evasive_right,maintain,crawl | -- | actor_action_clearance | None | ok |
| 0 | slow_yield | -- | -- | actor_action_clearance | None | ok |

### 014_clipgt-7c4f23b7-be14-4505-b332-6d1ceb36cd75

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | slow_yield | crawl | slow_yield,crawl,lane_recover,evasive_right | None | slow_yield | ok |
| 10 | evasive_right | evasive_left,nudge_left,slow_yield,lane_recover | -- | actor_action_clearance | None | ok |
| 5 | evasive_right | nudge_right,evasive_right,lane_recover,slow_yield | -- | actor_horizon_clearance | None | ok |
| 2 | maintain | nudge_right,evasive_right,lane_recover,maintain | -- | actor_horizon_clearance | None | ok |
| 1 | maintain | nudge_right,maintain,evasive_right,lane_recover | -- | actor_horizon_clearance | None | ok |
| 0 | maintain | nudge_right,maintain,evasive_right,lane_recover | -- | actor_horizon_clearance | None | ok |

### 018_clipgt-825dc886-0491-482a-abe3-e1743f5972b8

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | evasive_right | nudge_right,evasive_right,lane_recover,slow_yield | nudge_right,evasive_right,lane_recover,slow_yield | None | nudge_right | ok |
| 10 | evasive_right | nudge_right,evasive_right,lane_recover,slow_yield | -- | actor_horizon_clearance | None | ok |
| 5 | maintain | nudge_right,evasive_right,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 2 | maintain | nudge_right,evasive_right,lane_recover,maintain | -- | actor_action_clearance | None | ok |
| 1 | maintain | nudge_right,evasive_right,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 0 | maintain | -- | -- | actor_action_clearance | None | ok |

### 019_clipgt-994977b5-4db6-43eb-950a-c0faaccc930b

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | evasive_right | nudge_left,evasive_left,lane_recover,slow_yield | nudge_left,evasive_left,lane_recover,slow_yield | actor_horizon_clearance | nudge_left | ok |
| 10 | evasive_right | nudge_left,evasive_left,lane_recover,slow_yield | -- | actor_horizon_clearance | None | ok |
| 5 | maintain | nudge_left,evasive_left,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 2 | maintain | nudge_right,evasive_right,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 1 | maintain | nudge_right,evasive_right,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 0 | maintain | -- | -- | actor_action_clearance | None | ok |

### 020_clipgt-8578f2c7-70a0-4877-980a-7ba65bcded30

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | slow_yield | maintain,lane_recover,slow_yield,crawl | maintain,lane_recover,slow_yield,crawl | None | maintain | ok |
| 10 | slow_yield | nudge_left,evasive_left,crawl,maintain | nudge_left,evasive_left,crawl,maintain | None | nudge_left | ok |
| 5 | slow_yield | nudge_right,evasive_right,crawl,maintain | evasive_left | actor_action_clearance | evasive_left | ok |
| 2 | slow_yield | nudge_right,evasive_right,lane_recover,slow_yield | evasive_left | actor_action_clearance | evasive_left | ok |
| 1 | slow_yield | evasive_right | -- | actor_action_clearance | None | ok |
| 0 | slow_yield | -- | -- | actor_action_clearance | None | ok |

### 021_clipgt-7eaac028-ff62-4f54-9cca-52c8f49ba87e

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | evasive_right | maintain,lane_recover,nudge_left,nudge_right | maintain,lane_recover,nudge_left,nudge_right | None | maintain | ok |
| 10 | evasive_right | maintain,lane_recover,slow_yield,nudge_right | -- | actor_horizon_clearance | None | ok |
| 5 | maintain | nudge_left,evasive_left,lane_recover,slow_yield | -- | actor_action_clearance | None | ok |
| 2 | maintain | nudge_left,evasive_left,lane_recover,maintain | -- | actor_action_clearance | None | ok |
| 1 | maintain | nudge_left,evasive_left,lane_recover,maintain | -- | actor_action_clearance | None | ok |
| 0 | maintain | -- | -- | actor_action_clearance | None | ok |

### 022_clipgt-82da2d46-4ec7-4f8c-8f54-e6febb1c303b

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | slow_yield | nudge_left,evasive_left,maintain,lane_recover | -- | actor_horizon_clearance | None | ok |
| 10 | lane_recover | nudge_left,maintain,evasive_left,lane_recover | -- | actor_horizon_clearance | None | ok |
| 5 | lane_recover | evasive_left,nudge_left,crawl,slow_yield | -- | actor_action_clearance | None | ok |
| 2 | slow_yield | evasive_left,nudge_left | -- | actor_action_clearance | None | ok |
| 1 | slow_yield | evasive_left,nudge_left | -- | actor_action_clearance | None | ok |
| 0 | slow_yield | evasive_left | -- | actor_action_clearance | None | ok |

### 023_clipgt-89c2e887-cffa-4b7c-86cf-80b89be1cb06

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | slow_yield | maintain,lane_recover,slow_yield,crawl | slow_yield,crawl | None | slow_yield | ok |
| 10 | slow_yield | maintain,lane_recover,slow_yield,crawl | lane_recover,slow_yield,crawl,nudge_right | None | lane_recover | ok |
| 5 | slow_yield | nudge_right,evasive_right,crawl,maintain | -- | actor_action_clearance | None | ok |
| 2 | slow_yield | nudge_right,evasive_right,maintain,lane_recover | -- | actor_action_clearance | None | ok |
| 1 | slow_yield | nudge_right,evasive_right,maintain,lane_recover | -- | actor_action_clearance | None | ok |
| 0 | slow_yield | -- | -- | actor_action_clearance | None | ok |

### 025_clipgt-7c05aee0-88f1-4338-a48b-5cbe24d5b888

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | slow_yield | nudge_left,evasive_left,maintain,lane_recover | nudge_left,evasive_left,maintain,nudge_right | actor_horizon_clearance | nudge_left | ok |
| 10 | slow_yield | nudge_right,evasive_right,maintain,lane_recover | -- | actor_horizon_clearance | None | ok |
| 5 | slow_yield | nudge_right,evasive_right,maintain,lane_recover | -- | actor_action_clearance | None | ok |
| 2 | slow_yield | nudge_right,evasive_right,maintain,lane_recover | -- | actor_action_clearance | None | ok |
| 1 | slow_yield | nudge_right,evasive_right,maintain,lane_recover | -- | actor_action_clearance | None | ok |
| 0 | slow_yield | -- | -- | actor_action_clearance | None | ok |

### 030_clipgt-98dd56b2-3e60-49ec-9366-70b610dd93cc

| Lead | Selected | Axis safe | Actor-axis safe | Actor selected violation | Best actor-safe | Status |
| ---: | --- | --- | --- | --- | --- | --- |
| 20 | slow_yield | nudge_left,evasive_left,maintain,lane_recover | nudge_left,evasive_left,maintain,lane_recover | None | nudge_left | ok |
| 10 | slow_yield | nudge_left,evasive_left,maintain,lane_recover | -- | actor_horizon_clearance | None | ok |
| 5 | slow_yield | nudge_left,evasive_left,maintain,lane_recover | -- | actor_action_clearance | None | ok |
| 2 | slow_yield | nudge_left,evasive_left,maintain,lane_recover | -- | actor_action_clearance | None | ok |
| 1 | slow_yield | nudge_left,evasive_left,maintain,lane_recover | -- | actor_action_clearance | None | ok |
| 0 | slow_yield | -- | -- | actor_action_clearance | None | ok |

