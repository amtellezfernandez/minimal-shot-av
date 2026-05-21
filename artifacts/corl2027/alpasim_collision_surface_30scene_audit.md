# AlpaSim 30-scene collision-surface audit

Baseline: `axis_constrained_clamped`
Candidate: `world_frame_oracle_actor_completion`

## Summary

- Common scenes: `30`.
- Raw collision scenes: baseline `18`, candidate `18`.
- Paired collision change: `baseline_only=0`, `candidate_only=0`, `same=30/30`.
- Baseline first-impact selection: `top_candidate_maintain=18/18`, `structured_hazards_zero=18/18`.
- Baseline first-impact components: `collision_front=3`, `collision_lateral=1`, `collision_rear=14`.
- Candidate first-impact selection where logs exist: `logged=17/18`, `proxy_hits=17/17`, `top_not_maintain=17/17`, `selected_not_maintain=17/17`.
- Candidate rear-lane hazards at logged rear impacts: `14/14`.
- Collision timing delta candidate-baseline: median `0.100s`, mean `0.228s`, range `[-0.100, 1.500]s`.
- Controller speed delta at first impact: median `0.087 m/s`, mean `-0.089 m/s`; candidate slower on `6/18` paired impacts.

## Interpretation

The matched oracle actor run changes the selector state and selected action at impact, but it does not change the collision scene set. In the baseline, first-impact frames look like clear corridor frames to the scorer. In the oracle run, actors are visible, rear-lane hazards are present for logged rear impacts, and the selector usually moves to nudge/evasive/stop actions. The remaining collision surface is therefore not explained by actor visibility or action ranking alone. The unresolved split is between candidate-set coverage and controller/traffic execution; resolving it requires counterfactual candidate/controller replay.

## First-impact scene table

| Scene | Baseline impact | Candidate impact | Baseline sel/top/haz | Candidate sel/top/haz/proxy | dt (s) |
| --- | --- | --- | --- | --- | ---: |
| 001_clipgt-a309e228-26e1-423e-a44c-cb00aa7378cb | f37 collision_rear | f37 collision_rear | maintain/maintain/0 | evasive_left/nudge_left/24/True | 0.000 |
| 002_clipgt-804afc4a-fd1e-4f58-bd39-a4c486a916e5 | f43 collision_rear | f47 collision_rear | evasive_right/maintain/0 | nudge_left/nudge_left/22/True | 0.400 |
| 004_clipgt-9b3509c2-a5fa-42c0-b0e5-feb617e9bf4e | f51 collision_rear | f53 collision_rear | maintain/maintain/0 | nudge_right/nudge_right/18/True | 0.200 |
| 005_clipgt-90d1908c-9fdc-40ea-a5a1-351240aa323e | f30 collision_front | f45 collision_rear | evasive_right/maintain/0 | nudge_right/nudge_right/16/True | 1.500 |
| 007_clipgt-8aaa1135-573f-4f6c-8ff8-7ac6bd6ea5d6 | f40 collision_rear | f42 collision_rear | evasive_right/maintain/0 | evasive_left/nudge_left/24/True | 0.200 |
| 008_clipgt-9f8acd00-e8f5-467a-867d-b205f8d4d75d | f54 collision_rear | f58 collision_rear | lane_recover/maintain/0 | nudge_left/nudge_left/19/True | 0.400 |
| 009_clipgt-97a2e621-a27b-4b87-9a58-b6b740fc1b4e | f49 collision_rear | f49 collision_rear | lane_recover/maintain/0 | evasive_left/nudge_left/24/True | 0.000 |
| 011_clipgt-9b6a51bd-cb54-4a68-8ed8-0e33a504c67e | f29 collision_front | f28 collision_front | lane_recover/maintain/0 | evasive_right/evasive_right/21/True | -0.100 |
| 013_clipgt-89f2eeb8-f4e4-42a1-8e07-d60003d223dc | f94 collision_rear | f94 collision_rear | slow_yield/maintain/0 | None/None/None/None | 0.000 |
| 014_clipgt-7c4f23b7-be14-4505-b332-6d1ceb36cd75 | f28 collision_front | f28 collision_front | maintain/maintain/0 | evasive_right/evasive_right/10/True | 0.000 |
| 018_clipgt-825dc886-0491-482a-abe3-e1743f5972b8 | f45 collision_rear | f45 collision_rear | maintain/maintain/0 | evasive_right/nudge_right/24/True | 0.000 |
| 019_clipgt-994977b5-4db6-43eb-950a-c0faaccc930b | f44 collision_rear | f44 collision_rear | maintain/maintain/0 | evasive_left/nudge_left/18/True | 0.000 |
| 020_clipgt-8578f2c7-70a0-4877-980a-7ba65bcded30 | f173 collision_lateral | f173 collision_lateral | slow_yield/maintain/0 | stop/evasive_left/14/True | 0.000 |
| 021_clipgt-7eaac028-ff62-4f54-9cca-52c8f49ba87e | f50 collision_rear | f54 collision_rear | maintain/maintain/0 | nudge_left/nudge_left/10/True | 0.400 |
| 022_clipgt-82da2d46-4ec7-4f8c-8f54-e6febb1c303b | f32 collision_rear | f33 collision_rear | slow_yield/maintain/0 | evasive_left/evasive_left/24/True | 0.100 |
| 023_clipgt-89c2e887-cffa-4b7c-86cf-80b89be1cb06 | f142 collision_rear | f150 collision_rear | slow_yield/maintain/0 | evasive_left/nudge_left/22/True | 0.800 |
| 025_clipgt-7c05aee0-88f1-4338-a48b-5cbe24d5b888 | f123 collision_rear | f124 collision_rear | slow_yield/maintain/0 | nudge_right/nudge_right/23/True | 0.100 |
| 030_clipgt-98dd56b2-3e60-49ec-9366-70b610dd93cc | f53 collision_rear | f54 collision_rear | slow_yield/maintain/0 | nudge_left/nudge_left/22/True | 0.100 |
