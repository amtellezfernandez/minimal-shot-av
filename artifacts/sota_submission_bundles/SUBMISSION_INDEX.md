# Submission Index

There are three separate tracks. Do not merge the claims.

## 1. SoTA Grand Commission

Upload / attach:

- `artifacts/sota_submission_bundles/grand_commission.tar.gz`
- GitHub repository URL
- Video or slide deck based on `docs/sota-grand-slide-script.md`

SHA-256: `79c5f7743e2d61e1727be1a8cc11cc7e203479735ef6f7616de3e883ce766391`
Git commit: `d23712195b64a47240ba19c8bde74563fe0fa33f`

Claim: minimal-shot autonomy architecture prototype with closed-loop demos.

## 2. SoTA Minor Commission

Upload / attach:

- `artifacts/sota_submission_bundles/minor_commission.tar.gz`
- GitHub repository URL
- Video or slide deck based on `docs/sota-minor-slide-script.md`

SHA-256: `a6f6f2e7e88c66ba64638f32543bbafac830cb13e7dc6e4f72d4ce64a7d582c4`
Git commit: `d23712195b64a47240ba19c8bde74563fe0fa33f`

Claim: randomized long-tail simulation environment with closed-loop evaluation.

## 3. Waymo WOD-E2E Challenge

Separate benchmark track. Do not present it as a SoTA submission artifact.

Current status:

- WOD-E2E test split missing locally
- no confirmed hidden-test leaderboard result recorded

First restore the missing train/test shards and official frame list:

`python3 scripts/prepare_wod_e2e_data.py --data-root /home/amdev/waymo --output artifacts/wod_e2e_data_restore_plan.json`

When ready, generate pre-registered official Waymo tarballs through the two-gate wrapper:

`PYTHONPATH=src:.wod-protos .venv-wod/bin/python scripts/run_wod_leaderboard_attack.py --data-root /home/amdev/waymo --frame-list data/waymo/e2e/submission_frames/test_frames.json --output-dir artifacts/wod_e2e_submission_matrix --account-name <account> --authors 'Alba Maria Tellez Fernandez'`
