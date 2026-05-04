# Submission Index

There are three separate tracks. Do not merge the claims.

## 1. SoTA Grand Commission

Upload / attach:

- `artifacts/sota_submission_bundles/grand_commission.tar.gz`
- GitHub repository URL
- Video or slide deck based on `docs/sota-grand-slide-script.md`

SHA-256: `9ca484525281d2c5340c4214b68fc4da6f5f10644cfc1de0bca325f0c8e40e3c`
Git commit: `8d3a46391f20e756084af915ab5a652c7e8458c0`

Claim: minimal-shot autonomy architecture prototype with closed-loop demos.

## 2. SoTA Minor Commission

Upload / attach:

- `artifacts/sota_submission_bundles/minor_commission.tar.gz`
- GitHub repository URL
- Video or slide deck based on `docs/sota-minor-slide-script.md`

SHA-256: `7a51792d320612ae49072b1f059b919b0ebc588d70483e1fbf63901ae84d94ae`
Git commit: `8d3a46391f20e756084af915ab5a652c7e8458c0`

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
