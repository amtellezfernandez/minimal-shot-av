# Submission Index

There are three separate tracks. Do not merge the claims.

## 1. SoTA Grand Commission

Upload / attach:

- `artifacts/sota_submission_bundles/grand_commission.tar.gz`
- GitHub repository URL
- Video or slide deck based on `docs/sota-grand-slide-script.md`

SHA-256: `64b6d227ceedb101479f3285428c03488b87e65c103ff4e6cc31180e5889918d`
Git commit: `173f80633ee60997795569a90ca5ec4688c702e0`

Claim: minimal-shot autonomy architecture prototype with closed-loop demos.

## 2. SoTA Minor Commission

Upload / attach:

- `artifacts/sota_submission_bundles/minor_commission.tar.gz`
- GitHub repository URL
- Video or slide deck based on `docs/sota-minor-slide-script.md`

SHA-256: `14bb3510ad3533b14a10d6d35a87cb2f41857cc32d2bdfe283c5b58e8273f2ad`
Git commit: `173f80633ee60997795569a90ca5ec4688c702e0`

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
