# Submission Index

There are three separate tracks. Do not merge the claims.

## 1. SoTA Grand Commission

Upload / attach:

- `artifacts/sota_submission_bundles/grand_commission.tar.gz`
- GitHub repository URL
- Video or slide deck based on `docs/sota-grand-slide-script.md`

SHA-256: `6876fcfafc4f915170e88502c8cbfdce38fa96e71fc65100c9a0486032c68dbf`

Claim: minimal-shot autonomy architecture prototype with closed-loop demos.

## 2. SoTA Minor Commission

Upload / attach:

- `artifacts/sota_submission_bundles/minor_commission.tar.gz`
- GitHub repository URL
- Video or slide deck based on `docs/sota-minor-slide-script.md`

SHA-256: `ca5ced7c167b75afaf38112d2ae59c657e830ec09bdeda4cfcbfd3f0bf1ecc24`

Claim: randomized long-tail simulation environment with closed-loop evaluation.

## 3. Waymo WOD-E2E Challenge

Separate benchmark track. Do not present it as a SoTA submission artifact.

Current status:

- WOD-E2E test split missing locally
- no confirmed hidden-test leaderboard result recorded

When ready, generate official Waymo tarballs with:

`PYTHONPATH=src:.wod-protos .venv-wod/bin/python scripts/prepare_wod_e2e_submission_matrix.py --test-dir waymo_open_dataset_end_to_end_camera_v_1_0_0/test --output-dir artifacts/wod_e2e_submission_matrix --account-name <account> --authors 'Alba Maria Tellez Fernandez'`
