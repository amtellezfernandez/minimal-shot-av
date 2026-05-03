# Preference-Calibrated Evidence

This records the strongest `7.8+` preference-calibrated result found so far and
the boundary that prevents it from becoming the headline WOD claim.

## Achieved Signal

- Local-RFS neural preference-calibrated heldout report:
  `artifacts/wod_neural_holdout/neural_ensemble3_sourcegate_speedfine_p0_local.json`
- Split: 159 heldout validation-preference frames.
- Selected local RFS: `7.838`.
- Candidate oracle local RFS: `9.212`.

There is also an official-scored heldout report with normalized RFS above `7.8`:

- Official-RFS heldout report:
  `artifacts/wod_neural_holdout/neural_top1_pc0_familycal_l2_010_heldout_official.json`
- Selected official mean RFS: `7.628`.
- Selected official normalized RFS: `7.894`.
- Candidate oracle official mean RFS: `9.145`.

## Failed Promotion Check

The same neural preference-calibrated family was rerun on the full retained
479-frame validation cache with official RFS:

- Full retained-validation official report:
  `artifacts/wod_preference_calibrated_ensemble3_full_official.json`
- Selected official mean RFS: `7.262`.
- Selected official normalized RFS: `7.566`.
- Candidate oracle official mean RFS: `9.225`.

This is not promoted. The selector over-trusts learned candidates on the full
validation cache and loses to the current structured fallback report
(`7.657` official mean RFS). The useful result is that the preference-calibrated
candidate family has real headroom, but the gating policy is not stable enough
for the main Grand claim.

## Submission Use

Use this as a secondary evidence point only:

- Good: "preference-calibrated heldout experiments reached `7.838` local RFS,
  and official normalized heldout reached `7.894`, but the full official
  validation rerun did not promote."
- Bad: "the WOD model reached `7.8+` official mean RFS."
