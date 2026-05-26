# NAVSIM Zero-Perception Ridge Probe

This is the paper-safe probe for the NAVSIM candidate-bank audit. It receives no
camera pixels, map rasters, hidden VLA states, scene tokens, or candidate-order
cues. The feature vector has 35 scalar inputs: 8 prompt kinematics, 23 candidate
trajectory geometry features, and 4 decode metadata features.

## Prompt kinematics

These are parsed from the OneVL text prompt.

- `acc_x`
- `acc_y`
- `cmd_forward`
- `hist_h`
- `hist_x`
- `hist_y`
- `vel_x`
- `vel_y`

## Candidate trajectory geometry

These are computed from the generated trajectory only.

- `backward_steps`
- `final_h`
- `final_x`
- `final_y`
- `heading_change`
- `is_len8`
- `is_padded`
- `max_abs_acc2`
- `max_abs_heading_rate`
- `max_abs_jerk`
- `max_abs_y`
- `max_curv`
- `max_step`
- `mean_abs_heading_rate`
- `mean_abs_y`
- `mean_acc2`
- `mean_curv`
- `mean_step`
- `min_step`
- `orig_len`
- `path_len`
- `progress_ratio`
- `std_step`

## Decode metadata

These are scalar statistics from token generation, not visual features.

- `avg_entropy`
- `avg_log_prob`
- `logprob_finite`
- `seq_confidence`

## Excluded order cues

The full engineering probe can optionally include these two features, but the
Paper 1 zero-perception baseline must exclude them with `--drop-order-features`.

- `candidate_id`
- `source_top1`
