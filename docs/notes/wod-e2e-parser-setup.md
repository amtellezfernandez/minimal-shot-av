# WOD-E2E Official Parser Setup

This note records what was done to make the official Waymo E2E parser usable in
this workspace, and what the current validation check proves.

## Goal

The immediate goal is to replace the current synthetic RFS references with real
Waymo validation `preference_trajectories`.

That requires parsing downloaded WOD-E2E validation TFRecords as official
`E2EDFrame` protos, then extracting:

- `frame.context.name`
- `future_states`
- `past_states`
- `intent`
- valid `preference_trajectories`
- each preference trajectory's `preference_score`

## Official Sources

The official parser contract comes from the local Waymo checkout:

- `workspace/waymo-open-dataset/tutorial/tutorial_vision_based_e2e_driving.ipynb`
- `workspace/waymo-open-dataset/src/waymo_open_dataset/protos/end_to_end_driving_data.proto`
- `workspace/waymo-open-dataset/src/waymo_open_dataset/protos/end_to_end_driving_submission.proto`
- `workspace/waymo-open-dataset/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py`

The official parsing pattern is:

```python
import tensorflow as tf
from waymo_open_dataset.protos import end_to_end_driving_data_pb2 as wod_e2ed_pb2

dataset = tf.data.TFRecordDataset(filenames, compression_type="")

frame = wod_e2ed_pb2.E2EDFrame()
frame.ParseFromString(next(iter(dataset)).numpy())
```

## Local Setup

The active system Python did not have TensorFlow, protobuf, pip, or venv support.
To avoid changing system Python, a local `uv` environment was created:

```bash
UV_CACHE_DIR=.uv-cache uv venv .venv-wod --python /usr/bin/python3.12
UV_CACHE_DIR=.uv-cache uv pip install --python .venv-wod/bin/python \
  protobuf grpcio-tools numpy tensorflow-cpu
```

The Waymo repo provides `.proto` files, but not generated Python `*_pb2.py`
modules in this checkout. Those were generated into `.wod-protos/` from the
official proto sources. Both `.venv-wod/` and `.wod-protos/` are local generated
artifacts and are ignored by git.

## Validation Data

The WOD-E2E validation split is downloaded under:

```text
workspace/waymo_open_dataset_end_to_end_camera_v_1_0_0/val/
```

Current verified state:

```text
validation_shards=93
local_size=226G
download_transfer_size=225.2 GiB
```

The files are ignored by git.

## Smoke Test

The reusable checker is:

```bash
PYTHONPATH=.wod-protos .venv-wod/bin/python scripts/check_wod_e2e_parser.py
```

Observed output:

```text
tensorflow=2.21.0
parsed=val_202504211843.tfrecord-00000-of-00093
record_in_shard=594
records_seen=594
context_name=d5fa64f6a8c44c60aaec7b07776e23c4-150
preference_count=3 valid_count=3
first_score=10.000
first_preference_len=(21,21)
future_states_len=(20,20)
```

This proves the local parser can read real downloaded validation records through
the official TensorFlow/protobuf path, and that the validation set includes real
human-rated preference trajectories.

## Loader and Scoring Path

The reusable loader is:

```text
src/minimal_shot_av/model/wod_e2e.py
```

It lazily imports TensorFlow and the generated official Waymo protos, so normal
simulator tests do not require the WOD parser environment. Run WOD-specific
commands with:

```bash
PYTHONPATH=.wod-protos .venv-wod/bin/python ...
```

The official-RFS validation scoring CLI is:

```bash
PYTHONPATH=.wod-protos .venv-wod/bin/python \
  scripts/evaluate_wod_e2e_rfs.py --max-shards 5 --max-preference-frames 5
```

Observed sample:

```text
0001 frame=d5fa64f6a8c44c60aaec7b07776e23c4-150 refs=3 init_speed=0.004 log_future_rfs=4.794
0002 frame=4b390f75983c892cbb4144d5af260054-148 refs=3 init_speed=0.065 log_future_rfs=8.000
0003 frame=a72707e8b188cea7d5724aaf76dda5f3-147 refs=3 init_speed=1.223 log_future_rfs=7.126
0004 frame=88ae24affc3070ff2380eb79ca250045-149 refs=3 init_speed=0.000 log_future_rfs=6.000
0005 frame=bc0d724d46340fc33504ae9f24b9df55-148 refs=3 init_speed=5.697 log_future_rfs=10.000
summary frames=5 mean_log_future_rfs=7.184 min=4.794 max=10.000
```

The scorer loads Waymo's official
`workspace/waymo-open-dataset/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py`
by file path to avoid a package path conflict between the source checkout and
the generated `.wod-protos` modules.

## Important Alignment Caveat

`future_states` has 20 future points, matching the benchmark submission horizon.
The first valid `preference_trajectory` observed has 21 x/y points.

Do not blindly pass raw preference trajectories into an RFS implementation that
expects `(20, 2)`. The loader should explicitly align the rater trajectory to
the prediction horizon before scoring.

The official `rater_feedback_utils.process_rater_specified_trajectories`
truncates rater trajectories longer than the target waypoint count and pads
shorter trajectories by repeating the final waypoint. The local loader mirrors
that behavior rather than applying an inferred anchor-point shift.

## Next Implementation Step

The first implementation step is now complete:

1. Reads validation shards with `tf.data.TFRecordDataset`.
2. Parses each record into `wod_e2ed_pb2.E2EDFrame`.
3. Skips frames with no valid preference scores or score `-1`.
4. Converts valid rater trajectories into official-metric-aligned `(20, 2)`
   arrays.
5. Returns `(frame_name, past_states, intent, future_states, references)`.

The next step is to train or fit the declared preference-calibrated verifier:

1. Generate candidate trajectories for each validation frame.
2. Score candidates against official RFS using real `preference_trajectories`.
3. Fit a lightweight ranker/verifier to predict candidate RFS from trajectory
   and scene/ego features.
4. Use the verifier only for candidate selection, not for AV-policy fine-tuning.

## Preference Ranker Baseline

The first ranker baseline is intentionally small and trajectory-feature only:

```bash
PYTHONPATH=.wod-protos .venv-wod/bin/python \
  scripts/build_wod_preference_dataset.py \
  --output artifacts/wod_preference_candidates.full.jsonl

.venv-wod/bin/python scripts/train_wod_preference_ranker.py \
  --input artifacts/wod_preference_candidates.full.jsonl \
  --output artifacts/wod_preference_ranker.full.json \
  --test-fraction 0.2 \
  --ridge 1.0 \
  --seed 17 \
  --folds 5
```

Observed full validation candidate dataset:

```text
frames=479
rows=12079
mean_logged=8.175
mean_oracle_best=9.192
```

Observed segment-grouped 5-fold ridge-ranker result:

```text
cv folds=5 frames=479 selected_mean=8.226 logged_mean=8.175 oracle_mean=9.192 top1=0.639
train frames=383 selected_mean=8.171 logged_mean=8.119 oracle_mean=9.139 top1=0.634
test frames=96 selected_mean=8.668 logged_mean=8.399 oracle_mean=9.401 top1=0.677
```

The full validation result proves the pipeline can produce exact-RFS supervised
labels across all 479 preference-labeled validation frames. The candidate set
has large oracle headroom, but the linear ranker only recovers a small part of
it. The next model bottleneck is ranker capacity, not candidate availability.

This result uses a segment-grouped, seed-stable split and does not use true RFS
as an evaluation tie-breaker. Earlier contaminated numbers should be ignored.

The saved ranker can be loaded at runtime through:

```text
src/minimal_shot_av/model/wod_ranker.py
```

Evaluate a saved ranker on any candidate JSONL file with:

```bash
.venv-wod/bin/python scripts/evaluate_wod_preference_ranker.py \
  --input artifacts/wod_preference_candidates.full.jsonl \
  --ranker artifacts/wod_preference_ranker.full.json
```

This evaluator reports selected, logged, oracle, regret, and top-1 oracle-match
rates without using true RFS during selection.

After that change, validation scoring can be described as exact WOD-E2E RFS
when using the official `preference_trajectories` and official metric contract.
