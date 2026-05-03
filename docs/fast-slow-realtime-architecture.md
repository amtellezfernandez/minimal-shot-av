# Fast/Slow Realtime Architecture

This proposes the next runtime shape for the submission: a very fast reflex
loop that never waits on rich reasoning, plus a slower deliberative loop that
keeps better plans warm in the background.

## Architecture

### Fast Reflex Loop

- Target rate: 50-100 Hz.
- Synchronous budget: below 2 ms p95 on the current numeric WOD controller.
- Inputs: ego history, route intent, nearest-agent summary, last accepted slow
  plan, and local safety envelope.
- Work per tick:
  - generate compact learned/kinematic candidates,
  - validate the cached slow plan against current kinematics,
  - choose between reflex candidates and the cached slow plan,
  - apply emergency escape, risk nudge, lane recovery, and stop/yield guards.
- Failure behavior: if the slow plan is stale, unsafe, or missing, continue with
  fast local candidates only.

### Slow Deliberative Loop

- Target rate: 5-10 Hz, configurable by `slow_refresh_frames`.
- Asynchronous budget: reported separately; it must not block actuation.
- Inputs: the same numeric state plus heavier temporal/richer candidate models.
- Work per refresh:
  - expand residual and temporal candidates,
  - score them with the contextual ranker,
  - publish one selected candidate as the cached slow plan.
- Failure behavior: slow refresh may be dropped or preempted without stopping
  the fast loop.

### Why This Is More Reactive

The old online benchmark is monolithic: every control tick regenerates and
scores the richer candidate set. The fast/slow architecture reduces the
synchronous tick to four candidates on this benchmark: compact fast candidates
plus one cached slow plan. The richer candidate search still exists, but it
runs at a lower rate and publishes only a small control-path contract.

## Validation

Benchmark command:

```bash
uv run --no-sync python scripts/benchmark_wod_runtime.py \
  --architecture fast-slow \
  --frames 5000 \
  --warmup 200 \
  --residual-modes 3 \
  --fast-residual-modes 1 \
  --slow-refresh-frames 5 \
  --target-ms 14 \
  --output benchmarks/current/wod_fast_slow_runtime.json
```

Same-settings monolithic reference:

```bash
uv run --no-sync python scripts/benchmark_wod_runtime.py \
  --architecture monolithic \
  --frames 5000 \
  --warmup 200 \
  --residual-modes 3 \
  --target-ms 14 \
  --output benchmarks/current/wod_monolithic_runtime_reference.json
```

Measured on the current local CPU environment:

| Runtime | Sync candidates | p95 sync latency | Mean sync latency | Throughput |
| --- | ---: | ---: | ---: | ---: |
| Monolithic reference | 14 | `2.124 ms` | `1.949 ms` | `494 fps` |
| Fast/slow reflex | 4 | `0.625 ms` | `0.558 ms` | `1047 fps` |

Slow refresh is not hidden: its p95 latency is `2.132 ms` and mean latency is
`1.933 ms`, recorded as `async_slow_refresh_ms`. This makes the proposal
stronger for realtime deployment because the actuation loop is bounded even
when the richer candidate refresh is late.

## Current Limitations

- This benchmark validates runtime architecture, not WOD-E2E score improvement.
- It uses synthetic WOD-like frames and local numeric models, matching the
  existing runtime benchmark contract.
- The cached slow plan still needs a stronger semantic validity check before it
  should be claimed as a deployed AV architecture.
- The next quality experiment should combine this scheduler with the current
  `7.657` official validation-CV WOD selector and verify that score does not
  regress while control latency improves.
