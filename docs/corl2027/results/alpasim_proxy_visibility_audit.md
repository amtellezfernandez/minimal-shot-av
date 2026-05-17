| Variant | Frames | Structured-hazard rate | Moving-hazard rate | Clear top-candidate rate | Veto rate | Collision | Offroad | Wrong lane |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Axis-constrained clamped | 1990 | 0.000 | 0.000 | 0.870 | 0.040 | 0.700 | 0.200 | 0.200 |
| Hard veto hybrid | 1990 | 0.000 | 0.000 | 0.907 | 0.998 | 0.800 | 0.200 | 0.700 |

Proxy-visibility audit from the matched 10-clip AlpaSim matrix. Both selectors had access
to the same structured-hazard interface, but the front-camera transfer runs provided no
structured hazards in the selection logs. In most frames the geometric scorer's top
candidate was therefore interpreted as a clear corridor. This explains why clamping and
axis constraints repair route geometry while failing to reduce collision: the selector is
ranking candidates against an actor-incomplete proxy state.

Reproduce:

```bash
./.venv/bin/python scripts/audit_alpasim_transfer_diagnostic.py \
  runs/alpasim_transfer_matrix_run_10scene_with_axis/token_dagger_iter2_axis_constrained_clamped__front_camera_10scene_smoke/*_clipgt-* \
  --format markdown --output artifacts/alpasim_axis_proxy_visibility_audit.json

./.venv/bin/python scripts/audit_alpasim_transfer_diagnostic.py \
  runs/alpasim_transfer_matrix_run_10scene_with_axis/token_dagger_iter2_hybrid_clamped__front_camera_10scene_smoke/*_clipgt-* \
  --format markdown --output artifacts/alpasim_hybrid_proxy_visibility_audit.json
```
