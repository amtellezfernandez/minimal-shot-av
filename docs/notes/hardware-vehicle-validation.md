# Hardware / Vehicle Validation Start

This repository now separates three claims:

- **Production AV readiness:** still blocked by `scripts/audit_production_av_readiness.py`.
- **Controlled HIL/VIL start readiness:** checked by `scripts/audit_hardware_vehicle_validation.py`.
- **Public-road deployment:** explicitly not authorized.

Run the controlled-validation gate:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_vehicle_validation_shadow.py
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/audit_hardware_vehicle_validation.py --hil-check
```

The default evidence bundle is:

- `artifacts/compass_evidence_report_sotif_seed381.json`
- `artifacts/av_stack_integration_manifest.json`
- `artifacts/safety_case.json`
- `configs/hardware_vehicle_validation_plan.json`
- `artifacts/vehicle_validation_shadow_report.json`

The audit can pass only when:

- strict `sotif-v0` simulator evidence has no missing ranked runs;
- perception, prediction, planning, control, safety monitor, fault injection,
  closed-loop interface, and runtime budget are validated in the simulator harness;
- safety-case sections exist;
- the hardware plan approves only HIL and supervised closed-course validation;
- physical estop, remote estop, safety driver, speed limit, geofence, manual
  takeover, shadow mode, command logging, and rollback controls are present;
- a shadow command replay exists with zero command-envelope violations;
- production/public-road control remains blocked.

Allowed validation order:

1. **Bench HIL:** send commands through the real hardware interface with wheels
   off ground or dyno-equivalent loading. Verify timing, command ranges, logging,
   estop, manual takeover, and rollback.
2. **Closed-course shadow:** keep the vehicle manually driven and compare shadow
   commands against speed, steering, braking, and geofence limits.
3. **Closed-course low-speed actuation:** allow limited actuation only with a
   safety driver, physical estop, remote estop, geofence, and enforced speed cap.

This is not a production release. Any new hardware evidence should be written as
an artifact and then used as an input to a stricter follow-up audit before
raising speed, expanding the ODD, or leaving a closed course.
