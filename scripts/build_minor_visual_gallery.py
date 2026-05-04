#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_ROOT = ROOT / "artifacts" / "sota_submission_bundles"

DEMO_DIRS = (
    "minor_construction_seed1",
    "minor_fod_seed2",
    "minor_spotlight_seed3",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a browser gallery for the Minor submission visuals.")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BUNDLE_ROOT / "minor_visual_gallery")
    args = parser.parse_args()

    output = build_gallery(args.bundle_root, args.output_dir)
    print(f"Wrote {output}")
    return 0


def build_gallery(bundle_root: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "index.html"
    demos = [_demo_summary(bundle_root / demo_dir) for demo_dir in DEMO_DIRS]
    wod_eval = _evaluation_summary(bundle_root / "minor_eval" / "scenario_eval.json")
    ood_eval = _evaluation_summary(bundle_root / "minor_ood_eval" / "scenario_eval.json")
    alpasignal = _json_or_empty(bundle_root / "minor_alpasignal_bridge" / "alpasignal_bridge_audit.json")
    runtime = _json_or_empty(bundle_root / "minor_runtime" / "minor_runtime_constraints.json")
    output.write_text(_html(demos, wod_eval, ood_eval, alpasignal, runtime), encoding="utf-8")
    return output


def _demo_summary(path: Path) -> dict[str, Any]:
    payload = _json_or_empty(path / "latest_rollout.json")
    rollout = payload.get("rollout", {})
    scenario = payload.get("scenario", {})
    steps = rollout.get("steps", [])
    clearances = [
        float(step["min_obstacle_distance"])
        for step in steps
        if isinstance(step, dict) and step.get("min_obstacle_distance") is not None
    ]
    return {
        "name": path.name.replace("minor_", "").replace("_", " "),
        "cluster": scenario.get("cluster", "unknown"),
        "seed": scenario.get("seed", "unknown"),
        "success": bool(rollout.get("success")),
        "collision": bool(rollout.get("collision")),
        "steps": len(steps),
        "min_clearance": min(clearances) if clearances else None,
        "svg": f"../{path.name}/latest_rollout.svg",
        "json": f"../{path.name}/latest_rollout.json",
    }


def _evaluation_summary(path: Path) -> dict[str, Any]:
    payload = _json_or_empty(path)
    runs = payload.get("runs", [])
    if not runs:
        return {"run_count": 0}
    return {
        "run_count": len(runs),
        "success_count": sum(1 for row in runs if row.get("success")),
        "benchmark_pass_count": sum(1 for row in runs if row.get("benchmark_pass")),
        "collision_count": sum(1 for row in runs if row.get("collision")),
        "suite_count": len({str(row.get("suite")) for row in runs}),
        "cluster_count": len({str(row.get("cluster")) for row in runs}),
    }


def _json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _html(
    demos: list[dict[str, Any]],
    wod_eval: dict[str, Any],
    ood_eval: dict[str, Any],
    alpasignal: dict[str, Any],
    runtime: dict[str, Any],
) -> str:
    demo_cards = "\n".join(_demo_card(demo) for demo in demos)
    evidence_cards = "\n".join(
        [
            _metric_card(
                "Randomized WOD Sweep",
                [
                    ("runs", wod_eval.get("run_count")),
                    ("successes", wod_eval.get("success_count")),
                    ("benchmark passes", wod_eval.get("benchmark_pass_count")),
                    ("collisions", wod_eval.get("collision_count")),
                    ("clusters", wod_eval.get("cluster_count")),
                ],
            ),
            _metric_card(
                "Compositional OOD Sweep",
                [
                    ("runs", ood_eval.get("run_count")),
                    ("successes", ood_eval.get("success_count")),
                    ("benchmark passes", ood_eval.get("benchmark_pass_count")),
                    ("collisions", ood_eval.get("collision_count")),
                    ("suites", ood_eval.get("suite_count")),
                ],
            ),
            _metric_card(
                "AlpaSignal Bridge",
                [
                    ("valid", alpasignal.get("valid")),
                    ("cases", len(alpasignal.get("cases", []))),
                    ("adapter", alpasignal.get("adapter", "")),
                ],
            ),
            _metric_card(
                "Runtime Constraints",
                [
                    ("valid", runtime.get("valid")),
                    ("rollouts", runtime.get("run_count")),
                    ("steps", runtime.get("total_steps")),
                    ("p95 step ms", _round(runtime.get("p95_step_latency_ms"))),
                    ("target step ms", runtime.get("target_step_ms")),
                    ("steps/s", _round(runtime.get("throughput_steps_per_second"))),
                ],
            ),
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SoTA Minor Visual Gallery</title>
  <style>
    :root {{
      color-scheme: light;
      --ink: #1f2933;
      --muted: #52606d;
      --line: #d9e2ec;
      --panel: #ffffff;
      --paper: #f4f7fb;
      --accent: #0f766e;
      --warn: #b45309;
    }}
    body {{
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--paper);
      color: var(--ink);
    }}
    header {{
      padding: 28px 36px 20px;
      background: #0b1f33;
      color: white;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 30px;
      letter-spacing: 0;
    }}
    header p {{
      margin: 0;
      color: #dbeafe;
      max-width: 900px;
      line-height: 1.45;
    }}
    main {{
      padding: 24px 36px 36px;
    }}
    h2 {{
      margin: 10px 0 14px;
      font-size: 20px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
    }}
    .card-body {{
      padding: 14px 16px 16px;
    }}
    .viz {{
      display: block;
      width: 100%;
      aspect-ratio: 3 / 2;
      border: 0;
      background: #f5f1e8;
    }}
    .title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
    }}
    .title h3 {{
      margin: 0;
      font-size: 16px;
      text-transform: capitalize;
    }}
    .badge {{
      font-size: 12px;
      padding: 3px 8px;
      border-radius: 999px;
      background: #ccfbf1;
      color: #115e59;
      white-space: nowrap;
    }}
    .badge.warn {{
      background: #ffedd5;
      color: var(--warn);
    }}
    dl {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 7px 12px;
      margin: 0;
      font-size: 13px;
    }}
    dt {{
      color: var(--muted);
    }}
    dd {{
      margin: 0;
      font-weight: 650;
      text-align: right;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 650;
    }}
    .links {{
      margin-top: 12px;
      display: flex;
      gap: 12px;
      font-size: 13px;
    }}
    @media (max-width: 700px) {{
      header, main {{
        padding-left: 16px;
        padding-right: 16px;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>SoTA Minor Visual Gallery</h1>
    <p>Closed-loop procedural simulation evidence for minimal-shot autonomy:
    navigation rollouts, randomized scenario sweeps, compositional OOD stress,
    AlpaSignal bridge checks, and explicit runtime constraints.</p>
  </header>
  <main>
    <h2>Navigation Demos</h2>
    <section class="grid">{demo_cards}</section>
    <h2>Evidence Summary</h2>
    <section class="grid">{evidence_cards}</section>
  </main>
</body>
</html>
"""


def _demo_card(demo: dict[str, Any]) -> str:
    status = "success" if demo["success"] and not demo["collision"] else "review"
    badge_class = "badge" if status == "success" else "badge warn"
    clearance = _round(demo.get("min_clearance"))
    return f"""
      <article class="card">
        <object class="viz" type="image/svg+xml" data="{html.escape(str(demo['svg']))}"></object>
        <div class="card-body">
          <div class="title">
            <h3>{html.escape(str(demo['cluster']))}</h3>
            <span class="{badge_class}">{status}</span>
          </div>
          <dl>
            <dt>seed</dt><dd>{html.escape(str(demo['seed']))}</dd>
            <dt>steps</dt><dd>{demo['steps']}</dd>
            <dt>minimum clearance</dt><dd>{clearance} m</dd>
            <dt>collision</dt><dd>{demo['collision']}</dd>
          </dl>
          <div class="links">
            <a href="{html.escape(str(demo['svg']))}">open SVG</a>
            <a href="{html.escape(str(demo['json']))}">open JSON</a>
          </div>
        </div>
      </article>
    """


def _metric_card(title: str, rows: list[tuple[str, Any]]) -> str:
    items = "\n".join(f"<dt>{html.escape(label)}</dt><dd>{html.escape(str(value))}</dd>" for label, value in rows)
    return f"""
      <article class="card">
        <div class="card-body">
          <div class="title"><h3>{html.escape(title)}</h3></div>
          <dl>{items}</dl>
        </div>
      </article>
    """


def _round(value: Any) -> Any:
    return round(float(value), 3) if isinstance(value, int | float) else value


if __name__ == "__main__":
    raise SystemExit(main())
