#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def mean(values):
    return sum(values) / len(values)


def std(values):
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def winner_not_top1_rate(summary):
    rows = summary.get("per_scene", [])
    if not rows:
        return 0.0
    return sum(1 for row in rows if int(row["oracle_candidate"]) != 0) / len(rows)


def summarize_k(sweep_dir: Path, k: int, seeds: list[int]):
    rows = []
    if k == 1:
        seed0 = next(sweep_dir.glob("holdout100_hybrid_k*_seed*_t08_p095_oracle.json"), None)
        if seed0 is None:
            return None
        top1 = float(load_json(seed0)["top1_score"])
        return {
            "k": 1,
            "seed_count": 1,
            "top1_mean": top1,
            "top1_std": 0.0,
            "oracle_mean": top1,
            "oracle_std": 0.0,
            "oracle_gap_mean": 0.0,
            "oracle_gap_std": 0.0,
            "winner_not_top1_rate_mean": 0.0,
            "winner_not_top1_rate_std": 0.0,
        }

    for seed in seeds:
        path = sweep_dir / f"holdout100_hybrid_k{k}_seed{seed}_t08_p095_oracle.json"
        if not path.exists():
            continue
        summary = load_json(path)
        oracle = float(summary.get("oracle_at_k_score", summary.get("oracle_at_4_score")))
        top1 = float(summary["top1_score"])
        rows.append(
            {
                "top1": top1,
                "oracle": oracle,
                "gap": oracle - top1,
                "winner_not_top1_rate": winner_not_top1_rate(summary),
            }
        )
    if not rows:
        return None
    return {
        "k": k,
        "seed_count": len(rows),
        "top1_mean": mean([row["top1"] for row in rows]),
        "top1_std": std([row["top1"] for row in rows]),
        "oracle_mean": mean([row["oracle"] for row in rows]),
        "oracle_std": std([row["oracle"] for row in rows]),
        "oracle_gap_mean": mean([row["gap"] for row in rows]),
        "oracle_gap_std": std([row["gap"] for row in rows]),
        "winner_not_top1_rate_mean": mean([row["winner_not_top1_rate"] for row in rows]),
        "winner_not_top1_rate_std": std([row["winner_not_top1_rate"] for row in rows]),
    }


def write_csv(path: Path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, rows):
    width, height = 760, 420
    margin_l, margin_r, margin_t, margin_b = 70, 30, 30, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    xs = [row["k"] for row in rows]
    ys = [row["top1_mean"] for row in rows] + [row["oracle_mean"] for row in rows]
    y_min = min(ys) - 0.01
    y_max = max(ys) + 0.01
    log_min = math.log2(min(xs))
    log_max = math.log2(max(xs))

    def sx(k):
        if log_max == log_min:
            return margin_l + plot_w / 2
        return margin_l + (math.log2(k) - log_min) / (log_max - log_min) * plot_w

    def sy(v):
        return margin_t + (y_max - v) / (y_max - y_min) * plot_h

    def poly(points):
        return " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in points)

    greedy = [(row["k"], row["top1_mean"]) for row in rows]
    oracle = [(row["k"], row["oracle_mean"]) for row in rows]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}" stroke="#222"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}" stroke="#222"/>',
        f'<polyline points="{poly(greedy)}" fill="none" stroke="#555" stroke-width="2.5"/>',
        f'<polyline points="{poly(oracle)}" fill="none" stroke="#0072B2" stroke-width="3"/>',
    ]
    for row in rows:
        k = row["k"]
        lines.append(f'<circle cx="{sx(k):.1f}" cy="{sy(row["top1_mean"]):.1f}" r="4" fill="#555"/>')
        lines.append(f'<circle cx="{sx(k):.1f}" cy="{sy(row["oracle_mean"]):.1f}" r="4" fill="#0072B2"/>')
        lines.append(f'<text x="{sx(k):.1f}" y="{height-28}" text-anchor="middle" font-size="12">{k}</text>')
    for i in range(5):
        v = y_min + i * (y_max - y_min) / 4
        y = sy(v)
        lines.append(f'<line x1="{margin_l-5}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}" stroke="#ddd"/>')
        lines.append(f'<text x="{margin_l-10}" y="{y+4:.1f}" text-anchor="end" font-size="11">{v:.3f}</text>')
    lines += [
        f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-size="13">Candidate bank size K</text>',
        f'<text x="16" y="{height/2}" transform="rotate(-90 16 {height/2})" text-anchor="middle" font-size="13">NAVSIM score</text>',
        '<rect x="540" y="34" width="12" height="12" fill="#0072B2"/>',
        '<text x="558" y="45" font-size="12">Oracle@K</text>',
        '<rect x="540" y="54" width="12" height="12" fill="#555"/>',
        '<text x="558" y="65" font-size="12">Greedy K=1</text>',
        "</svg>",
    ]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, default=Path("output/navsim/sweeps"))
    parser.add_argument("--ks", default="1,2,4,8,16,32")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-svg", type=Path)
    args = parser.parse_args()

    ks = [int(v) for v in args.ks.split(",") if v]
    seeds = [int(v) for v in args.seeds.split(",") if v]
    rows = [row for k in ks if (row := summarize_k(args.sweep_dir, k, seeds))]
    args.out_json.write_text(json.dumps({"rows": rows}, indent=2))
    if args.out_csv:
        write_csv(args.out_csv, rows)
    if args.out_svg:
        write_svg(args.out_svg, rows)
    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
