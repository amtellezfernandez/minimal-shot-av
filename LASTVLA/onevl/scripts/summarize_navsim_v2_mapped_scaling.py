#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def maybe_load_json(path: Path):
    if path.exists():
        return load_json(path)
    return None


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.6f}"


def collect_row(smoke_dir: Path, group: str, k: int, seed: int):
    oracle_path = smoke_dir / f"navhard_smoke_{group}_hybrid_k{k}_seed{seed}_oracle.json"
    oracle = maybe_load_json(oracle_path)
    if oracle is None:
        return None

    probe_path = smoke_dir / f"navhard_smoke_{group}_k{k}_seed{seed}_zero_perception_loso.json"
    probe = maybe_load_json(probe_path)

    row = {
        "group": group,
        "scene_count": int(oracle["scene_count"]),
        "k": k,
        "seed": seed,
        "top1_epdms": float(oracle["top1_epdms"]),
        "best_fixed_candidate": int(oracle["best_fixed_candidate"]),
        "best_fixed_epdms": float(oracle["best_fixed_epdms"]),
        "token_oracle_proxy_at_k_score": float(oracle["token_oracle_proxy_at_k_score"]),
    }
    if probe is not None:
        row["zero_perception_loso_proxy"] = float(probe["loso_proxy_ridge_score"])
        row["zero_perception_gap_closed"] = float(probe["gap_closed_vs_top1_proxy"])
    return row


def write_markdown(path: Path, rows, include_probe: bool):
    lines = [
        "# NAVSIM v2 mapped smoke scaling",
        "",
        "Seed-specific corrected hybrid decoding over matched NAVSIM v2 smoke groups. "
        "Official EPDMS uses fixed-candidate selection over each mapped subset. "
        "The token-oracle proxy is per-scene candidate-bank headroom and is not an official aggregate.",
        "",
    ]

    if include_probe:
        lines.append(
            "| Group | Scenes | K | Official top-1 | Best fixed candidate | Best fixed EPDMS | "
            "Token-oracle proxy | Zero-perception LOSO proxy |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in rows:
            lines.append(
                f"| {row['group']} | {row['scene_count']} | {row['k']} | "
                f"{fmt(row['top1_epdms'])} | {fmt(row['best_fixed_candidate'])} | "
                f"{fmt(row['best_fixed_epdms'])} | {fmt(row['token_oracle_proxy_at_k_score'])} | "
                f"{fmt(row.get('zero_perception_loso_proxy'))} |"
            )
    else:
        lines.append(
            "| Group | Scenes | K | Official top-1 | Best fixed candidate | Best fixed EPDMS | Token-oracle proxy |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for row in rows:
            lines.append(
                f"| {row['group']} | {row['scene_count']} | {row['k']} | "
                f"{fmt(row['top1_epdms'])} | {fmt(row['best_fixed_candidate'])} | "
                f"{fmt(row['best_fixed_epdms'])} | {fmt(row['token_oracle_proxy_at_k_score'])} |"
            )

    grouped = {}
    for row in rows:
        grouped.setdefault(row["group"], {})[row["k"]] = row

    deltas = []
    for group, items in grouped.items():
        ks = sorted(items)
        if len(ks) < 2:
            continue
        low = items[ks[0]]
        high = items[ks[-1]]
        delta = {
            "group": group,
            "scene_count": low["scene_count"],
            "best_fixed_delta": high["best_fixed_epdms"] - low["best_fixed_epdms"],
            "proxy_delta": high["token_oracle_proxy_at_k_score"] - low["token_oracle_proxy_at_k_score"],
        }
        if include_probe and low.get("zero_perception_loso_proxy") is not None and high.get("zero_perception_loso_proxy") is not None:
            delta["probe_delta"] = high["zero_perception_loso_proxy"] - low["zero_perception_loso_proxy"]
        deltas.append(delta)

    if deltas:
        lines.extend(["", "## Delta readout", ""])
        if include_probe and any("probe_delta" in delta for delta in deltas):
            lines.append("| Group | Scenes | Best fixed delta | Proxy delta | Zero-perception delta |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for delta in deltas:
                lines.append(
                    f"| {delta['group']} | {delta['scene_count']} | {fmt(delta['best_fixed_delta'])} | "
                    f"{fmt(delta['proxy_delta'])} | {fmt(delta.get('probe_delta'))} |"
                )
        else:
            lines.append("| Group | Scenes | Best fixed delta | Proxy delta |")
            lines.append("| --- | ---: | ---: | ---: |")
            for delta in deltas:
                lines.append(
                    f"| {delta['group']} | {delta['scene_count']} | {fmt(delta['best_fixed_delta'])} | "
                    f"{fmt(delta['proxy_delta'])} |"
                )

    path.write_text("\n".join(lines) + "\n")


def write_svg(path: Path, rows):
    groups = []
    seen = set()
    for row in rows:
        if row["group"] not in seen:
            groups.append(row["group"])
            seen.add(row["group"])

    ks = sorted({row["k"] for row in rows})
    if not groups or not ks:
        return

    width, height = 980, 360
    margin_l, margin_r, margin_t, margin_b = 60, 220, 30, 60
    plot_w = width - margin_l - margin_r
    plot_h = height - margin_t - margin_b
    y_values = []
    for row in rows:
        y_values.extend(
            [
                row["best_fixed_epdms"],
                row["token_oracle_proxy_at_k_score"],
                row["top1_epdms"],
            ]
        )
    y_min = min(0.0, min(y_values) - 0.02)
    y_max = max(y_values) + 0.03

    def sy(value):
        return margin_t + (y_max - value) / (y_max - y_min) * plot_h

    group_spacing = plot_w / max(1, len(groups))
    k_offsets = {k: (-18 if idx == 0 else 18) for idx, k in enumerate(ks[:2])}
    if len(ks) > 2:
        start = -18 * (len(ks) - 1) / 2
        k_offsets = {k: start + 18 * idx for idx, k in enumerate(ks)}

    colors = {
        ("best_fixed", ks[0]): "#2b6cb0",
        ("best_fixed", ks[-1]): "#c05621",
        ("proxy", ks[0]): "#2f855a",
        ("proxy", ks[-1]): "#805ad5",
    }

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{margin_l}" y1="{height-margin_b}" x2="{width-margin_r}" y2="{height-margin_b}" stroke="#222"/>',
        f'<line x1="{margin_l}" y1="{margin_t}" x2="{margin_l}" y2="{height-margin_b}" stroke="#222"/>',
    ]

    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5
        y = sy(value)
        lines.append(f'<line x1="{margin_l-4}" y1="{y:.1f}" x2="{width-margin_r}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        lines.append(f'<text x="{margin_l-8}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{value:.3f}</text>')

    for idx, group in enumerate(groups):
        cx = margin_l + group_spacing * idx + group_spacing / 2
        lines.append(f'<text x="{cx:.1f}" y="{height-28}" text-anchor="middle" font-family="Arial" font-size="12">{group}</text>')
        for key in ("best_fixed", "proxy"):
            for k in ks:
                row = next((item for item in rows if item["group"] == group and item["k"] == k), None)
                if row is None:
                    continue
                value = row["best_fixed_epdms"] if key == "best_fixed" else row["token_oracle_proxy_at_k_score"]
                x = cx + k_offsets[k]
                color = colors.get((key, k), "#4a5568")
                lines.append(f'<circle cx="{x:.1f}" cy="{sy(value):.1f}" r="4" fill="{color}"><title>{group} K{k} {key}: {value:.4f}</title></circle>')
        top1_row = next((item for item in rows if item["group"] == group), None)
        if top1_row is not None:
            y = sy(top1_row["top1_epdms"])
            lines.append(f'<line x1="{cx-28:.1f}" y1="{y:.1f}" x2="{cx+28:.1f}" y2="{y:.1f}" stroke="#4a5568" stroke-width="2"/>')
            lines.append(f'<title>{group} top1: {top1_row["top1_epdms"]:.4f}</title>')

    legend_x = width - margin_r + 20
    legend_y = 48
    legend = [
        ("#4a5568", "Top-1 official"),
        (colors.get(("best_fixed", ks[0]), "#2b6cb0"), f"K{ks[0]} best fixed"),
        (colors.get(("best_fixed", ks[-1]), "#c05621"), f"K{ks[-1]} best fixed"),
        (colors.get(("proxy", ks[0]), "#2f855a"), f"K{ks[0]} proxy oracle"),
        (colors.get(("proxy", ks[-1]), "#805ad5"), f"K{ks[-1]} proxy oracle"),
    ]
    for i, (color, label) in enumerate(legend):
        y = legend_y + i * 21
        if i == 0:
            lines.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+16}" y2="{y}" stroke="{color}" stroke-width="2"/>')
        else:
            lines.append(f'<circle cx="{legend_x+8}" cy="{y}" r="4" fill="{color}"/>')
        lines.append(f'<text x="{legend_x+24}" y="{y+4}" font-family="Arial" font-size="12" fill="#2d3748">{label}</text>')

    lines.extend(
        [
            f'<text x="{width/2}" y="{height-8}" text-anchor="middle" font-family="Arial" font-size="13">Mapped smoke group</text>',
            f'<text x="18" y="{height/2}" transform="rotate(-90 18 {height/2})" text-anchor="middle" font-family="Arial" font-size="13">EPDMS / proxy score</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-dir", type=Path, default=Path("LASTVLA/onevl/output/navsim_v2/smoke"))
    parser.add_argument("--groups", default="g1,g2,g3,g4")
    parser.add_argument("--ks", default="4,8")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path)
    parser.add_argument("--out-svg", type=Path)
    args = parser.parse_args()

    groups = [group.strip() for group in args.groups.split(",") if group.strip()]
    ks = [int(value.strip()) for value in args.ks.split(",") if value.strip()]

    rows = []
    for group in groups:
        for k in ks:
            row = collect_row(args.smoke_dir, group, k, args.seed)
            if row is not None:
                rows.append(row)

    payload = {
        "description": "Mapped NAVSIM v2 navhard smoke scaling across matched scene groups. "
        "Official values are fixed-candidate EPDMS; token_oracle_proxy_at_k_score is a per-scene proxy oracle over candidates and is not an official combined score.",
        "rows": rows,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n")

    include_probe = any("zero_perception_loso_proxy" in row for row in rows)
    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.out_md, rows, include_probe)
    if args.out_svg:
        args.out_svg.parent.mkdir(parents=True, exist_ok=True)
        write_svg(args.out_svg, rows)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
