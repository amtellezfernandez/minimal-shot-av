#!/usr/bin/env python3
"""Generate submission visuals: animated GIFs and architecture SVG diagrams.

Outputs to docs/images/:
  spotlight_success.gif        -- Spotlight Reflex on spotlight scenario
  construction_success.gif     -- Construction scenario
  intersection_stress.gif      -- Intersection stress case
  fod_success.gif              -- Foreign object debris scenario
  baseline_spotlight.gif       -- Baseline policy on spotlight (failure)
  grand-pipeline.svg           -- WOD-E2E model pipeline diagram
  simulator-architecture.svg   -- Spotlight Reflex simulator diagram
  alpasim-bridge.svg           -- AlpaSim integration diagram
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("PIL not available; install Pillow")
    sys.exit(1)

OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)

BUNDLES = ROOT / "artifacts" / "sota_submission_bundles"

# ── colour palette ────────────────────────────────────────────────────────────
BG           = (245, 241, 232)   # road background tan
LANE_FILL    = (200, 185, 154)   # lane surface
LANE_EDGE    = (62, 58, 54)      # lane edge dashes
EGO_COL      = (0, 95, 115)      # ego vehicle (teal)
EGO_TRAIL    = (0, 150, 180, 80) # ego trail (semi-transparent)
START_COL    = (10, 147, 150)    # start marker
GOAL_COL     = (238, 155, 0)     # goal marker
OBS_COL      = (163, 61, 43)     # obstacle (red-brown)
AMB_COL      = (180, 180, 170)   # ambient texture obstacle
HUD_BG       = (30, 30, 40)      # HUD background
HUD_TEXT     = (220, 220, 220)   # HUD text
MANEUVER_COL = {
    "stop":         (220, 53, 69),
    "crawl":        (255, 193, 7),
    "slow_yield":   (255, 193, 7),
    "maintain":     (40, 167, 69),
    "nudge_left":   (0, 123, 255),
    "nudge_right":  (0, 123, 255),
    "evasive_left": (255, 87, 34),
    "evasive_right":(255, 87, 34),
    "lane_recover": (111, 66, 193),
    "route_recenter":(150, 150, 150),
}
ACTOR_COL = {
    "vehicle":        (109, 89, 122),
    "pedestrian":     (157, 78, 221),
    "cyclist":        (42, 157, 143),
    "animal":         (127, 85, 57),
    "debris":         (92, 103, 125),
    "worker":         (247, 127, 0),
    "special_vehicle":(208, 0, 0),
}

SCALE     = 7        # px per meter
MARGIN    = 16       # px margin around scene
HUD_H     = 96       # px HUD strip height
EGO_R     = 5        # ego radius px
TRAIL_LEN = 30       # max trail steps to draw
GIF_FPS   = 12       # frames per second
FRAME_DUR = int(1000 / GIF_FPS)  # ms per frame
SIM_TICK_DT_S = 0.25


# ── geometry helpers ──────────────────────────────────────────────────────────

def _interp_lane(points: list) -> list[tuple[float, float]]:
    """Interpolate lane centre with a smooth spline at roughly 2 m spacing."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) < 3:
        return pts

    def catmull_rom(
        p0: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
        t: float,
    ) -> tuple[float, float]:
        t2 = t * t
        t3 = t2 * t
        x = 0.5 * (
            (2.0 * p1[0])
            + (-p0[0] + p2[0]) * t
            + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
            + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
        )
        y = 0.5 * (
            (2.0 * p1[1])
            + (-p0[1] + p2[1]) * t
            + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
            + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
        )
        return (x, y)

    out: list[tuple[float, float]] = [pts[0]]
    for index in range(len(pts) - 1):
        a = pts[index]
        b = pts[index + 1]
        p0 = pts[index - 1] if index > 0 else a
        p1 = a
        p2 = b
        p3 = pts[index + 2] if index + 2 < len(pts) else b
        dist = math.hypot(b[0] - a[0], b[1] - a[1])
        steps = max(1, int(dist / 2.0))
        for i in range(1, steps + 1):
            t = i / steps
            out.append(catmull_rom(p0, p1, p2, p3, t))
    return out


def _to_px(x: float, y: float, margin: int = MARGIN) -> tuple[int, int]:
    return (int(x * SCALE) + margin, int(y * SCALE) + margin)


def _actor_pose_at_tick(actor: dict[str, Any], tick: float) -> tuple[float, float] | None:
    active_from = float(actor.get("active_from", 0))
    active_until = float(actor.get("active_until", 10_000))
    if tick < active_from or tick > active_until:
        return None

    elapsed = max(0.0, tick - active_from) * SIM_TICK_DT_S
    x = float(actor["x"])
    y = float(actor["y"])
    speed = float(actor.get("speed", 0.0))
    vx = float(actor.get("vx", 0.0))
    vy = float(actor.get("vy", 0.0))
    heading = float(actor.get("heading", 0.0))
    behavior = str(actor.get("behavior", "linear"))

    if behavior in {"cut_in", "swerve"}:
        longitudinal = speed * elapsed
        lateral = min(4.5, 0.38 * elapsed * elapsed)
        lateral *= -1.0 if vy < 0.0 else 1.0
        return (x + math.cos(heading) * longitudinal, y + math.sin(heading) * longitudinal + lateral)
    if behavior in {"darting", "erratic_pedestrian"}:
        pause = 0.4 if int(elapsed * 2.0) % 3 == 0 else 1.0
        wobble = math.sin(elapsed * 3.7) * 0.55
        return (x + vx * elapsed * pause, y + vy * elapsed * pause + wobble)
    if behavior in {"sudden_brake", "hesitating"}:
        moving_time = min(elapsed, 1.2)
        creep_time = max(0.0, elapsed - 1.2)
        distance = speed * moving_time + speed * 0.15 * creep_time
        return (x + math.cos(heading) * distance, y + math.sin(heading) * distance)
    if behavior == "wrong_way":
        return (x - abs(vx) * elapsed, y + vy * elapsed)
    return (x + vx * elapsed, y + vy * elapsed)


def _lane_count(scenario: dict) -> int:
    for feature in scenario.get("map_features", []):
        if feature.get("kind") == "route_corridor":
            try:
                return max(1, int(feature.get("lane_count", 1)))
            except (TypeError, ValueError):
                return 1
    return 1


def _offset_ribbon(points: list[tuple[float, float]], offset_m: float) -> list[tuple[float, float]]:
    if not points:
        return []
    if len(points) == 1:
        return [points[0]]
    out: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        prev_point = points[index - 1] if index > 0 else points[index]
        next_point = points[index + 1] if index < len(points) - 1 else points[index]
        dx = next_point[0] - prev_point[0]
        dy = next_point[1] - prev_point[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-9:
            nx, ny = 0.0, 1.0
        else:
            nx, ny = -dy / norm, dx / norm
        out.append((point[0] + nx * offset_m, point[1] + ny * offset_m))
    return out


def _draw_road(draw: ImageDraw.ImageDraw, scenario: dict, lane: list[tuple[float, float]]) -> None:
    half_width = float(scenario["lane_half_width"])
    lane_count = _lane_count(scenario)
    road_half_width = half_width * max(1.0, lane_count / 2.0)
    shoulder_half_width = road_half_width + 1.25
    left_shoulder = _offset_ribbon(lane, shoulder_half_width)
    right_shoulder = _offset_ribbon(lane, -shoulder_half_width)
    left_edge = _offset_ribbon(lane, road_half_width)
    right_edge = _offset_ribbon(lane, -road_half_width)
    shoulder_polygon = [_to_px(x, y) for x, y in left_shoulder] + [_to_px(x, y) for x, y in reversed(right_shoulder)]
    road_polygon = [_to_px(x, y) for x, y in left_edge] + [_to_px(x, y) for x, y in reversed(right_edge)]
    draw.polygon(shoulder_polygon, fill=(173, 184, 165))
    draw.polygon(road_polygon, fill=(73, 81, 90))
    edge_left = [_to_px(x, y) for x, y in left_edge]
    edge_right = [_to_px(x, y) for x, y in right_edge]
    if len(edge_left) >= 2:
        draw.line(edge_left, fill=(186, 193, 201), width=1)
    if len(edge_right) >= 2:
        draw.line(edge_right, fill=(186, 193, 201), width=1)
    if lane_count > 1:
        lane_width = (road_half_width * 2.0) / lane_count
        for divider in range(1, lane_count):
            divider_points = _offset_ribbon(lane, -road_half_width + lane_width * divider)
            divider_px = [_to_px(x, y) for x, y in divider_points]
            divider_color = (248, 250, 252) if lane_count != 2 else (250, 204, 21)
            for i in range(0, len(divider_px) - 1, 5):
                a = divider_px[i]
                b = divider_px[min(i + 2, len(divider_px) - 1)]
                draw.line([a, b], fill=divider_color, width=2)
    else:
        center_px = [_to_px(x, y) for x, y in lane]
        for i in range(0, len(center_px) - 1, 5):
            a = center_px[i]
            b = center_px[min(i + 2, len(center_px) - 1)]
            draw.line([a, b], fill=(250, 204, 21), width=2)


def _draw_crosswalk(draw: ImageDraw.ImageDraw, px: int, py: int, width_px: int) -> None:
    stripe_count = 7
    stripe_h = max(4, int(width_px / (stripe_count * 2.3)))
    gap = max(3, int(width_px / stripe_count) - stripe_h)
    top = py - width_px // 2
    for index in range(stripe_count):
        y0 = top + index * (stripe_h + gap)
        draw.rectangle([px - 8, y0, px + 8, y0 + stripe_h], fill=(248, 250, 252, 210))


def _draw_cone(draw: ImageDraw.ImageDraw, px: int, py: int, r: int) -> None:
    points = [(px, py - r), (px - int(r * 0.8), py + r), (px + int(r * 0.8), py + r)]
    draw.polygon(points, fill=(249, 115, 22))
    draw.line([points[1], points[0], points[2], points[1]], fill=(255, 247, 237), width=1)


def _draw_vehicle_body(draw: ImageDraw.ImageDraw, px: int, py: int, length_px: int, width_px: int, fill: tuple[int, int, int]) -> None:
    draw.rounded_rectangle(
        [px - length_px // 2, py - width_px // 2, px + length_px // 2, py + width_px // 2],
        radius=max(2, width_px // 4),
        fill=fill,
    )
    window_w = max(3, int(length_px * 0.22))
    window_h = max(2, int(width_px * 0.46))
    draw.rounded_rectangle(
        [px - window_w // 2, py - window_h // 2, px + window_w // 2, py + window_h // 2],
        radius=2,
        fill=(224, 242, 254),
    )


# ── per-frame renderer ────────────────────────────────────────────────────────

def _draw_frame(
    scenario: dict,
    steps: list[dict],
    idx: int,
    img_w: int,
    img_h: int,
    draw_trail: bool = True,
) -> Image.Image:
    img = Image.new("RGB", (img_w, img_h + HUD_H), (223, 231, 212))
    draw = ImageDraw.Draw(img, "RGBA")

    lane = _interp_lane(scenario["lane_center"])
    _draw_road(draw, scenario, lane)

    # map features
    for feat in scenario.get("map_features", []):
        kind = feat.get("kind", "")
        fx, fy = float(feat.get("x", 0)), float(feat.get("y", 0))
        px, py = _to_px(fx, fy)
        if kind == "crosswalk":
            w = int(float(feat.get("width", 12)) * SCALE)
            _draw_crosswalk(draw, px, py, w)
        elif kind == "conflict_zone":
            r = int(float(feat.get("radius", 8)) * SCALE)
            draw.ellipse([px-r, py-r, px+r, py+r], outline=(245, 158, 11, 120), width=2, fill=(245, 158, 11, 30))
        elif kind == "lane_closure":
            l = int(float(feat.get("length", 20)) * SCALE)
            draw.rectangle([px, py-8, px+l, py+8], fill=(249, 115, 22, 42), outline=(251, 146, 60, 120), width=1)
        elif kind in {"temporary_taper", "merge_taper"}:
            l = int(float(feat.get("length", 18)) * SCALE)
            draw.polygon([(px, py - 10), (px + l, py - 3), (px + l, py + 3), (px, py + 10)], fill=(249, 115, 22, 28), outline=(251, 146, 60, 110))
        elif kind == "merge_zone":
            l = int(float(feat.get("length", 20)) * SCALE)
            draw.rectangle([px, py-12, px+l, py+12], fill=(56, 189, 248, 24), outline=(2, 132, 199, 90), width=1)
        elif kind == "avoidance_corridor":
            w = int(float(feat.get("width", 10)) * SCALE)
            draw.rectangle([px - 18, py - w // 2, px + 58, py + w // 2], fill=(34, 197, 94, 18), outline=(22, 163, 74, 80), width=1)

    # obstacles
    for obs in scenario.get("obstacles", []):
        ox, oy = float(obs["x"]), float(obs["y"])
        r  = max(2, int(float(obs.get("radius", 1.0)) * SCALE))
        px, py = _to_px(ox, oy)
        kind = str(obs.get("kind", "obstacle"))
        if kind == "ambient":
            draw.ellipse([px-r, py-r, px+r, py+r], fill=(148, 163, 184, 72))
        elif kind == "cone":
            _draw_cone(draw, px, py, max(3, int(r * 1.6)))
        elif kind in {"vehicle", "special_vehicle"}:
            _draw_vehicle_body(draw, px, py, max(12, int(r * 3.6)), max(7, int(r * 1.7)), (124, 45, 18))
        else:
            draw.ellipse([px-r, py-r, px+r, py+r], fill=OBS_COL)

    # actors at current time
    tick = float(steps[idx]["t"]) if idx < len(steps) else 0.0
    for actor in scenario.get("actors", []):
        pose = _actor_pose_at_tick(actor, tick)
        if pose is None:
            continue
        ax, ay = pose
        akind = actor.get("kind", "vehicle")
        acol  = ACTOR_COL.get(akind, (108, 117, 125))
        if akind in {"pedestrian", "animal", "debris", "worker"}:
            r = max(3, int(float(actor.get("width", 1.0)) * SCALE // 2))
            px, py = _to_px(ax, ay)
            draw.ellipse([px-r, py-r, px+r, py+r], fill=acol)
            draw.ellipse([px-r-1, py-r-1, px+r+1, py+r+1], outline=(255, 247, 237), width=1)
        else:
            w = max(1, int(float(actor.get("width",  2.0)) * SCALE))
            l = max(1, int(float(actor.get("length", 4.0)) * SCALE))
            px, py = _to_px(ax, ay)
            _draw_vehicle_body(draw, px, py, l, w, acol)

    # ego trail
    trail_start = max(0, idx - TRAIL_LEN)
    trail_pts = [_to_px(steps[i]["x"], steps[i]["y"]) for i in range(trail_start, idx+1)]
    if len(trail_pts) >= 2:
        for i in range(len(trail_pts)-1):
            alpha = int(60 + 160 * i / max(1, len(trail_pts)-1))
            draw.line([trail_pts[i], trail_pts[i+1]], fill=(0,95,115,alpha), width=3)

    # start / goal
    sx, sy = _to_px(*scenario["start"])
    gx, gy = _to_px(*scenario["goal"])
    draw.ellipse([sx-6, sy-6, sx+6, sy+6], fill=START_COL)
    draw.ellipse([gx-6, gy-6, gx+6, gy+6], fill=GOAL_COL)

    # ego vehicle
    if idx < len(steps):
        step = steps[idx]
        ex, ey = _to_px(step["x"], step["y"])
        _draw_vehicle_body(draw, ex, ey, 24, 12, EGO_COL)
        draw.rounded_rectangle([ex - 14, ey - 8, ex + 14, ey + 8], radius=4, outline=(255, 255, 255), width=2)

    # ── HUD strip ─────────────────────────────────────────────────────────────
    hud_y = img_h
    draw.rectangle([0, hud_y, img_w, img_h + HUD_H], fill=HUD_BG)

    step = steps[idx] if idx < len(steps) else {}
    maneuver = step.get("selected_maneuver") or "—"
    score    = float(step.get("selector_score") or 0.0)
    pressure = float(step.get("obstacle_pressure") or 0.0)
    blockage = float(step.get("route_blockage") or 0.0)
    t_str    = f"t={step.get('t', 0):05.1f}s"
    cluster  = scenario.get("cluster", "")
    seed     = scenario.get("seed", 0)

    mcol = MANEUVER_COL.get(maneuver, (150, 150, 150))

    try:
        font_sm = ImageFont.load_default(size=13)
        font_lg = ImageFont.load_default(size=15)
    except Exception:
        font_sm = font_lg = ImageFont.load_default()

    def _tw(txt: str, font: Any) -> int:
        try:
            return int(font.getlength(txt))
        except Exception:
            return len(txt) * 7

    # ── Adaptive three-column layout ─────────────────────────────────
    # right column: fixed content "t=012.3s", "clust #s", "N/M"
    right_w = max(90, _tw("t=999.9s", font_sm) + 10)
    # left badge: proportional, max 170
    badge_w = min(170, max(110, int(img_w * 0.37)))
    # middle: whatever remains (at least 60px for bars)
    mid_start = badge_w + 14
    mid_avail = img_w - right_w - 8 - mid_start  # px for label+bar+value
    lbl_w_m   = min(46, int(mid_avail * 0.25))
    val_w     = _tw("0.00", font_sm) + 6
    bar_w     = max(24, mid_avail - lbl_w_m - val_w - 4)
    bx        = mid_start + lbl_w_m  # bar start x

    # ── Left badge ───────────────────────────────────────────────────
    draw.rectangle([6, hud_y+4, 6+badge_w, hud_y+HUD_H-4], fill=mcol)
    label_text = maneuver.upper().replace("_", " ")
    draw.text((12, hud_y+10), label_text, fill=(255, 255, 255), font=font_lg)
    draw.text((12, hud_y+34), f"score {score:.1f}", fill=(230, 230, 230), font=font_sm)
    draw.text((12, hud_y+54), f"{idx+1}/{len(steps)}", fill=(190, 190, 190), font=font_sm)

    # ── Middle bars ───────────────────────────────────────────────────
    # PRES row
    draw.text((mid_start, hud_y+13), "PRES:", fill=(160, 160, 160), font=font_sm)
    draw.rectangle([bx, hud_y+10, bx+bar_w, hud_y+24],
                   outline=(80, 80, 90), fill=(50, 50, 60))
    if pressure > 0:
        draw.rectangle([bx, hud_y+10, bx+int(pressure*bar_w), hud_y+24],
                       fill=(220, 53, 69))
    draw.text((bx+bar_w+4, hud_y+13), f"{pressure:.2f}",
              fill=(200, 200, 200), font=font_sm)

    # BLOC row
    draw.text((mid_start, hud_y+43), "BLOC:", fill=(160, 160, 160), font=font_sm)
    draw.rectangle([bx, hud_y+40, bx+bar_w, hud_y+54],
                   outline=(80, 80, 90), fill=(50, 50, 60))
    if blockage > 0:
        draw.rectangle([bx, hud_y+40, bx+int(blockage*bar_w), hud_y+54],
                       fill=(255, 193, 7))
    draw.text((bx+bar_w+4, hud_y+43), f"{blockage:.2f}",
              fill=(200, 200, 200), font=font_sm)

    # ── Right column: anchored to right edge ─────────────────────────
    rx = img_w - 6
    def rtext(txt: str, y: int, col: tuple) -> None:
        draw.text((int(rx - _tw(txt, font_sm)), y), txt, fill=col, font=font_sm)

    rtext(t_str,                  hud_y+13, HUD_TEXT)
    rtext(f"{cluster}#{seed}",    hud_y+43, (140, 170, 195))

    return img


def _bar(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int,
         val: float, col: tuple, label: str, font: Any) -> None:
    draw.rectangle([x, y, x+w, y+h], outline=(80, 80, 90), fill=(50, 50, 60))
    filled = int(val * w)
    if filled > 0:
        draw.rectangle([x, y, x+filled, y+h], fill=col)
    draw.text((x+w+6, y), f"{val:.2f}", fill=(180, 180, 180), font=font)


# ── GIF builder ───────────────────────────────────────────────────────────────

def _build_gif(rollout_json: Path, out_path: Path, step_skip: int = 2) -> None:
    data     = json.loads(rollout_json.read_text())
    scenario = data.get("scenario", data)
    steps    = data.get("rollout", {}).get("steps", data.get("steps", []))
    if not steps:
        print(f"  no steps in {rollout_json.name}")
        return

    W = int(float(scenario["width"])  * SCALE) + 2 * MARGIN
    H = int(float(scenario["height"]) * SCALE) + 2 * MARGIN

    frames: list[Image.Image] = []
    indices = list(range(0, len(steps), step_skip))
    if indices[-1] != len(steps) - 1:
        indices.append(len(steps) - 1)

    for i in indices:
        f = _draw_frame(scenario, steps, i, W, H)
        frames.append(f.convert("P", palette=Image.ADAPTIVE, colors=128))

    # hold last frame longer
    durations = [FRAME_DUR] * len(frames)
    durations[-1] = 1500

    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"  wrote {out_path.name}  ({len(frames)} frames, {len(steps)} steps)")


def build_all_gifs() -> None:
    specs = [
        (BUNDLES / "grand_spotlight_demo" / "latest_rollout.json", "spotlight_success.gif", 2),
        (BUNDLES / "grand_intersection_stress_seed3" / "latest_rollout.json", "intersection_stress.gif", 2),
        (BUNDLES / "grand_baseline_spotlight_demo" / "latest_rollout.json", "baseline_spotlight.gif", 2),
        (BUNDLES / "minor_construction_seed1" / "latest_rollout.json", "construction_success.gif", 2),
        (BUNDLES / "minor_fod_seed2" / "latest_rollout.json", "fod_success.gif", 2),
        (BUNDLES / "minor_spotlight_seed3" / "latest_rollout.json", "spotlight_minor.gif", 2),
    ]
    print("Generating GIFs...")
    for src, gif_name, skip in specs:
        if not src.exists():
            print(f"  skip {src} (no rollout JSON)")
            continue
        _build_gif(src, OUT / gif_name, skip)


# ── architecture SVGs ─────────────────────────────────────────────────────────

def _svg_header(w: int, h: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" style="font-family:\'Helvetica Neue\',Arial,sans-serif">\n'
        f'<rect width="{w}" height="{h}" fill="#1a1a2e"/>\n'
    )

def _svg_footer() -> str:
    return "</svg>\n"

def _box(x: int, y: int, w: int, h: int, fill: str, stroke: str,
         label: str, sublabel: str = "", r: int = 6) -> str:
    out = (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>\n'
    )
    cy = y + h // 2 - (6 if sublabel else 0)
    out += f'<text x="{x+w//2}" y="{cy}" text-anchor="middle" dominant-baseline="middle" fill="#f0f0f0" font-size="13" font-weight="600">{label}</text>\n'
    if sublabel:
        out += f'<text x="{x+w//2}" y="{cy+16}" text-anchor="middle" dominant-baseline="middle" fill="#a0a0c0" font-size="10">{sublabel}</text>\n'
    return out

def _arrow(x1: int, y1: int, x2: int, y2: int, col: str = "#4fc3f7") -> str:
    dx, dy = x2-x1, y2-y1
    length = math.hypot(dx, dy)
    if length < 1:
        return ""
    ux, uy = dx/length, dy/length
    # arrowhead
    ax1 = x2 - 10*ux + 5*uy
    ay1 = y2 - 10*uy - 5*ux  # actually needs fixing but good enough
    ax2 = x2 - 10*ux - 5*uy
    ay2 = y2 - 10*uy + 5*ux
    # use marker-end approach via a simple polygon
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{col}" stroke-width="2" marker-end="url(#arr)"/>\n'
    )

def _label(x: int, y: int, text: str, col: str = "#90caf9", size: int = 11) -> str:
    return f'<text x="{x}" y="{y}" text-anchor="middle" fill="{col}" font-size="{size}">{text}</text>\n'

_DEFS = '''<defs>
  <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L0,6 L8,3 z" fill="#4fc3f7"/>
  </marker>
  <marker id="arr_gold" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L0,6 L8,3 z" fill="#ffd54f"/>
  </marker>
  <marker id="arr_grn" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
    <path d="M0,0 L0,6 L8,3 z" fill="#81c784"/>
  </marker>
</defs>\n'''

def _arrow_gold(x1,y1,x2,y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#ffd54f" stroke-width="2" marker-end="url(#arr_gold)"/>\n')
def _arrow_grn(x1,y1,x2,y2):
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#81c784" stroke-width="2" marker-end="url(#arr_grn)"/>\n')


def build_grand_pipeline_svg() -> None:
    W, H = 820, 540
    svg = [_svg_header(W, H), _DEFS]

    # title
    svg.append(f'<text x="{W//2}" y="30" text-anchor="middle" fill="#e0e0e0" font-size="16" font-weight="700">WOD-E2E Model Pipeline — Grand Commission</text>\n')

    # row 1: input
    svg.append(_box(300, 55, 220, 48, "#0d3b66", "#4fc3f7", "WOD-E2E TFRecord", "E2EDFrame proto · 93 val shards"))
    svg.append(_arrow(410, 103, 410, 130))

    # row 2: parser
    svg.append(_box(290, 130, 240, 44, "#1a3a5c", "#4fc3f7", "E2EDFrame Parser", "frame name · past states · intent · init_speed"))
    svg.append(_arrow(410, 174, 410, 200))

    # row 3: candidate generation (3 columns)
    svg.append(f'<text x="410" y="196" text-anchor="middle" fill="#90caf9" font-size="11">Candidate Generation</text>\n')
    svg.append(f'<rect x="55" y="200" width="710" height="68" rx="8" fill="#0f2942" stroke="#2a5a8a" stroke-width="1" stroke-dasharray="4 3"/>\n')
    svg.append(_box( 65, 210, 190, 48, "#1b4f72", "#4fc3f7", "Kinematic",  "const-vel · accel · heading · stop"))
    svg.append(_box(285, 210, 190, 48, "#1b4f72", "#4fc3f7", "Ridge Learned", "31 features · segment-CV · r=175"))
    svg.append(_box(505, 210, 190, 48, "#1b4f72", "#4fc3f7", "Temporal", "ego-history trends · CV-trained"))
    svg.append(_box(710, 210, 100, 48, "#2a2a3a", "#555577", "Anchor†", "disabled"))

    # arrows from candidates to ranker
    for cx in [160, 380, 600]:
        svg.append(_arrow(cx, 268, 410, 300))

    # row 4: ranker
    svg.append(_box(230, 300, 360, 56, "#1a3a1a", "#81c784",
                    "Contextual Ranker", "WodPreferenceRanker · HGB / ridge · ~137 features"))
    svg.append(_arrow(410, 356, 410, 385))

    # row 5: selection + submission
    svg.append(_box(270, 385, 280, 44, "#2a1a0a", "#ffd54f", "Selected Trajectory", "top ranker_score per frame"))
    svg.append(_arrow(410, 429, 410, 458))
    svg.append(_box(260, 458, 300, 48, "#1a2a0a", "#81c784",
                    "E2EDChallengeSubmission", "20×2 waypoints · .tar.gz · validated"))

    # side annotation: oracle gap
    svg.append(f'<rect x="660" y="300" width="145" height="76" rx="6" fill="#1a1020" stroke="#7b1fa2" stroke-width="1"/>\n')
    svg.append(f'<text x="732" y="320" text-anchor="middle" fill="#ce93d8" font-size="11" font-weight="600">Val-CV Evidence</text>\n')
    svg.append(f'<text x="668" y="336" fill="#b39ddb" font-size="10">Constant-vel baseline: 7.022</text>\n')
    svg.append(f'<text x="668" y="350" fill="#b39ddb" font-size="10">Champion RFS:  7.880</text>\n')
    svg.append(f'<text x="668" y="364" fill="#b39ddb" font-size="10">Oracle RFS:    9.068</text>\n')
    svg.append(f'<text x="668" y="378" fill="#7c7c9c" font-size="9">479 val frames · seg-grouped CV</text>\n')

    # note: anchor disabled
    svg.append(f'<text x="760" y="224" fill="#555577" font-size="9">†not selected</text>\n')

    svg.append(_svg_footer())
    path = OUT / "grand-pipeline.svg"
    path.write_text("".join(svg))
    print(f"  wrote {path.name}")


def build_simulator_architecture_svg() -> None:
    W, H = 820, 580
    svg = [_svg_header(W, H), _DEFS]

    svg.append(f'<text x="{W//2}" y="30" text-anchor="middle" fill="#e0e0e0" font-size="16" font-weight="700">Spotlight Reflex Simulator Architecture</text>\n')

    # input
    svg.append(_box(280, 52, 260, 44, "#0d3b66", "#4fc3f7", "Scenario (cluster + seed)", "deterministic · reproducible · 11 WOD clusters"))
    svg.append(_arrow(410, 96, 410, 124))

    # perception
    svg.append(_box(290, 124, 240, 44, "#1a3a5c", "#4fc3f7", "ScenePerception", "obstacle field · actor set · visibility"))
    svg.append(_arrow(410, 168, 410, 196))

    # world state
    svg.append(_box(270, 196, 280, 52, "#1a2a4a", "#4fc3f7",
                    "WorldState", "pressure · route_blockage · corridor · clearances · escape_side"))
    svg.append(_arrow(410, 248, 410, 275))

    # maneuver candidates group
    svg.append(f'<text x="410" y="272" text-anchor="middle" fill="#90caf9" font-size="11">Maneuver Candidate Library</text>\n')
    svg.append(f'<rect x="40" y="278" width="740" height="56" rx="8" fill="#0f2942" stroke="#2a5a8a" stroke-width="1" stroke-dasharray="4 3"/>\n')

    maneuvers = ["stop", "crawl", "maintain", "slow_yield", "nudge_L", "nudge_R", "evasive_L", "evasive_R", "recover"]
    for i, m in enumerate(maneuvers):
        bx = 46 + i * 80
        svg.append(_box(bx, 284, 72, 40, "#163551", "#4fc3f7", m, ""))
    svg.append(_arrow(410, 334, 410, 362))

    # reference rules
    svg.append(_box(220, 362, 380, 56, "#1a2a1a", "#81c784",
                    "Reference Rule Engine", "clear → maintain/center  ·  obstacle → slow/nudge/evasive  ·  uncertain → crawl/stop"))
    svg.append(_arrow(410, 418, 410, 446))

    # trajectory selector
    svg.append(_box(240, 446, 340, 52, "#2a1a0a", "#ffd54f",
                    "Trajectory Selector", "3s region (1.0m lat · 4.0m lng)  +  5s region (1.8m · 7.2m)"))
    svg.append(_arrow(410, 498, 410, 526))

    # output
    svg.append(_box(250, 526, 320, 44, "#1a3a1a", "#81c784",
                    "Selected Maneuver + reasoning_text", "20-pt trajectory · full decision JSON"))

    # side panel: world state indicators
    svg.append(f'<rect x="640" y="196" width="165" height="120" rx="6" fill="#1a1020" stroke="#7b1fa2" stroke-width="1"/>\n')
    svg.append(f'<text x="722" y="214" text-anchor="middle" fill="#ce93d8" font-size="11" font-weight="600">WorldState fields</text>\n')
    for i, line in enumerate([
        "obstacle_pressure  [0,1]",
        "route_blockage     [0,1]",
        "corridor_blocked   bool",
        "left_clearance     m",
        "right_clearance    m",
        "preferred_escape_side",
    ]):
        svg.append(f'<text x="648" y="{230+i*15}" fill="#b39ddb" font-size="9">{line}</text>\n')

    svg.append(_svg_footer())
    path = OUT / "simulator-architecture.svg"
    path.write_text("".join(svg))
    print(f"  wrote {path.name}")


def build_alpasim_bridge_svg() -> None:
    W, H = 820, 540
    svg = [_svg_header(W, H), _DEFS]

    svg.append(f'<text x="{W//2}" y="30" text-anchor="middle" fill="#e0e0e0" font-size="16" font-weight="700">AlpaSim Integration — Spotlight Reflex Adapter</text>\n')

    # AlpaSim box (left)
    svg.append(f'<rect x="20" y="55" width="175" height="420" rx="8" fill="#0a1628" stroke="#1565c0" stroke-width="1.5"/>\n')
    svg.append(f'<text x="107" y="78" text-anchor="middle" fill="#64b5f6" font-size="13" font-weight="700">AlpaSim</text>\n')
    for i, line in enumerate([
        "8-camera frames",
        "route command",
        "speed + accel",
        "ego pose history",
        "structured hazards",
        "(optional AlpaSignal)",
    ]):
        svg.append(f'<text x="107" y="{100+i*22}" text-anchor="middle" fill="#90caf9" font-size="10">{line}</text>\n')

    # PredictionInput arrow
    svg.append(_arrow(195, 265, 240, 265))
    svg.append(f'<text x="217" y="258" text-anchor="middle" fill="#90caf9" font-size="9">PredictionInput</text>\n')

    # Adapter box (centre)
    svg.append(f'<rect x="240" y="55" width="335" height="420" rx="8" fill="#0d2137" stroke="#4fc3f7" stroke-width="2"/>\n')
    svg.append(f'<text x="407" y="78" text-anchor="middle" fill="#4fc3f7" font-size="13" font-weight="700">SpotlightReflexAlpaSimModel</text>\n')
    svg.append(f'<text x="407" y="93" text-anchor="middle" fill="#4fc3f7" font-size="9">alpasim_spotlight.py  ·  BaseTrajectoryModel</text>\n')

    # 4 signal extractors
    extractors = [
        (255, 115, "#e91e63", "Route → Lane Geometry",        "LEFT/STRAIGHT/RIGHT → sigmoid spline"),
        (175, 130, "#ff9800", "Camera → Visibility Risk",     "mean brightness < 0.35 → risk signal"),
        (175, 162, "#ff9800", "Dynamics → Braking Risk",      "accel < −4 m/s²  or  speed < 1.5 m/s"),
        (255, 162, "#9c27b0", "AlpaSignal → Obstacles/Actors","static: Obstacle · moving: Actor"),
    ]
    svg.append(f'<text x="407" y="108" text-anchor="middle" fill="#90caf9" font-size="10" font-style="italic">extract_alpasim_signal()</text>\n')
    svg.append(f'<rect x="248" y="112" width="319" height="98" rx="5" fill="#091824" stroke="#2a5a8a" stroke-width="1" stroke-dasharray="3 2"/>\n')
    for bx, by, col, label, sub in extractors:
        svg.append(_box(bx, by, 148, 38, "#101e2e", col, label, sub, r=4))

    # Scenario construction
    svg.append(_arrow(407, 210, 407, 232))
    svg.append(_box(290, 232, 234, 40, "#0f2233", "#4fc3f7", "scenario_from_command()", "Scenario with obstacles + actors"))
    svg.append(_arrow(407, 272, 407, 296))

    # Spotlight Reflex policy
    svg.append(_box(268, 296, 278, 44, "#163820", "#81c784", "Spotlight Reflex Policy", "perceive → world_state → select_maneuver"))
    svg.append(_arrow(407, 340, 407, 364))

    # resample
    svg.append(_box(290, 364, 234, 40, "#1a1a10", "#ffd54f", "_resample_to_frequency()", "20 pts @ 4 Hz → N pts @ output Hz"))
    svg.append(_arrow(407, 404, 407, 428))

    # output
    svg.append(_box(266, 428, 280, 40, "#163820", "#81c784", "ModelPrediction", "trajectory_xy · headings · reasoning_text"))

    # ModelPrediction arrow to AlpaSim controller
    svg.append(_arrow(575, 448, 620, 448))
    svg.append(f'<text x="597" y="442" text-anchor="middle" fill="#90caf9" font-size="9">ModelPrediction</text>\n')

    # AlpaSim controller (right)
    svg.append(f'<rect x="620" y="55" width="175" height="420" rx="8" fill="#0a1628" stroke="#1565c0" stroke-width="1.5"/>\n')
    svg.append(f'<text x="707" y="78" text-anchor="middle" fill="#64b5f6" font-size="13" font-weight="700">AlpaSim</text>\n')
    for i, line in enumerate([
        "trajectory controller",
        "closed-loop physics",
        "collision detection",
        "metrics collection:",
        "collision_at_fault",
        "offroad",
        "plan_deviation",
        "safety_monitor",
    ]):
        svg.append(f'<text x="707" y="{100+i*24}" text-anchor="middle" fill="#90caf9" font-size="10">{line}</text>\n')

    svg.append(_svg_footer())
    path = OUT / "alpasim-bridge.svg"
    path.write_text("".join(svg))
    print(f"  wrote {path.name}")


def build_minor_pipeline_svg() -> None:
    W, H = 820, 500
    svg = [_svg_header(W, H), _DEFS]

    svg.append(f'<text x="{W//2}" y="30" text-anchor="middle" fill="#e0e0e0" font-size="16" font-weight="700">Minor Commission — Simulation Environment Pipeline</text>\n')

    # generators side by side
    svg.append(f'<rect x="30" y="52" width="360" height="78" rx="8" fill="#0f2942" stroke="#2a5a8a" stroke-width="1" stroke-dasharray="4 3"/>\n')
    svg.append(f'<text x="210" y="70" text-anchor="middle" fill="#90caf9" font-size="11">WOD-E2E Procedural Generator</text>\n')
    svg.append(_box( 38, 78, 160, 44, "#163551", "#4fc3f7", "11 Named Clusters", "construction · FOD · spotlight · …"))
    svg.append(_box(216, 78, 160, 44, "#163551", "#4fc3f7", "Seeded RNG", "cluster + seed → full scenario"))

    svg.append(f'<rect x="430" y="52" width="360" height="78" rx="8" fill="#0f2233" stroke="#7b1fa2" stroke-width="1" stroke-dasharray="4 3"/>\n')
    svg.append(f'<text x="610" y="70" text-anchor="middle" fill="#ce93d8" font-size="11">Compositional OOD Generator</text>\n')
    svg.append(_box(438, 78, 160, 44, "#2a1540", "#9c27b0", "Independent Axes", "topology · hazards · conditions · objects"))
    svg.append(_box(616, 78, 160, 44, "#2a1540", "#9c27b0", "5 OOD Suites", "compositional · adversarial · gauntlet · hidden"))

    # both → scenario
    svg.append(_arrow(210, 130, 410, 160))
    svg.append(_arrow(610, 130, 410, 160))
    svg.append(_box(270, 160, 280, 44, "#1a3a1a", "#81c784", "Scenario Object", "lane · obstacles · actors · map_features · env"))
    svg.append(_arrow(410, 204, 410, 232))

    # policy
    svg.append(_box(270, 232, 280, 52, "#1b4f72", "#4fc3f7",
                    "Spotlight Reflex Policy", "closed-loop · step-by-step reaction"))
    svg.append(_arrow(410, 284, 410, 312))

    # rollout
    svg.append(_box(260, 312, 300, 52, "#2a1a0a", "#ffd54f",
                    "Closed-Loop Rollout", "108+ steps · x,y · maneuver · scores · world state"))
    svg.append(_arrow(410, 364, 410, 392))

    # evidence
    svg.append(f'<rect x="30" y="392" width="760" height="82" rx="8" fill="#0a1a0a" stroke="#388e3c" stroke-width="1.5"/>\n')
    svg.append(f'<text x="410" y="410" text-anchor="middle" fill="#81c784" font-size="12" font-weight="600">Evidence Package</text>\n')
    for i, (col, txt) in enumerate([
        ("#4caf50", "550 rollouts · 0 collisions · 326/350 benchmark passes"),
        ("#81c784", "Gauntlet 36/60 · AlpaSignal bridge audit · Runtime ≤50 ms p95"),
        ("#a5d6a7", "SVG/JSON per scenario · COMPASS report · Visual gallery"),
    ]):
        svg.append(f'<text x="410" y="{426+i*16}" text-anchor="middle" fill="{col}" font-size="10">{txt}</text>\n')

    svg.append(_svg_footer())
    path = OUT / "minor-pipeline.svg"
    path.write_text("".join(svg))
    print(f"  wrote {path.name}")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Building architecture SVGs...")
    build_grand_pipeline_svg()
    build_simulator_architecture_svg()
    build_alpasim_bridge_svg()
    build_minor_pipeline_svg()
    print("\nBuilding animated GIFs...")
    build_all_gifs()
    print(f"\nAll visuals written to {OUT}")
