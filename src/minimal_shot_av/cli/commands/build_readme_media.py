#!/usr/bin/env python3
"""Build compact README media panels from existing repo assets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[4]
IMAGES = ROOT / "docs" / "images"

BG = "#0b1020"
PANEL = "#121932"
PANEL_2 = "#0f1730"
TEXT = "#f5f7fb"
MUTED = "#b7c0d8"
ACCENT = "#62d0ff"
GREEN = "#41d39d"
RED = "#ff6b6b"
ORANGE = "#ffb454"
BORDER = "#24304f"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


FONT_H1 = font(58, bold=True)
FONT_H2 = font(32, bold=True)
FONT_H3 = font(24, bold=True)
FONT_BODY = font(20)
FONT_SMALL = font(18)
FONT_TINY = font(16)


def rounded(draw: ImageDraw.ImageDraw, box, radius: int, fill, outline=None, width: int = 1):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def fit_crop(path: Path, size: tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    src_w, src_h = img.size
    dst_w, dst_h = size
    src_ratio = src_w / src_h
    dst_ratio = dst_w / dst_h
    if src_ratio > dst_ratio:
        new_h = dst_h
        new_w = int(dst_h * src_ratio)
    else:
        new_w = dst_w
        new_h = int(dst_w / src_ratio)
    img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - dst_w) // 2
    top = (new_h - dst_h) // 2
    return img.crop((left, top, left + dst_w, top + dst_h))


def add_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], title: str, subtitle: str, width: int):
    x, y = xy
    draw.text((x, y), title, font=FONT_H3, fill=TEXT)
    draw.text((x, y + 34), subtitle, font=FONT_TINY, fill=MUTED)
    draw.line((x, y + 66, x + width, y + 66), fill=BORDER, width=2)


def build_hero():
    w, h = 1600, 900
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle((0, 0, w, h), fill=BG)
    draw.ellipse((-200, -120, 520, 420), fill="#10204c")
    draw.ellipse((1080, 540, 1760, 1160), fill="#122449")

    draw.text((70, 58), "SPOTLIGHT REFLEX", font=FONT_H1, fill=TEXT)
    draw.text((72, 132), "Minimal-shot autonomy that shows decisions, transfer, and failure boundaries.",
              font=FONT_BODY, fill=MUTED)

    left_box = (70, 215, 780, 695)
    right_box = (820, 215, 1530, 695)
    rounded(draw, left_box, 24, PANEL, BORDER, 2)
    rounded(draw, right_box, 24, PANEL, BORDER, 2)

    left_img = fit_crop(IMAGES / "kf_spotlight_success.png", (670, 360))
    right_img = fit_crop(IMAGES / "baseline_spotlight.gif", (670, 360))
    img.paste(left_img, (90, 235))
    img.paste(right_img, (840, 235))

    add_label(draw, (92, 615), "Spotlight Reflex", "Wrong-way actor cleared in closed loop", 666)
    add_label(draw, (842, 615), "Baseline", "Same seed, same hazard, collision", 666)

    metric_y = 760
    metrics = [
        ("350", "closed-loop rollouts"),
        ("57.6%", "gauntlet pass"),
        ("7.848", "best tracked RFS"),
        ("0.600", "canonical AlpaSim collision"),
    ]
    box_w = 340
    gap = 20
    for i, (value, label) in enumerate(metrics):
        x0 = 70 + i * (box_w + gap)
        rounded(draw, (x0, metric_y, x0 + box_w, metric_y + 92), 18, PANEL_2, BORDER, 2)
        draw.text((x0 + 24, metric_y + 14), value, font=FONT_H2, fill=ACCENT)
        draw.text((x0 + 24, metric_y + 54), label, font=FONT_SMALL, fill=MUTED)

    img.save(IMAGES / "readme_hero_banner.png", quality=95)


def build_showcase():
    w, h = 1600, 1180
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    draw.text((70, 46), "Scenario + Transfer Showcase", font=FONT_H2, fill=TEXT)
    draw.text((70, 90), "The repo already has the right proof sequence: breadth, pressure, then external transfer.",
              font=FONT_SMALL, fill=MUTED)

    cards = [
        ("Construction corridor", "narrow lane closure, explicit maneuver change", IMAGES / "kf_construction.png"),
        ("Intersection stress", "crossing actors and timing conflict", IMAGES / "kf_intersection_stress.png"),
        ("Foreign object debris", "obstacle avoidance with lane recovery", IMAGES / "kf_fod.png"),
        ("AlpaSim reasoning", "sensor-realistic adapter and decision surface", IMAGES / "alpasim_reasoning_panel.png"),
    ]
    margin_x = 70
    margin_y = 150
    gap_x = 30
    gap_y = 34
    card_w = 715
    card_h = 440
    image_h = 300

    for idx, (title, subtitle, path) in enumerate(cards):
        row = idx // 2
        col = idx % 2
        x0 = margin_x + col * (card_w + gap_x)
        y0 = margin_y + row * (card_h + gap_y)
        rounded(draw, (x0, y0, x0 + card_w, y0 + card_h), 24, PANEL, BORDER, 2)
        panel = fit_crop(path, (card_w - 40, image_h))
        img.paste(panel, (x0 + 20, y0 + 20))
        draw.text((x0 + 22, y0 + 340), title, font=FONT_H3, fill=TEXT)
        draw.text((x0 + 22, y0 + 376), subtitle, font=FONT_SMALL, fill=MUTED)

    img.save(IMAGES / "readme_showcase_grid.png", quality=95)


def build_scoreboard():
    w, h = 1600, 620
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    draw.text((70, 52), "What matters", font=FONT_H2, fill=TEXT)
    draw.text((70, 96), "Compact scorecards for simulation, benchmark lift, and transfer honesty.",
              font=FONT_SMALL, fill=MUTED)

    cards = [
        ("Closed-loop simulator", GREEN, [("350", "primary rollouts"), ("0.0", "collision rate"), ("9.137 / 10", "COMPASS")]),
        ("Matched baseline gap", ORANGE, [("57.6%", "Spotlight pass"), ("2.1%", "baseline pass"), ("7.9%", "Spotlight collision")]),
        ("WOD-E2E selector", ACCENT, [("7.848", "best tracked RFS"), ("1.407", "oracle gap"), ("5", "CV folds")]),
        ("AlpaSim diagnostic", RED, [("0.600", "raw collision"), ("0.600", "actor-complete rerun"), ("per-axis", "failure accounting")]),
    ]

    card_w = 340
    card_h = 360
    gap = 20
    y0 = 170
    for i, (title, color, rows) in enumerate(cards):
        x0 = 70 + i * (card_w + gap)
        rounded(draw, (x0, y0, x0 + card_w, y0 + card_h), 22, PANEL, BORDER, 2)
        draw.rectangle((x0, y0, x0 + card_w, y0 + 10), fill=color)
        draw.text((x0 + 20, y0 + 26), title, font=FONT_H3, fill=TEXT)
        yy = y0 + 96
        for value, label in rows:
            draw.text((x0 + 20, yy), value, font=FONT_H2, fill=color)
            draw.text((x0 + 20, yy + 40), label, font=FONT_SMALL, fill=MUTED)
            yy += 88

    img.save(IMAGES / "readme_scoreboard.png", quality=95)


def main():
    build_hero()
    build_showcase()
    build_scoreboard()
    print("Generated README media assets in docs/images/")


if __name__ == "__main__":
    main()
