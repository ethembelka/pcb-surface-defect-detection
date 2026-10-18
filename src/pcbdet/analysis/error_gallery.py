"""Visual galleries of errors: crops around missed boxes (FN), false alarms (FP) and confusions.

Boxes on 226 px PCB images are tiny, so each example is cropped around the box with context,
enlarged to ``crop_size`` px and tiled into a grid. Colour code (same everywhere):
green = ground truth, red = prediction (with class and score), title = image stem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont

from pcbdet.data.classes import CLASS_NAMES

GREEN, RED = (0, 220, 0), (255, 40, 40)


def _font(size: int = 12):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def crop_window(bbox: Sequence[float], img_w: int, img_h: int, margin: float, min_px: int) -> tuple[int, int, int, int]:
    """Square window around ``bbox`` (x, y, w, h) with ``margin`` x box size of context on each side,
    at least ``min_px`` wide, clamped to the image. Returns (x0, y0, x1, y1)."""
    x, y, w, h = bbox
    side = max(w, h) * (1 + 2 * margin)
    side = max(side, min_px)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    # shift inside the image instead of shrinking, so the box stays centred when possible
    if x0 < 0:
        x1, x0 = x1 - x0, 0
    if y0 < 0:
        y1, y0 = y1 - y0, 0
    if x1 > img_w:
        x0, x1 = max(0, x0 - (x1 - img_w)), img_w
    if y1 > img_h:
        y0, y1 = max(0, y0 - (y1 - img_h)), img_h
    return x0, y0, x1, y1


def render_crop(image_path: Path, gt_boxes: list[tuple], pred_boxes: list[tuple], focus_bbox: Sequence[float],
                margin: float, min_px: int, size: int, caption: str) -> Image.Image:
    """Crop around ``focus_bbox`` and draw GT (green) and predictions (red, ``(bbox, label)``)."""
    im = Image.open(image_path).convert("RGB")
    x0, y0, x1, y1 = crop_window(focus_bbox, im.width, im.height, margin, min_px)
    crop = im.crop((x0, y0, x1, y1))
    scale = size / max(crop.width, crop.height)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.BICUBIC)
    draw = ImageDraw.Draw(crop)

    def rect(b, colour, width=2):
        bx, by, bw, bh = b
        draw.rectangle([(bx - x0) * scale, (by - y0) * scale, (bx + bw - x0) * scale, (by + bh - y0) * scale],
                       outline=colour, width=width)

    for b in gt_boxes:
        rect(b, GREEN)
    for b, label in pred_boxes:
        rect(b, RED)
        draw.text(((b[0] - x0) * scale + 2, (b[1] - y0) * scale + 2), label, fill=RED, font=_font(11))
    canvas = Image.new("RGB", (size, size + 16), (255, 255, 255))
    canvas.paste(crop, (0, 16))
    ImageDraw.Draw(canvas).text((2, 1), caption, fill=(0, 0, 0), font=_font(11))
    return canvas


def grid(tiles: list[Image.Image], cols: int, path: Path, title: str) -> Path | None:
    if not tiles:
        return None
    w, h = tiles[0].size
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h + 20), (255, 255, 255))
    ImageDraw.Draw(sheet).text((4, 3), title, fill=(0, 0, 0), font=_font(13))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * w, 20 + (i // cols) * h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return path


def label(category_id: int, score: float | None = None) -> str:
    name = CLASS_NAMES[category_id - 1]
    return f"{name} {score:.2f}" if score is not None else name
