"""OCR boxes and strings to a geometry-aligned guidance map."""

import gzip
import json
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def polygon_area(pts):
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def average_char_area(pts, text):
    n = max(len(text.strip()), 1)
    return polygon_area(pts) / n


def _principal_angle(pts):
    best_len = -1.0
    best = 0.0
    n = len(pts)
    for i in range(n):
        v = pts[(i + 1) % n] - pts[i]
        length = float(np.hypot(v[0], v[1]))
        if length > best_len:
            best_len = length
            best = math.atan2(float(v[1]), float(v[0]))
    return best


def _frame(pts):
    """Rotate so the long side of the box lies on the horizontal axis."""
    theta = _principal_angle(pts)
    center = pts.mean(axis=0)

    def extents(angle):
        a = -angle
        rot = np.array(
            [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]],
            dtype=np.float64,
        )
        local = (pts - center) @ rot.T
        width = float(local[:, 0].max() - local[:, 0].min())
        height = float(local[:, 1].max() - local[:, 1].min())
        return width, height

    width, height = extents(theta)
    if height > width:
        theta += math.pi / 2.0
        width, height = extents(theta)
    return theta, center, max(width, 1.0), max(height, 1.0)


def _fit_font(text, width, height, font_path):
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    lo, hi = 1, max(int(height), 1)
    best = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        bbox = probe.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if tw <= width and th <= height:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return ImageFont.truetype(font_path, best)


def _render_horizontal(text, width, height, font_path):
    width = max(int(math.ceil(width)), 1)
    height = max(int(math.ceil(height)), 1)
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    font = _fit_font(text, width, height, font_path)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x = (width - tw) / 2.0 - bbox[0]
    y = (height - th) / 2.0 - bbox[1]
    draw.text((x, y), text, fill=255, font=font)
    return image


def _paste_oriented(canvas, patch, center, theta):
    """Place a horizontal patch so its +x axis follows `theta` (y down)."""
    # PIL rotates counter-clockwise in display space. Negating theta maps the
    # patch +x axis onto the box's long edge in image coordinates.
    rotated = patch.rotate(
        -math.degrees(theta),
        resample=Image.BILINEAR,
        expand=True,
        fillcolor=0,
    )
    left = int(round(center[0] - rotated.size[0] / 2.0))
    top = int(round(center[1] - rotated.size[1] / 2.0))
    canvas.paste(rotated, (left, top), rotated)


def render_guidance(width, height, instances, font_path):
    """Render filtered instances onto a black canvas.

    `instances` is a list of {"text": str, "points": array (N, 2)} in the
    canvas coordinate frame. Returns guidance (3, H, W) and mask (H, W),
    both float32 in [0, 1]. An empty list yields zeros.
    """
    guide = Image.new("L", (int(width), int(height)), 0)
    mask = Image.new("L", (int(width), int(height)), 0)
    mask_draw = ImageDraw.Draw(mask)
    for inst in instances:
        text = inst["text"].strip()
        pts = np.asarray(inst["points"], dtype=np.float64).reshape(-1, 2)
        if not text or len(pts) < 3:
            continue
        mask_draw.polygon([tuple(map(float, p)) for p in pts], fill=255)
        theta, center, box_w, box_h = _frame(pts)
        box_w = min(box_w, float(width))
        box_h = min(box_h, float(height))
        patch = _render_horizontal(text, box_w, box_h, font_path)
        _paste_oriented(guide, patch, center, theta)
    guide_np = np.asarray(guide, dtype=np.float32) / 255.0
    mask_np = np.asarray(mask, dtype=np.float32) / 255.0
    guidance = np.stack([guide_np, guide_np, guide_np], axis=0)
    return guidance, mask_np


def filter_instances(records, threshold):
    """Keep strings whose average character area is below `threshold`."""
    kept = []
    for rec in records:
        text = str(rec.get("text", "")).strip()
        pts = np.asarray(rec["points"], dtype=np.float64).reshape(-1, 2)
        if not text or len(pts) < 3:
            continue
        if average_char_area(pts, text) < float(threshold):
            kept.append({"text": text, "points": pts})
    return kept


def pack_sideinfo(instances):
    payload = []
    for inst in instances:
        pts = np.asarray(inst["points"], dtype=np.float64).reshape(-1, 2)
        payload.append({"text": inst["text"], "points": pts.round(2).tolist()})
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return gzip.compress(raw)


def sideinfo_bpp(instances, height, width):
    blob = pack_sideinfo(instances)
    return (len(blob) * 8.0) / float(height * width)
