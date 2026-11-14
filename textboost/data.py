"""Image crops and TextOCR guidance."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from textboost.render import filter_instances, render_guidance

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(root):
    root = Path(root)
    files = [
        p
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and not p.name.startswith(".")
    ]
    if not files:
        raise FileNotFoundError(f"no images under {root}")
    return sorted(files)


def _open_rgb(path):
    with Image.open(path) as img:
        return img.convert("RGB")


def _fit_min_side(image, patch):
    w, h = image.size
    scale = 1.0
    if min(w, h) < patch:
        scale = patch / float(min(w, h))
        nw = max(int(round(w * scale)), patch)
        nh = max(int(round(h * scale)), patch)
        image = image.resize((nw, nh), Image.BILINEAR)
    return image, scale


def _random_window(width, height, patch):
    left = 0 if width == patch else random.randint(0, width - patch)
    top = 0 if height == patch else random.randint(0, height - patch)
    return left, top


class CroppedImageFolder(Dataset):
    """Stage-1 images. Random 256 crop, no OCR guidance."""

    def __init__(self, root, patch=256):
        self.files = list_images(root)
        self.patch = int(patch)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, _ = _fit_min_side(_open_rgb(self.files[index]), self.patch)
        left, top = _random_window(image.size[0], image.size[1], self.patch)
        crop = image.crop((left, top, left + self.patch, top + self.patch))
        tensor = torch.from_numpy(np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return tensor


def _ann_text(ann):
    for key in ("utf8_string", "text", "transcription", "word"):
        if key in ann and ann[key] is not None:
            return str(ann[key])
    return ""


def _ann_points(ann):
    if "points" in ann and ann["points"]:
        return np.asarray(ann["points"], dtype=np.float64).reshape(-1, 2)
    bbox = ann.get("bbox")
    if bbox is not None and len(bbox) >= 4:
        x, y, w, h = [float(v) for v in bbox[:4]]
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)
    raise KeyError("annotation has neither points nor bbox")


def load_textocr(ann_path):
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    imgs = data["imgs"]
    anns = data["anns"]
    img_to_anns = data.get("imgToAnns") or {}
    records = {}
    for image_id, meta in imgs.items():
        file_name = meta.get("file_name") or meta.get("filename") or ""
        ann_ids = img_to_anns.get(image_id, [])
        words = []
        for ann_id in ann_ids:
            ann = anns[ann_id]
            text = _ann_text(ann)
            try:
                pts = _ann_points(ann)
            except KeyError:
                continue
            words.append({"text": text, "points": pts})
        records[str(image_id)] = {
            "file_name": file_name,
            "words": words,
        }
    return records


def _index_files(root):
    index = {}
    for path in list_images(root):
        index.setdefault(path.name, path)
    return index


class TextOCRCrops(Dataset):
    """TextOCR crops with a guidance map and a text-region mask.

    Average character area is measured on the original annotation, then the
    image and the kept boxes are cropped together.
    """

    def __init__(self, image_root, ann_path, patch=256, area_threshold=150.0, font_path=None):
        self.patch = int(patch)
        self.area_threshold = float(area_threshold)
        self.font_path = font_path or _default_font()
        self.file_index = _index_files(image_root)
        records = load_textocr(ann_path)
        self.samples = []
        for image_id, rec in records.items():
            path = self._resolve(rec["file_name"], image_id)
            if path is None:
                continue
            self.samples.append((path, rec["words"]))
        if not self.samples:
            raise FileNotFoundError(
                f"no TextOCR images from {ann_path} were found under {image_root}"
            )

    def _resolve(self, file_name, image_id):
        name = os.path.basename(file_name) if file_name else ""
        if name and name in self.file_index:
            return self.file_index[name]
        for candidate in (image_id, image_id + ".jpg", image_id + ".png"):
            if candidate in self.file_index:
                return self.file_index[candidate]
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, words = self.samples[index]
        image = _open_rgb(path)
        image, scale = _fit_min_side(image, self.patch)
        left, top = _random_window(image.size[0], image.size[1], self.patch)
        crop = image.crop((left, top, left + self.patch, top + self.patch))
        kept = filter_instances(words, self.area_threshold)
        shifted = []
        for inst in kept:
            pts = inst["points"] * scale
            pts = pts - np.array([left, top], dtype=np.float64)
            if pts[:, 0].max() < 0 or pts[:, 1].max() < 0:
                continue
            if pts[:, 0].min() >= self.patch or pts[:, 1].min() >= self.patch:
                continue
            shifted.append({"text": inst["text"], "points": pts})
        guidance, mask = render_guidance(self.patch, self.patch, shifted, self.font_path)
        image_t = torch.from_numpy(np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255.0)
        return {
            "image": image_t,
            "guidance": torch.from_numpy(guidance.copy()),
            "mask": torch.from_numpy(mask.copy()),
        }


def full_image_sample(path, words, area_threshold, font_path, pad=16):
    """Full-resolution sample padded to a multiple of `pad`."""
    image = _open_rgb(path)
    w, h = image.size
    kept = filter_instances(words, area_threshold)
    guidance, mask = render_guidance(w, h, kept, font_path)
    pad_w = (pad - w % pad) % pad
    pad_h = (pad - h % pad) % pad
    if pad_w or pad_h:
        canvas = Image.new("RGB", (w + pad_w, h + pad_h))
        canvas.paste(image, (0, 0))
        image = canvas
        g = np.zeros((3, h + pad_h, w + pad_w), dtype=np.float32)
        m = np.zeros((h + pad_h, w + pad_w), dtype=np.float32)
        g[:, :h, :w] = guidance
        m[:h, :w] = mask
        guidance, mask = g, m
    image_t = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0)
    return {
        "image": image_t,
        "guidance": torch.from_numpy(guidance.copy()),
        "mask": torch.from_numpy(mask.copy()),
        "size": (h, w),
        "path": str(path),
        "instances": kept,
    }


def _default_font():
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("no TrueType font found for text rendering")
