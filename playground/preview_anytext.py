"""Original | small-text guidance | all annotated text, for one LAION and one Wukong image."""

import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from textboost.render import filter_instances, render_guidance

BASE = "/mnt/cpfs/wbx/datasets/anytext_benchmark/unpacked/benchmark"
OUT = "/mnt/cpfs/wbx/runs/textboost/fast/preview"
FONT_EN = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_ZH = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"


def words_of(rec):
    words = []
    for ann in rec.get("annotations") or []:
        if ann.get("valid") is False or not ann.get("polygon"):
            continue
        words.append({
            "text": "" if ann.get("text") is None else str(ann["text"]),
            "points": np.asarray(ann["polygon"], dtype=np.float64).reshape(-1, 2),
        })
    return words


def panel(image, longest=512):
    w, h = image.size
    scale = longest / max(w, h)
    if scale < 1:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return image


def guide_image(path, words, font):
    image = Image.open(path).convert("RGB")
    guidance, _mask = render_guidance(image.size[0], image.size[1], words, font)
    guide = Image.fromarray((guidance.transpose(1, 2, 0) * 255.0).astype(np.uint8))
    return image, guide


def main():
    os.makedirs(OUT, exist_ok=True)
    label_font = ImageFont.truetype(FONT_EN, 22)
    jobs = [
        ("laion_word", FONT_EN, "laion"),
        ("wukong_word", FONT_ZH, "wukong"),
    ]
    for folder, font, tag in jobs:
        data = json.load(open(os.path.join(BASE, folder, "test1k.json"), encoding="utf-8"))
        rec = max(data["data_list"], key=lambda item: len(item.get("annotations") or []))
        path = os.path.join(BASE, folder, "imgs", rec["img_name"])
        words = words_of(rec)
        small = filter_instances(words, 800.0)
        image, guide_all = guide_image(path, words, font)
        _, guide_small = guide_image(path, small, font)
        tiles = [panel(image), panel(guide_small), panel(guide_all)]
        labels = [
            "original  %d words" % len(words),
            "T=800 kept %d" % len(small),
            "all annotated text",
        ]
        gap = 8
        width = sum(tile.size[0] for tile in tiles) + gap * (len(tiles) - 1)
        height = max(tile.size[1] for tile in tiles) + 36
        canvas = Image.new("RGB", (width, height), (20, 20, 20))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for tile, label in zip(tiles, labels):
            canvas.paste(tile, (x, 36))
            draw.text((x + 6, 6), label, fill=(230, 230, 230), font=label_font)
            x += tile.size[0] + gap
        dest = os.path.join(OUT, "%s_%s_guidance.jpg" % (tag, os.path.splitext(rec["img_name"])[0][:12]))
        canvas.save(dest, quality=90)
        print(dest, "words", len(words), "kept", len(small), flush=True)


if __name__ == "__main__":
    main()
