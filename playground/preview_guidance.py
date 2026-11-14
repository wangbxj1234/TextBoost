"""Save a few TextOCR guidance maps next to the source image."""

import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from textboost.data import _default_font, _index_files, load_textocr
from textboost.render import filter_instances, render_guidance

ANN = "/mnt/cpfs/wbx/datasets/textocr/TextOCR_0.1_val.json"
IMAGES = "/mnt/cpfs/wbx/datasets/textocr/train_images"
EVAL = "/mnt/cpfs/wbx/runs/textboost/fast/eval"
OUT = "/mnt/cpfs/wbx/runs/textboost/fast/preview"
THRESHOLD = 800.0


def panel(image, longest=960):
    w, h = image.size
    scale = longest / max(w, h)
    if scale < 1:
        image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    return image


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for shard in range(8):
        with open(os.path.join(EVAL, "shard_%d.jsonl" % shard), encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("instances", 0) >= 8:
                    rows.append(row)
    rows.sort(key=lambda row: row["instances"], reverse=True)
    picked = rows[:4]
    records = load_textocr(ANN)
    index = _index_files(IMAGES)
    by_name = {os.path.basename(rec["file_name"]): (image_id, rec) for image_id, rec in records.items()}
    font = _default_font()
    label_font = ImageFont.truetype(font, 28)
    for row in picked:
        image_id, rec = by_name[row["file"]]
        path = index[row["file"]]
        image = Image.open(path).convert("RGB")
        kept = filter_instances(rec["words"], THRESHOLD)
        guidance, _mask = render_guidance(image.size[0], image.size[1], kept, font)
        guide = Image.fromarray((guidance.transpose(1, 2, 0) * 255.0).astype(np.uint8))
        left = panel(image)
        right = panel(guide)
        canvas = Image.new("RGB", (left.size[0] + right.size[0] + 8, max(left.size[1], right.size[1]) + 40), (20, 20, 20))
        canvas.paste(left, (0, 40))
        canvas.paste(right, (left.size[0] + 8, 40))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 6), "original  %s" % row["file"], fill=(230, 230, 230), font=label_font)
        draw.text((left.size[0] + 16, 6), "guidance  kept %d" % len(kept), fill=(230, 230, 230), font=label_font)
        dest = os.path.join(OUT, os.path.splitext(row["file"])[0] + "_guidance.jpg")
        canvas.save(dest, quality=90)
        print(dest, image.size, "kept", len(kept), flush=True)


if __name__ == "__main__":
    main()
