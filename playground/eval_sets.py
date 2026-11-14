"""Evaluate the stage-2 checkpoint on a folder of images.

AnyText records carry polygons and transcripts. Kodak has no text
annotation, so the guidance tensor stays zero and the side stream is empty.
"""

import argparse
import json
import math
import os
import sys
import traceback

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.config import model_config
from textboost.data import _default_font, full_image_sample, list_images
from textboost.model import TextBoostELIC
from textboost.render import pack_sideinfo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("anytext", "images"), required=True)
    parser.add_argument("--ann", default="")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--area-threshold", type=float, default=800.0)
    parser.add_argument("--font", default=None)
    parser.add_argument("--only", default="")
    return parser.parse_args()


def stream_bytes(obj):
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))
    total = 0
    for item in obj:
        total += stream_bytes(item)
    return total


def psnr(mse):
    if mse <= 0:
        return 99.0
    return 10.0 * math.log10(1.0 / mse)


def load_anytext(ann_path, image_root):
    with open(ann_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    items = []
    missing = 0
    for rec in data["data_list"]:
        name = rec["img_name"]
        path = os.path.join(image_root, name)
        if not os.path.isfile(path):
            missing += 1
            continue
        words = []
        for ann in rec.get("annotations") or []:
            if ann.get("valid") is False:
                continue
            polygon = ann.get("polygon")
            if not polygon:
                continue
            words.append({
                "text": "" if ann.get("text") is None else str(ann["text"]),
                "points": np.asarray(polygon, dtype=np.float64).reshape(-1, 2),
            })
        items.append((name, path, words))
    if missing:
        print("missing images %d" % missing, flush=True)
    return items


def load_images(image_root):
    return [(path.name, str(path), []) for path in list_images(image_root)]


def main():
    args = parse_args()
    device = torch.device("cuda")
    font = args.font or _default_font()
    if args.kind == "anytext":
        items = load_anytext(args.ann, args.image_root)
    else:
        items = load_images(args.image_root)
    if args.only:
        items = [it for it in items if args.only in os.path.basename(it[1])]
    else:
        items = [it for i, it in enumerate(items) if i % args.shards == args.shard]
    if args.limit:
        items = items[: args.limit]
    print("shard %d items %d font %s" % (args.shard, len(items), font), flush=True)

    net = TextBoostELIC(model_config()).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    net.update(force=True)

    try:
        from pytorch_msssim import ms_ssim
    except ImportError:
        ms_ssim = None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for name, path, words in items:
            row = {"id": name, "file": os.path.basename(path)}
            try:
                sample = full_image_sample(path, words, args.area_threshold, font, pad=64)
                h, w = sample["size"]
                image = sample["image"].unsqueeze(0).to(device)
                guidance = sample["guidance"].unsqueeze(0).to(device)
                with torch.inference_mode():
                    compressed = net.compress(image)
                    restored = net.decompress(
                        compressed["strings"], compressed["shape"], guidance=guidance
                    )
                hat = restored["x_hat"][:, :, :h, :w].clamp(0, 1)
                src = image[:, :, :h, :w]
                mse = torch.mean((hat - src) ** 2).item()
                image_bytes = stream_bytes(compressed["strings"])
                side_bytes = len(pack_sideinfo(sample["instances"]))
                pixels = float(h * w)
                mask = sample["mask"][:h, :w].to(device)
                text_mse = None
                if float(mask.sum()) > 0:
                    err = (hat - src) ** 2
                    text_mse = (err * mask.unsqueeze(0)).sum().item() / (float(mask.sum()) * 3.0)
                msssim = None
                if ms_ssim is not None and min(h, w) >= 160:
                    msssim = float(ms_ssim(hat, src, data_range=1.0, size_average=True))
                row.update({
                    "height": h,
                    "width": w,
                    "instances": len(sample["instances"]),
                    "bpp_image": (image_bytes * 8.0) / pixels,
                    "bpp_side": (side_bytes * 8.0) / pixels,
                    "bpp": ((image_bytes + side_bytes) * 8.0) / pixels,
                    "psnr": psnr(mse),
                    "msssim": msssim,
                    "psnr_smalltext": None if text_mse is None else psnr(text_mse),
                })
                del image, guidance, hat, src, restored, compressed
            except RuntimeError as exc:
                torch.cuda.empty_cache()
                row["error"] = str(exc).splitlines()[-1][:300]
            except Exception:
                row["error"] = traceback.format_exc().splitlines()[-1][:300]
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            done += 1
            if done % 25 == 0 or done == len(items):
                print("shard %d %d/%d" % (args.shard, done, len(items)), flush=True)
    print("shard %d wrote %s" % (args.shard, args.out), flush=True)


if __name__ == "__main__":
    main()
