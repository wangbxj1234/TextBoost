"""Evaluate one TextBoost checkpoint on the TextOCR validation split.

The side stream is that image's own small-text strings and polygons, filtered
at T_test=800 and gzip-compressed. An image with none sends nothing. Bitrate
is the arithmetic-coded image stream plus that side stream.
"""

import argparse
import json
import math
import os
import sys
import traceback

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.config import model_config
from textboost.data import _default_font, _index_files, full_image_sample, load_textocr
from textboost.model import TextBoostELIC
from textboost.render import pack_sideinfo


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ann", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--area-threshold", type=float, default=800.0)
    p.add_argument("--font", default=None)
    p.add_argument("--only", default="", help="Evaluate one file name")
    return p.parse_args()


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


def main():
    args = parse_args()
    device = torch.device("cuda")
    font = args.font or _default_font()
    records = load_textocr(args.ann)
    index = _index_files(args.image_root)
    items = []
    for image_id, rec in sorted(records.items()):
        name = os.path.basename(rec["file_name"])
        path = index.get(name)
        if path is None:
            continue
        items.append((image_id, path, rec["words"]))
    if args.only:
        items = [it for it in items if args.only in os.path.basename(str(it[1]))]
    else:
        items = [it for i, it in enumerate(items) if i % args.shards == args.shard]
    if args.limit:
        items = items[: args.limit]

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
        for n, (image_id, path, words) in enumerate(items):
            row = {"id": image_id, "file": os.path.basename(str(path))}
            try:
                sample = full_image_sample(path, words, args.area_threshold, font, pad=64)
                h, w = sample["size"]
                image = sample["image"].unsqueeze(0).to(device)
                guidance = sample["guidance"].unsqueeze(0).to(device)
                with torch.inference_mode():
                    compressed = net.compress(image)
                    restored = net.decompress(compressed["strings"], compressed["shape"], guidance=guidance)
                hat = restored["x_hat"][:, :, :h, :w].clamp(0, 1)
                src = image[:, :, :h, :w]
                mse = torch.mean((hat - src) ** 2).item()
                image_bytes = stream_bytes(compressed["strings"])
                side_bytes = len(pack_sideinfo(sample["instances"]))
                pixels = float(h * w)
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
            if done % 20 == 0 or done == len(items):
                print("shard %d %d/%d" % (args.shard, done, len(items)), flush=True)
    print("shard %d wrote %s" % (args.shard, args.out), flush=True)


if __name__ == "__main__":
    main()
