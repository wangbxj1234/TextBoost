"""File bpp of an ELIC decode, using the bitstream size rather than the loss."""

import argparse
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.config import model_config
from textboost.data import full_image_sample, load_textocr, _index_files
from textboost.model import TextBoostELIC


def stream_bytes(obj):
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, str):
        return len(obj.encode("utf-8"))
    return sum(stream_bytes(item) for item in obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    device = torch.device("cuda")
    net = TextBoostELIC(model_config()).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    net.update(force=True)
    records = load_textocr(args.ann)
    index = _index_files(args.image_root)
    items = []
    for _image_id, rec in sorted(records.items()):
        path = index.get(os.path.basename(rec["file_name"]))
        if path is not None:
            items.append(str(path))
    items = [path for i, path in enumerate(items) if i % args.shards == args.shard]
    if args.limit:
        items = items[: args.limit]
    bits = 0.0
    pixels = 0.0
    for path in items:
        sample = full_image_sample(path, [], 800.0, None, pad=64)
        h, w = sample["size"]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.inference_mode():
            compressed = net.compress(image)
        bits += stream_bytes(compressed["strings"]) * 8.0
        pixels += float(h * w)
        del image, compressed
    print("images %d bits %.0f pixels %.0f bpp %.6f" % (len(items), bits, pixels, bits / pixels), flush=True)


if __name__ == "__main__":
    main()
