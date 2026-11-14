"""Detection and end-to-end F1 with EasyOCR.

UNITS pins torch 1.11, which does not run on these 4090s. EasyOCR is the
spotter used here, the same one on the original image, the ELIC reconstruction
(decoder RGB before fusion), and the TextBoost reconstruction.

A match needs polygon IoU >= 0.5. End-to-end also needs the same alphanumeric
string, case ignored. Large words are don't-care when scoring the small-text
subset: a prediction that only hits one of them is neither a hit nor a false
positive.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from shapely.geometry import Polygon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.config import model_config
from textboost.data import full_image_sample, load_textocr
from textboost.model import TextBoostELIC
from textboost.render import average_char_area

MODEL_DIR = "/mnt/cpfs/wbx/datasets/easyocr_models"
IOU_THRESHOLD = 0.5
SMALL_AREA = 800.0


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("textocr", "anytext"), required=True)
    parser.add_argument("--ann", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--font", default=None)
    parser.add_argument("--download-only", action="store_true")
    return parser.parse_args()


def norm_text(text):
    return "".join(ch for ch in str(text).casefold() if ch.isalnum())


def polygon_iou(pts_a, pts_b):
    try:
        poly_a = Polygon(np.asarray(pts_a, dtype=np.float64))
        poly_b = Polygon(np.asarray(pts_b, dtype=np.float64))
    except Exception:
        return 0.0
    if not poly_a.is_valid:
        poly_a = poly_a.buffer(0)
    if not poly_b.is_valid:
        poly_b = poly_b.buffer(0)
    if poly_a.is_empty or poly_b.is_empty:
        return 0.0
    union = poly_a.union(poly_b).area
    if union <= 0:
        return 0.0
    return float(poly_a.intersection(poly_b).area / union)


def load_anytext(ann_path, image_root):
    with open(ann_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    items = []
    for rec in data["data_list"]:
        name = rec["img_name"]
        path = os.path.join(image_root, name)
        if not os.path.isfile(path):
            continue
        words = []
        for ann in rec.get("annotations") or []:
            if ann.get("valid") is False or not ann.get("polygon"):
                continue
            words.append({
                "text": "" if ann.get("text") is None else str(ann["text"]),
                "points": np.asarray(ann["polygon"], dtype=np.float64).reshape(-1, 2),
            })
        items.append((name, path, words))
    return items


def load_items(args):
    if args.kind == "textocr":
        records = load_textocr(args.ann)
        from textboost.data import _index_files
        index = _index_files(args.image_root)
        items = []
        for image_id, rec in sorted(records.items()):
            path = index.get(os.path.basename(rec["file_name"]))
            if path is None:
                continue
            items.append((image_id, str(path), rec["words"]))
        return items
    return load_anytext(args.ann, args.image_root)


def ground_truth(words):
    care = []
    small = []
    large = []
    for rec in words:
        text = norm_text(rec.get("text", ""))
        pts = np.asarray(rec["points"], dtype=np.float64).reshape(-1, 2)
        if not text or len(pts) < 3:
            continue
        item = (pts, text)
        care.append(item)
        raw = str(rec.get("text", ""))
        if average_char_area(pts, raw) < SMALL_AREA:
            small.append(item)
        else:
            large.append(item)
    return care, small, large


def spot(reader, rgb):
    # EasyOCR treats a numpy array as BGR.
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    found = []
    for box, text, _conf in reader.readtext(bgr):
        cleaned = norm_text(text)
        if not cleaned:
            continue
        pts = np.asarray(box, dtype=np.float64).reshape(-1, 2)
        if len(pts) < 3:
            continue
        found.append((pts, cleaned))
    return found


def score(preds, care, ignore):
    """Return detection and end-to-end (tp, fp, fn)."""
    det_tp, det_used_pred, det_used_gt = _match(preds, care, require_text=False)
    e2e_tp, e2e_used_pred, e2e_used_gt = _match(preds, care, require_text=True)
    det_fp = _false_positives(preds, det_used_pred, ignore)
    e2e_fp = _false_positives(preds, e2e_used_pred, ignore)
    return {
        "det": [det_tp, det_fp, len(care) - len(det_used_gt)],
        "e2e": [e2e_tp, e2e_fp, len(care) - len(e2e_used_gt)],
    }


def _match(preds, care, require_text):
    pairs = []
    for i, (pred_pts, pred_text) in enumerate(preds):
        for j, (gt_pts, gt_text) in enumerate(care):
            if require_text and pred_text != gt_text:
                continue
            iou = polygon_iou(pred_pts, gt_pts)
            if iou >= IOU_THRESHOLD:
                pairs.append((iou, i, j))
    pairs.sort(reverse=True)
    used_pred = set()
    used_gt = set()
    for _iou, i, j in pairs:
        if i in used_pred or j in used_gt:
            continue
        used_pred.add(i)
        used_gt.add(j)
    return len(used_gt), used_pred, used_gt


def _false_positives(preds, used_pred, ignore):
    fp = 0
    for i, (pred_pts, _text) in enumerate(preds):
        if i in used_pred:
            continue
        if any(polygon_iou(pred_pts, gt_pts) >= IOU_THRESHOLD for gt_pts, _gt_text in ignore):
            continue
        fp += 1
    return fp


def to_rgb(tensor):
    return (
        tensor.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().permute(1, 2, 0).numpy()
    )


def make_reader(lang):
    import easyocr
    langs = ["ch_sim", "en"] if lang == "ch_sim" else ["en"]
    return easyocr.Reader(
        langs,
        gpu=True,
        model_storage_directory=MODEL_DIR,
        user_network_directory=MODEL_DIR,
        download_enabled=True,
        verbose=False,
    )


def empty_counts():
    return {"det": [0, 0, 0], "e2e": [0, 0, 0]}


def main():
    args = parse_args()
    os.makedirs(MODEL_DIR, exist_ok=True)
    if args.download_only:
        import easyocr
        langs = ["ch_sim", "en"] if args.lang == "ch_sim" else ["en"]
        easyocr.Reader(
            langs,
            gpu=False,
            model_storage_directory=MODEL_DIR,
            user_network_directory=MODEL_DIR,
            download_enabled=True,
            verbose=True,
        )
        print("downloaded", langs, flush=True)
        return

    items = load_items(args)
    items = [it for i, it in enumerate(items) if i % args.shards == args.shard]
    if args.limit:
        items = items[: args.limit]
    print("shard %d items %d" % (args.shard, len(items)), flush=True)

    device = torch.device("cuda")
    net = TextBoostELIC(model_config()).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    net.update(force=True)
    reader = make_reader(args.lang)
    font = args.font
    if font is None:
        from textboost.data import _default_font
        font = _default_font()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for name, path, words in items:
            row = {"id": name, "file": os.path.basename(path)}
            try:
                sample = full_image_sample(path, words, SMALL_AREA, font, pad=64)
                h, w = sample["size"]
                image = sample["image"].unsqueeze(0).to(device)
                guidance = sample["guidance"].unsqueeze(0).to(device)
                with torch.inference_mode():
                    compressed = net.compress(image)
                    restored = net.decompress(compressed["strings"], compressed["shape"], guidance=guidance)
                care, small, large = ground_truth(words)
                views = {
                    "orig": to_rgb(image[0, :, :h, :w]),
                    "base": to_rgb(restored["x_base"][0, :, :h, :w]),
                    "boost": to_rgb(restored["x_hat"][0, :, :h, :w]),
                }
                row["counts"] = {}
                row["small"] = {}
                for key, rgb in views.items():
                    preds = spot(reader, rgb)
                    full = score(preds, care, [])
                    tiny = score(preds, small, large)
                    row["counts"][key] = full
                    row["small"][key] = tiny
                    row["n_" + key] = len(preds)
                    if done == 0 and key == "orig":
                        row["preview_orig"] = [text for _pts, text in preds[:12]]
                row["n_gt"] = len(care)
                row["n_small"] = len(small)
                del image, guidance, restored, compressed
            except Exception as exc:
                torch.cuda.empty_cache()
                row["error"] = str(exc).splitlines()[-1][:300]
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            done += 1
            if done % 10 == 0 or done == len(items):
                print("shard %d %d/%d" % (args.shard, done, len(items)), flush=True)
    print("shard %d wrote %s" % (args.shard, args.out), flush=True)


if __name__ == "__main__":
    main()
