#!/bin/bash
# EasyOCR F1 on original, ELIC decoder RGB, and TextBoost. Restores keepalive.
set -euo pipefail
keep() {
  export KEEPALIVE_GPUS=0,1,2,3,4,5,6,7
  export KEEPALIVE_GIB=28
  nohup /mnt/cpfs/wbx/envs/dit4dit_flower/bin/python -u /mnt/cpfs/wbx/code/wbx1/tools/gpu_idle_keepalive.py >> /mnt/cpfs/wbx/logs/gpu_idle_keepalive.log 2>&1 &
  echo "keepalive $!"
}
trap 'status=$?; if [ "$status" -ne 0 ]; then keep; fi' EXIT

ROOT=/mnt/cpfs/wbx/code/textboost
PY=/mnt/cpfs/wbx/envs/textboost/bin/python
CKPT=/mnt/cpfs/wbx/runs/textboost/fast/stage2/checkpoint_best.pth.tar
OUT=/mnt/cpfs/wbx/runs/textboost/fast/eval_f1
BASE=/mnt/cpfs/wbx/datasets/anytext_benchmark/unpacked/benchmark
FONT_ZH=/usr/share/fonts/truetype/wqy/wqy-microhei.ttc
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"

"$PY" - <<'PY'
import os, signal
def cmdline(pid):
    try:
        return open("/proc/%s/cmdline" % pid, "rb").read().replace(b"\x00", b" ").decode()
    except OSError:
        return ""
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    parts = cmdline(pid).split()
    if not parts or not parts[0].endswith("python"):
        continue
    if any(arg.endswith("gpu_idle_keepalive.py") or arg.endswith("eval_f1.py") for arg in parts):
        os.kill(int(pid), signal.SIGTERM)
        print("stopped", pid, parts[-1], flush=True)
PY
sleep 3

run_set() {
  local name="$1" kind="$2" lang="$3" ann="$4" images="$5" font="$6"
  local dir="$OUT/$name"
  mkdir -p "$dir"
  echo "==== $name ===="
  local extra=(--kind "$kind" --lang "$lang" --ann "$ann" --image-root "$images" --checkpoint "$CKPT")
  if [ -n "$font" ]; then extra+=(--font "$font"); fi
  local pids=()
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i "$PY" -u "$ROOT/playground/eval_f1.py" \
      "${extra[@]}" --out "$dir/shard_${i}.jsonl" --shard "$i" --shards 8 \
      > "$dir/shard_${i}.log" 2>&1 &
    pids+=($!)
    echo "shard $i pid ${pids[-1]}"
  done
  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [ "$failed" -ne 0 ]; then
    echo "$name shard failed"
    exit 1
  fi
}

run_set textocr textocr en \
  /mnt/cpfs/wbx/datasets/textocr/TextOCR_0.1_val.json \
  /mnt/cpfs/wbx/datasets/textocr/train_images ""
run_set laion anytext en \
  "$BASE/laion_word/test1k.json" "$BASE/laion_word/imgs" ""
run_set wukong anytext ch_sim \
  "$BASE/wukong_word/test1k.json" "$BASE/wukong_word/imgs" "$FONT_ZH"

"$PY" - <<'PY'
import json, os
root = "/mnt/cpfs/wbx/runs/textboost/fast/eval_f1"

def f1(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": score}

for name in ("textocr", "laion", "wukong"):
    rows = []
    folder = os.path.join(root, name)
    for i in range(8):
        with open(os.path.join(folder, "shard_%d.jsonl" % i), encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
    ok = [r for r in rows if "counts" in r]
    err = [r for r in rows if "error" in r]
    summary = {"images": len(rows), "ok": len(ok), "errors": len(err)}
    if err:
        summary["error_example"] = err[0]["error"]
    for bucket, key in (("all", "counts"), ("small", "small")):
        summary[bucket] = {}
        for view in ("orig", "base", "boost"):
            summary[bucket][view] = {}
            for task in ("det", "e2e"):
                tp = fp = fn = 0
                for row in ok:
                    a, b, c = row[key][view][task]
                    tp += a
                    fp += b
                    fn += c
                summary[bucket][view][task] = f1(tp, fp, fn)
    dest = os.path.join(folder, "summary.json")
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(name, json.dumps(summary))
PY

keep
sleep 8
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
