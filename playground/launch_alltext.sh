#!/bin/bash
# Same checkpoint, but every annotated word goes into the side stream.
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
BASE=/mnt/cpfs/wbx/datasets/anytext_benchmark/unpacked/benchmark
OUT=/mnt/cpfs/wbx/runs/textboost/fast/eval_more
FONT_ZH=/usr/share/fonts/truetype/wqy/wqy-microhei.ttc
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

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
    if any(arg.endswith("gpu_idle_keepalive.py") or arg.endswith("eval_sets.py") for arg in parts):
        os.kill(int(pid), signal.SIGTERM)
        print("stopped", pid, parts[-1], flush=True)
PY
sleep 3

run_set() {
  local name="$1" ann="$2" images="$3" font="$4"
  local dir="$OUT/$name"
  mkdir -p "$dir"
  echo "==== $name ===="
  local pids=()
  for i in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$i "$PY" -u "$ROOT/playground/eval_sets.py" \
      --kind anytext --image-root "$images" --checkpoint "$CKPT" \
      --ann "$ann" --font "$font" --area-threshold 1000000000 \
      --out "$dir/shard_${i}.jsonl" --shard "$i" --shards 8 \
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

run_set laion_all "$BASE/laion_word/test1k.json" "$BASE/laion_word/imgs" /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
run_set wukong_all "$BASE/wukong_word/test1k.json" "$BASE/wukong_word/imgs" "$FONT_ZH"

"$PY" - <<'PY'
import json, os
root = "/mnt/cpfs/wbx/runs/textboost/fast/eval_more"
for name in ("laion_all", "wukong_all"):
    rows = []
    folder = os.path.join(root, name)
    for i in range(8):
        path = os.path.join(folder, "shard_%d.jsonl" % i)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
    ok = [r for r in rows if "psnr" in r]
    err = [r for r in rows if "error" in r]
    pixels = sum(r["height"] * r["width"] for r in ok)
    def wmean(key):
        return sum(r[key] * r["height"] * r["width"] for r in ok) / pixels
    msssim = [r["msssim"] for r in ok if r.get("msssim") is not None]
    small = [r["psnr_smalltext"] for r in ok if r.get("psnr_smalltext") is not None]
    summary = {
        "images": len(rows),
        "ok": len(ok),
        "errors": len(err),
        "bpp_image": wmean("bpp_image"),
        "bpp_side": wmean("bpp_side"),
        "bpp": wmean("bpp"),
        "psnr": sum(r["psnr"] for r in ok) / len(ok),
        "msssim": sum(msssim) / len(msssim) if msssim else None,
        "msssim_n": len(msssim),
        "psnr_text": sum(small) / len(small) if small else None,
        "images_with_text": len(small),
        "kept_instances": sum(r["instances"] for r in ok),
        "area_threshold": 1e9,
    }
    if err:
        summary["error_example"] = err[0]["error"]
    dest = os.path.join(folder, "summary.json")
    with open(dest, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(name, json.dumps(summary))
PY

keep
sleep 8
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
