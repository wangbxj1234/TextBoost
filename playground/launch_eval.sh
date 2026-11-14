#!/bin/bash
# Free the cards, check one image, then evaluate TextOCR val on all 8 GPUs.
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
OUT=/mnt/cpfs/wbx/runs/textboost/fast/eval
ANN=/mnt/cpfs/wbx/datasets/textocr/TextOCR_0.1_val.json
IMAGES=/mnt/cpfs/wbx/datasets/textocr/train_images
CKPT=/mnt/cpfs/wbx/runs/textboost/fast/stage2/checkpoint_best.pth.tar
mkdir -p "$OUT"
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
    if any(arg.endswith("gpu_idle_keepalive.py") or arg.endswith("eval_textocr.py") for arg in parts):
        os.kill(int(pid), signal.SIGTERM)
        print("stopped", pid, parts[-1], flush=True)
PY
sleep 2
ONLY=$("$PY" - <<'PY'
import json
name=""
for i in range(8):
    path="/mnt/cpfs/wbx/runs/textboost/fast/eval/shard_%d.jsonl"%i
    try:
        lines=open(path)
    except FileNotFoundError:
        continue
    for line in lines:
        row=json.loads(line)
        if "error" in row and "size of tensor" in row["error"]:
            name=row["file"]
            break
    if name:
        break
print(name)
PY
)
echo "retry $ONLY"
rm -f "$OUT"/shard_*.jsonl "$OUT"/smoke.jsonl
if [ -z "$ONLY" ]; then
  echo "no failed file to retry"
  exit 1
fi
CUDA_VISIBLE_DEVICES=0 "$PY" -u "$ROOT/playground/eval_textocr.py" \
  --ann "$ANN" --image-root "$IMAGES" --checkpoint "$CKPT" \
  --out "$OUT/smoke.jsonl" --only "$ONLY"
echo "==== smoke ===="
cat "$OUT/smoke.jsonl"
grep -q '"psnr"' "$OUT/smoke.jsonl"

for i in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$i nohup "$PY" -u "$ROOT/playground/eval_textocr.py" \
    --ann "$ANN" --image-root "$IMAGES" --checkpoint "$CKPT" \
    --out "$OUT/shard_${i}.jsonl" --shard "$i" --shards 8 \
    > "$OUT/shard_${i}.log" 2>&1 &
  echo "shard $i pid $!"
done
