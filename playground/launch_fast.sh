#!/bin/bash
# Replace the single-GPU runs with one 8-GPU job. Kill by process image, then start.
set -euo pipefail
ROOT=/mnt/cpfs/wbx/code/textboost
PY=/mnt/cpfs/wbx/envs/textboost/bin/python
RUN=/mnt/cpfs/wbx/runs/textboost/fast
DATA=/mnt/cpfs/wbx/datasets/textocr
mkdir -p "$RUN/stage1" "$RUN/stage2"
cp -a /mnt/cpfs/wbx/runs/textboost/l0.0001/stage1/checkpoint_last.pth.tar "$RUN/init_stage1.pth.tar"
"$PY" - <<'PY'
import torch
path = "/mnt/cpfs/wbx/runs/textboost/fast/init_stage1.pth.tar"
ckpt = torch.load(path, map_location="cpu")
print("init_ok", ckpt.get("epoch"), len(ckpt["state_dict"]), flush=True)
PY
"$PY" - <<'PY'
import os
import signal
import time

def cmdline(pid):
    try:
        return open("/proc/%s/cmdline" % pid, "rb").read().replace(b"\x00", b" ").decode()
    except OSError:
        return ""

def matches(parts):
    if not parts:
        return False
    if parts[0].endswith("python") and any(
        arg.endswith("train_stage1.py")
        or arg.endswith("train_stage2.py")
        or arg.endswith("train_fast.py")
        or arg.endswith("gpu_idle_keepalive.py")
        for arg in parts
    ):
        return True
    if parts[0].endswith("bash") and any(
        ("train_stage1.py" in arg) or ("train_stage2.py" in arg) for arg in parts
    ):
        return True
    return False

killed = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    parts = cmdline(pid).split()
    if matches(parts):
        os.kill(int(pid), signal.SIGTERM)
        killed.append(int(pid))
print("term", killed, flush=True)
time.sleep(4)
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    parts = cmdline(pid).split()
    if matches(parts):
        os.kill(int(pid), signal.SIGKILL)
        print("kill", pid, flush=True)
PY
rm -f "$RUN/stage1/checkpoint_last.pth.tar" "$RUN/stage1/checkpoint_best.pth.tar" "$RUN/stage1/checkpoint_last.pth.tar.tmp" "$RUN/stage1/checkpoint_best.pth.tar.tmp"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
nohup "$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$ROOT/playground/train_fast.py" \
  --dataset "$DATA/train_images" \
  --image-root "$DATA" \
  --ann "$DATA/TextOCR_0.1_train.json" \
  --init-checkpoint "$RUN/init_stage1.pth.tar" \
  --save-dir "$RUN" \
  --lambda 0.0001 \
  --batch-size 32 \
  --learning-rate 0.0002 \
  --aux-learning-rate 0.001 \
  --stage2-learning-rate 0.002 \
  --num-workers 4 \
  > "$RUN/driver.log" 2>&1 &
echo "launched $!"
