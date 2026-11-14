#!/bin/bash
# Train an ELIC-only model. Fusion is not updated and is not the reconstruction.
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
DATA=/mnt/cpfs/wbx/datasets/textocr
RUN=/mnt/cpfs/wbx/runs/textboost/elic_matched
mkdir -p "$RUN"
export PYTHONPATH="$ROOT"
export NCCL_IB_DISABLE=1
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1

"$PY" - <<'PY'
import os, signal, time
def cmdline(pid):
    try:
        return open("/proc/%s/cmdline" % pid, "rb").read().replace(b"\x00", b" ").decode()
    except OSError:
        return ""
killed = []
for pid in os.listdir("/proc"):
    if not pid.isdigit():
        continue
    parts = cmdline(pid).split()
    if not parts or not parts[0].endswith("python"):
        continue
    if any(arg.endswith("gpu_idle_keepalive.py") or arg.endswith("train_fast.py") or arg.endswith("measure_bpp.py") for arg in parts):
        os.kill(int(pid), signal.SIGTERM)
        killed.append(pid)
print("stopped", killed, flush=True)
time.sleep(3)
PY

"$PY" -m torch.distributed.run --standalone --nproc_per_node=8 \
  "$ROOT/playground/train_fast.py" \
  --dataset "$DATA/train_images" \
  --image-root "$DATA" \
  --ann "$DATA/TextOCR_0.1_train.json" \
  --init-checkpoint /mnt/cpfs/wbx/runs/textboost/fast/init_stage1.pth.tar \
  --save-dir "$RUN" \
  --lambda 0.00033 \
  --batch-size 32 \
  --learning-rate 0.0002 \
  --aux-learning-rate 0.001 \
  --num-workers 4 \
  --stage1-max-epochs 18 \
  --stage1-only \
  --elic-only

keep
sleep 8
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
