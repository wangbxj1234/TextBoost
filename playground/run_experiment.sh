#!/bin/bash
# Eight ultra-low-rate TextBoost runs, one per GPU.
# Stage 1: 1000 epochs, Adam 1e-4, batch 32, crop 256, zero guidance.
# Stage 2: 20 epochs, Adam 1e-3, alpha 10, T_train 150, fusion block only.
# Lambdas use this ELIC code's 255^2 MSE Lagrangian and cover the paper's
# ultra-low bitrate region. ImageNet is not mounted here; stage 1 trains on
# the TextOCR training images.
set -euo pipefail

ROOT=/mnt/cpfs/wbx/code/textboost
PY=/mnt/cpfs/wbx/envs/textboost/bin/python
DATA=/mnt/cpfs/wbx/datasets/textocr
RUNS=/mnt/cpfs/wbx/runs/textboost
ANN=$DATA/TextOCR_0.1_train.json
STAGE1_IMAGES=${1:-$DATA/train}

cd "$ROOT"
export PYTHONPATH="$ROOT"

ZIP=$DATA/train_val_images.zip
echo "waiting for $ZIP to finish"
while [ ! -f "$ZIP" ] || [ "$(stat -c%s "$ZIP" 2>/dev/null || echo 0)" -lt 7000000000 ]; do
  sleep 60
done
if ! find "$DATA" -name 'a4ea732cd3d5948a.jpg' | grep -q .; then
  if ! pgrep -x unzip >/dev/null 2>&1; then
    echo "unzipping TextOCR images"
    unzip -q -o "$ZIP" -d "$DATA"
  fi
  echo "waiting for image extract"
  while ! find "$DATA" -name 'a4ea732cd3d5948a.jpg' | grep -q .; do
    sleep 20
  done
fi
# Let an in-flight unzip finish so the training set is complete.
while pgrep -x unzip >/dev/null 2>&1; do
  sleep 20
done
SAMPLE=$(find "$DATA" -name 'a4ea732cd3d5948a.jpg' | head -1)
if [ -z "$SAMPLE" ]; then
  echo "TextOCR image a4ea732cd3d5948a.jpg not found after unzip" >&2
  exit 1
fi
if [ ! -d "$STAGE1_IMAGES" ]; then
  STAGE1_IMAGES=$(dirname "$SAMPLE")
fi
echo "stage1 images: $STAGE1_IMAGES"
while [ ! -f "$ANN" ]; do
  sleep 10
done

if pgrep -f gpu_idle_keepalive.py >/dev/null 2>&1; then
  echo "stopping gpu keepalive before training"
  pkill -f gpu_idle_keepalive.py || true
  sleep 8
fi

LAMBDAS=(0.0001 0.00018 0.0003 0.0005 0.0008 0.0012 0.0018 0.0035)
for i in 0 1 2 3 4 5 6 7; do
  lam=${LAMBDAS[$i]}
  tag=$(printf "l%s" "$lam")
  s1=$RUNS/$tag/stage1
  s2=$RUNS/$tag/stage2
  mkdir -p "$s1" "$s2"
  if grep -q "saved stage2 epoch 20" "$s2/train_stage2.log" 2>/dev/null; then
    echo "skip gpu $i $tag"
    continue
  fi
  CUDA_VISIBLE_DEVICES=$i nohup bash -c "
    set -euo pipefail
    export PYTHONPATH='$ROOT'
    if ! grep -q 'saved epoch 1000' '$s1/train_stage1.log' 2>/dev/null; then
      '$PY' -u '$ROOT/playground/train_stage1.py' \
        --dataset '$STAGE1_IMAGES' \
        --experiment '$tag' \
        --save-dir '$s1' \
        --lambda $lam \
        --epochs 1000 \
        --learning-rate 1e-4 \
        --batch-size 32 \
        --patch-size 256 \
        --num-workers 4 \
        --checkpoint '$s1/checkpoint_last.pth.tar'
    fi
    '$PY' -u '$ROOT/playground/train_stage2.py' \
      --image-root '$DATA' \
      --ann '$ANN' \
      --checkpoint '$s1/checkpoint_last.pth.tar' \
      --resume '$s2/checkpoint_last.pth.tar' \
      --experiment '$tag' \
      --save-dir '$s2' \
      --lambda $lam \
      --alpha 10 \
      --area-threshold 150 \
      --epochs 20 \
      --learning-rate 1e-3 \
      --batch-size 32 \
      --patch-size 256 \
      --num-workers 4
  " > "$RUNS/$tag/driver.log" 2>&1 &
  echo "gpu $i pid $! lambda $lam"
done
echo "launched"
