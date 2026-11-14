# TextBoost

Reproduction of TextBoost on the ELIC backbone, with the measured rate, distortion, and OCR F1 from one low-rate checkpoint. The backbone code is the Apache-2.0 unofficial ELIC of Jiang et al. Numbers and the training schedule are in [REPRODUCTION.md](REPRODUCTION.md).

## Weights

The checkpoints that produced those numbers are on the [weights release](https://github.com/wangbxj1234/TextBoost/releases/tag/weights). Each file is a dict with `state_dict` for `TextBoostELIC` (`N=192`, `M=320`, plus the fusion block). Optimizer state is not included.

| File | What to score |
|---|---|
| [textboost_stage2.pt](https://github.com/wangbxj1234/TextBoost/releases/download/weights/textboost_stage2.pt) | Fused reconstruction `x_hat`. This is the published TextOCR, LAION, Wukong, and Kodak checkpoint. |
| [elic_matched.pt](https://github.com/wangbxj1234/TextBoost/releases/download/weights/elic_matched.pt) | Decoder RGB before fusion, `x_base`. On TextOCR its file size is 0.0305 bpp, against TextBoost's total of 0.0304 bpp. The fusion weights in this file were frozen and are not the trained TextBoost fusion. |

```bash
curl -L -o textboost_stage2.pt \
  https://github.com/wangbxj1234/TextBoost/releases/download/weights/textboost_stage2.pt
curl -L -o elic_matched.pt \
  https://github.com/wangbxj1234/TextBoost/releases/download/weights/elic_matched.pt
```

```python
ckpt = torch.load(path, map_location="cpu")
net.load_state_dict(ckpt["state_dict"])
```

Dependencies are PyTorch, CompressAI, Pillow, NumPy, and `pytorch-msssim`. EasyOCR is only needed for F1. `compress` pads each image to a multiple of 64 and expects a CUDA GPU. The side stream is the dataset's own boxes and transcripts whose average character area is below 800, gzip-compressed. English guidance uses DejaVu Sans. Chinese guidance needs a CJK font passed with `--font`.

TextOCR annotations are `TextOCR_0.1_val.json`. The validation `file_name` values point at TextVQA training images; the loader matches each one by basename under `--image-root`.

Rate and distortion of TextBoost, fused output:

```bash
PYTHONPATH=. python playground/eval_textocr.py \
  --ann /path/to/TextOCR_0.1_val.json \
  --image-root /path/to/train_images \
  --checkpoint textboost_stage2.pt \
  --out results/textocr.jsonl
```

File size of the rate-matched ELIC, in bits per pixel of the original image:

```bash
PYTHONPATH=. python playground/measure_bpp.py \
  --checkpoint elic_matched.pt \
  --ann /path/to/TextOCR_0.1_val.json \
  --image-root /path/to/train_images
```

EasyOCR F1 writes three views per image. `counts` is all text and `small` is average character area below 800. Each of `det` and `e2e` is `[tp, fp, fn]`. Sum those counts over the jsonl and compute micro F1 as `2*tp / (2*tp + fp + fn)`.

| Checkpoint | View to read |
|---|---|
| `textboost_stage2.pt` | `boost` |
| `elic_matched.pt` | `base` |

```bash
PYTHONPATH=. python playground/eval_f1.py \
  --kind textocr \
  --ann /path/to/TextOCR_0.1_val.json \
  --image-root /path/to/train_images \
  --checkpoint textboost_stage2.pt \
  --lang en \
  --out results/textocr_f1.jsonl
```

`--lang ch_sim` loads Chinese and English for Wukong. LAION and Wukong use `--kind anytext` with that benchmark's `test1k.json`. These are one low-rate operating point, scored with EasyOCR, not the paper's four bitrate points and not UNITS.

# ELIC
Unofficial Implementation of CVPR22 papar "ELIC: Efficient Learned Image Compression with Unevenly Grouped Space-Channel Contextual Adaptive Coding".

My implementation is based on CompressAI.

This implementation of ELIC is part of the MLIC project, if you find this repository useful, please cite

```
@article{jiang2022multi,
  title={Multi-Reference Entropy Model for Learned Image Compression},
  author={Jiang, Wei and Yang, Jiayu and Zhai, Yongqi and Wang, Ronggang},
  journal={arXiv preprint arXiv:2211.07273},
  year={2022}
}
```

```
@misc{jiang2022unofficialelic,
    author={Jiang, Wei},
    title={Unofficial ELIC},
    howpublished={\url{https://github.com/JiangWeibeta/ELIC}},
    year={2022}
}
```

```
@inproceedings{he2022elic,
  title={Elic: Efficient learned image compression with unevenly grouped space-channel contextual adaptive coding},
  author={He, Dailan and Yang, Ziming and Peng, Weikun and Ma, Rui and Qin, Hongwei and Wang, Yan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5718--5727},
  year={2022}
}
```
# Contact
Feel free to contact me if there is any question. (Wei Jiang, wei.jiang1999@outlook.com)
