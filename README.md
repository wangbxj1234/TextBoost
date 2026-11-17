# TextBoost

Reproduction of TextBoost on the ELIC backbone. `textboost_stage2.pt` is the checkpoint behind the measured rate, distortion, and EasyOCR F1. Four more stage-2 checkpoints, `textboost_t018.pt` through `textboost_t033.pt`, are on the same release. The backbone code is the Apache-2.0 unofficial ELIC of Jiang et al. Numbers and the training schedule are in [REPRODUCTION.md](REPRODUCTION.md).

## Weights

The checkpoints that produced those numbers are on the [weights release](https://github.com/wangbxj1234/TextBoost/releases/tag/weights). Each file is a dict with `state_dict` for `TextBoostELIC` (`N=192`, `M=320`, plus the fusion block). Optimizer state is not included.

| File | What to score |
|---|---|
| [textboost_stage2.pt](https://github.com/wangbxj1234/TextBoost/releases/download/weights/textboost_stage2.pt) | Fused reconstruction `x_hat`. Full TextOCR val total is 0.0234 bpp (image 0.0162 + compressed small text 0.0072). This is the published TextOCR, LAION, Wukong, and Kodak checkpoint. |
| [elic_matched.pt](https://github.com/wangbxj1234/TextBoost/releases/download/weights/elic_matched.pt) | Decoder RGB before fusion, `x_base`. On TextOCR its file size is 0.0305 bpp. TextBoost's total on the same set is 0.0234 bpp. The fusion weights in this file were frozen and are not the trained TextBoost fusion. |
| [textboost_t018.pt](https://github.com/wangbxj1234/TextBoost/releases/download/weights/textboost_t018.pt) | λ = 0.0000055. Full TextOCR val total 0.0097 bpp (image 0.0026 + compressed small text 0.0072). PSNR 19.23. |
| [textboost_t024.pt](https://github.com/wangbxj1234/TextBoost/releases/download/weights/textboost_t024.pt) | λ = 0.000037. Full TextOCR val total 0.0159 bpp (image 0.0087 + compressed small text 0.0072). PSNR 22.88. |
| [textboost_t029.pt](https://github.com/wangbxj1234/TextBoost/releases/download/weights/textboost_t029.pt) | λ = 0.000083. Full TextOCR val total 0.0213 bpp (image 0.0141 + compressed small text 0.0072). PSNR 24.44. |
| [textboost_t033.pt](https://github.com/wangbxj1234/TextBoost/releases/download/weights/textboost_t033.pt) | λ = 0.000135. Full TextOCR val total 0.0259 bpp (image 0.0187 + compressed small text 0.0072). PSNR 25.39. |

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

Dependencies are PyTorch, CompressAI, Pillow, NumPy, and `pytorch-msssim`. EasyOCR is only needed for F1. `compress` pads each image to a multiple of 64 and expects a CUDA GPU. The side stream of each image is that image's own strings whose average character area is below 800, with their polygons quantized to integer pixels and gzip-compressed. An image with no such string sends nothing. English guidance uses DejaVu Sans. Chinese guidance needs a CJK font passed with `--font`.

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

`--lang ch_sim` loads Chinese and English for Wukong. LAION and Wukong use `--kind anytext` with that benchmark's `test1k.json`. EasyOCR F1 in the tables is for `textboost_stage2.pt` only, not UNITS.

`textboost_t018.pt`, `textboost_t024.pt`, `textboost_t029.pt`, and `textboost_t033.pt` are four more stage-2 checkpoints from the same recipe, at the lambdas above. λ is the coefficient in `bpp + λ * 255^2 * MSE`. The totals above are the full 3124-image TextOCR val set: arithmetic-coded image stream plus that image's compressed small text. Raw summaries are in `results/paper4/`. EasyOCR F1 was not run for these four.

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
