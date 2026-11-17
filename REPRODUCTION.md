# Reproduction

The two checkpoints behind the tables below are on the [weights release](https://github.com/wangbxj1234/TextBoost/releases/tag/weights): `textboost_stage2.pt` for TextBoost, and `elic_matched.pt` for the rate-matched ELIC. Download commands and the eval entry points are in the [README](README.md).

The tables below are one TextBoost checkpoint, trained on 8 GPUs until the loss plateaued. This is not the paper's 1000-epoch ImageNet schedule. Four more lambdas, measured on the full TextOCR val set, follow that table.

The image codec is ELIC (`N=192`, `M=320`). Stage 1 trains the whole network on TextOCR training images with

`loss = bpp + 0.0001 * 255^2 * MSE`.

Stage 2 freezes that backbone and trains the fusion block with guidance-consistent loss, `alpha=10`, train-time area threshold 150. Per-GPU batch size is 32. Stage-1 learning rate starts at `2e-4` and stage-2 at `2e-3`. Both stop when the loss stops improving. The decoder pads each image to a multiple of 64 before `compress`.

The side stream of each image is that image's own strings whose average character area is below 800, together with their polygons. Polygons are quantized to integer pixels and the payload is gzip-compressed. An image with no such string sends nothing. It is not the output of a separate OCR model. English guidance uses DejaVu Sans. Chinese guidance uses WenQuanYi Micro Hei. Empty guidance is an all-zero tensor.

OCR F1 uses EasyOCR 1.7.2 (CRAFT detector, `english_g2` / `zh_sim_g2`). UNITS pins torch 1.11 and was not run. A detection hit requires polygon IoU of at least 0.5. End-to-end also requires the same alphanumeric string, ignoring case. "Backbone" below is the ELIC RGB from this same bitstream before the fusion block. The image bpp of those two decodes is identical. TextBoost adds only the side stream.

Raw summaries are in `results/reproduction/`.

## Rate and distortion

Bitrate is the arithmetic-coded file size plus the compressed small-text side stream, divided by the original pixel count. Side bpp in the tables is total side bytes over total pixels. It is not a constant added to every image. PSNR and MS-SSIM are unweighted means.

| Set | Images | Image bpp | Side bpp | Total bpp | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| TextOCR val | 3124 | 0.0162 | 0.0072 | 0.0234 | 24.59 | 0.840 |
| LAION English | 1000 | 0.0232 | 0.0008 | 0.0241 | 24.27 | 0.898 |
| Wukong Chinese | 1000 | 0.0226 | 0.0002 | 0.0228 | 22.98 | 0.869 |
| Kodak | 24 | 0.0146 | 0 | 0.0146 | 23.04 | 0.759 |

Kodak has no text annotation, so its side stream is empty. On LAION and Wukong most annotated words are larger than the area-800 filter, so little text is sent.

Four more stage-2 checkpoints use the same recipe at other λ. Each was trained until the loss plateaued, then scored on all 3124 TextOCR val images. These five checkpoints read the same annotations, so they share one TextOCR side rate. Raw summaries are in `results/paper4/`. These four were not run on LAION, Wukong, Kodak, or EasyOCR.

| Checkpoint | λ | Image bpp | Side bpp | Total bpp | PSNR | MS-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| `textboost_t018.pt` | 0.0000055 | 0.0026 | 0.0072 | 0.0097 | 19.23 | 0.631 |
| `textboost_t024.pt` | 0.000037 | 0.0087 | 0.0072 | 0.0159 | 22.88 | 0.780 |
| `textboost_t029.pt` | 0.000083 | 0.0141 | 0.0072 | 0.0213 | 24.44 | 0.834 |
| `textboost_stage2.pt` | 0.0001 | 0.0162 | 0.0072 | 0.0234 | 24.59 | 0.840 |
| `textboost_t033.pt` | 0.000135 | 0.0187 | 0.0072 | 0.0259 | 25.39 | 0.860 |

`textboost_t033.pt` is the 0.0259 point. `textboost_t024.pt` is the 0.0159 point.

## EasyOCR F1

| Set | Original DET | Backbone DET | TextBoost DET | Original E2E | Backbone E2E | TextBoost E2E |
|---|---:|---:|---:|---:|---:|---:|
| TextOCR | 0.398 | 0.048 | 0.200 | 0.149 | 0.009 | 0.029 |
| LAION English | 0.550 | 0.431 | 0.547 | 0.338 | 0.143 | 0.237 |
| Wukong Chinese | 0.549 | 0.430 | 0.553 | 0.272 | 0.065 | 0.101 |

Small text only (average character area below 800; larger words are don't-care):

| Set | Original DET | Backbone DET | TextBoost DET | Original E2E | Backbone E2E | TextBoost E2E |
|---|---:|---:|---:|---:|---:|---:|
| TextOCR | 0.367 | 0.013 | 0.169 | 0.130 | 0.000 | 0.015 |
| LAION English | 0.263 | 0.095 | 0.279 | 0.163 | 0.007 | 0.078 |
| Wukong Chinese | 0.108 | 0.042 | 0.145 | 0.040 | 0.003 | 0.019 |

Against the ELIC decode of the same bitstream, the fusion block raises small-text detection F1 on every set. These F1 numbers are EasyOCR scores. They are not the paper's UNITS scores.

## Rate-matched ELIC

The comparison above keeps the image bitstream fixed, so the backbone is not allowed to spend the side-stream bits. A second ELIC was trained on the same images, with the loss on the decoder RGB before fusion and with the fusion weights frozen. It was initialized from the same stage-1 checkpoint as TextBoost and stopped at `lambda=0.00033`. On TextOCR val its file bpp is **0.0305**. TextBoost's total on the same set is **0.0234** (image 0.0162 plus compressed small text 0.0072).

EasyOCR F1 of that ELIC, scored on its own reconstruction:

| TextOCR | Original | Rate-matched ELIC | TextBoost | Same-bitstream ELIC |
|---|---:|---:|---:|---:|
| DET, all text | 0.398 | 0.133 | 0.200 | 0.048 |
| E2E, all text | 0.149 | 0.029 | 0.029 | 0.009 |
| DET, small text | 0.367 | 0.074 | 0.169 | 0.013 |
| E2E, small text | 0.130 | 0.005 | 0.015 | 0.000 |

TextBoost's small-text detection F1 is 0.169 against 0.074 for this ELIC. End-to-end F1 over all TextOCR words is tied at 0.029. Giving extra bits to ELIC itself recovers part of the gap from the same-bitstream decode (detection 0.048 to 0.133) and leaves most of the small-text gap. This ELIC was trained on TextOCR only. Its bitrate on LAION and Wukong was not matched to those sets.
