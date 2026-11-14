# Reproduction

One TextBoost checkpoint, trained on 8 GPUs until the loss plateaued. This is not the paper's 1000-epoch ImageNet schedule, and it is one operating point rather than the paper's four bitrate points.

The image codec is ELIC (`N=192`, `M=320`). Stage 1 trains the whole network on TextOCR training images with

`loss = bpp + 0.0001 * 255^2 * MSE`.

Stage 2 freezes that backbone and trains the fusion block with guidance-consistent loss, `alpha=10`, train-time area threshold 150. Per-GPU batch size is 32. Stage-1 learning rate starts at `2e-4` and stage-2 at `2e-3`. Both stop when the loss stops improving. The decoder pads each image to a multiple of 64 before `compress`.

The side stream is gzip of the dataset's own boxes and transcripts whose average character area is below 800. It is not the output of a separate OCR model. English guidance uses DejaVu Sans. Chinese guidance uses WenQuanYi Micro Hei. Empty guidance is an all-zero tensor.

OCR F1 uses EasyOCR 1.7.2 (CRAFT detector, `english_g2` / `zh_sim_g2`). UNITS pins torch 1.11 and was not run. A detection hit requires polygon IoU of at least 0.5. End-to-end also requires the same alphanumeric string, ignoring case. "Backbone" below is the ELIC RGB from this same bitstream before the fusion block. The image bpp of those two decodes is identical. TextBoost adds only the side stream.

Raw summaries are in `results/reproduction/`.

## Rate and distortion

Bitrate is the arithmetic-coded file size plus the gzip side stream, divided by the original pixel count. PSNR and MS-SSIM are unweighted means.

| Set | Images | Image bpp | Side bpp | Total bpp | PSNR | MS-SSIM | Small-text PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| TextOCR val | 3124 | 0.0162 | 0.0142 | 0.0304 | 24.59 | 0.840 | 18.17 |
| LAION English | 1000 | 0.0232 | 0.0020 | 0.0252 | 24.27 | 0.898 | 16.56 |
| Wukong Chinese | 1000 | 0.0226 | 0.0011 | 0.0237 | 22.98 | 0.869 | 16.51 |
| Kodak | 24 | 0.0146 | 0.0004 | 0.0151 | 23.04 | 0.759 | — |

Kodak has no text annotation, so the side stream is an empty gzip. On TextOCR the side stream is about as large as the image stream. On LAION and Wukong most annotated words are larger than the area-800 filter, so the side stream is a few thousandths of a bit per pixel.

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

Against the ELIC decode of the same bitstream, the fusion block raises small-text detection F1 on every set. These F1 numbers are EasyOCR scores. They are not the paper's UNITS scores, and the backbone here was not retrained to spend the side-stream bits on the image. A rate-matched ELIC, trained separately so that its image bpp matches TextBoost's total bpp, is not in this snapshot.
