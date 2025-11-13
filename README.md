# Image Sharpening using Knowledge Distillation (KD)

This repository explores an efficient approach to image sharpening and super-resolution by training a lightweight Student Model through Knowledge Distillation (KD) from a powerful pretrained Teacher Model. The objective is to achieve high-quality sharpening with significantly reduced computational cost, making the solution suitable for real-time and edge deployment.

## Overview

This project applies Knowledge Distillation to transfer the sharpening capabilities of a deep ResNet-50 Teacher Model to a compact CNN Student Model. The teacher learns complex high-level features, and the student learns to approximate these while maintaining fast inference speed.

### Core Components

| Component | Description |
|----------|-------------|
| Task | Image Sharpening / Super-Resolution |
| Teacher Model | ResNet-50 based encoder-decoder |
| Student Model | Lightweight Convolutional Neural Network (CNN) |
| Dataset | DIV2K (or synthetic fallback) |
| Training | KD Loss + MSE + SSIM + PSNR |

## Results Summary

The lightweight student model achieved strong performance after Knowledge Distillation:

| Metric | Value | Meaning |
|--------|--------|---------|
| SSIM | ~86.24 | High structural similarity with ground truth |
| PSNR | ~29.2 | Good reconstruction quality |
| FPS | ~9.88 | Efficient real-time inference on target hardware |

## Project Structure

```
.
├── main.py
├── models/
│   ├── student_model.py
│   └── teacher_model.py
├── data/
│   └── dataset_loader.py
├── training/
│   ├── train.py
│   └── evaluate.py
├── utils/
│   ├── image_utils.py
│   └── metrics.py
├── inference/
│   └── sharpen_image.py
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Installation

```
git clone https://github.com/yourusername/image-sharpening-kd.git
cd image-sharpening-kd
pip install -r requirements.txt
```

### 2. Training

```
python main.py
```

### 3. Inference

```
python inference/sharpen_image.py
```

## References

- ResNet-50 Architecture — https://arxiv.org/abs/2108.10257
- DIV2K Dataset — https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/
- Knowledge Distillation — Hinton et al.

