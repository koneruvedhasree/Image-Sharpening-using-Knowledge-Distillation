# Image Sharpening using Knowledge Distillation

This project uses a teacher-student architecture to perform image sharpening based on a super-resolution task. It loads the DIV2K dataset (or synthetic fallback), trains a deep ResNet-based teacher model and a lightweight student CNN using knowledge distillation.

## Project Structure

```
.
├── main.py                         # Entry point to train, evaluate and save the student model
├── models/
│   ├── student_model.py           # Lightweight CNN model
│   └── teacher_model.py           # ResNet34-based encoder-decoder teacher model
├── data/
│   ├── dataset_loader.py          # Loads DIV2K or synthetic data
│   
├── training/
│   ├── train.py                   # Training logic using knowledge distillation
│   └── evaluate.py                # Evaluation metrics: SSIM, FPS
├── utils/
│   ├── image_utils.py             # Image display, saving, visualization
│   └── metrics.py                 # PSNR metric
├── inference/
│   └── sharpen_image.py           # Run inference on a new image using trained model
├── requirements.txt               # Dependencies
└── README.md                      # This file
student_sharpening_model.pt        # trianded model 
```

## Getting Started

1. Clone the repo or unzip the files.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run training:
```bash
python main.py
```
4. Run inference:
```bash
python inference/sharpen_image.py
```

## Model Output

- Trained student model is saved as `student_sharpening_model.pt`.
- Inference script will prompt for an image and output `sharpened_output.png`.

## Dependencies

See `requirements.txt`.

