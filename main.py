import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from datasets import load_dataset
from torchmetrics import StructuralSimilarityIndexMeasure
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, TensorDataset
def main():
    """Main execution function"""
    # Load dataset
    train_pairs, val_pairs = load_div2k_dataset(num_samples=200)

    if len(train_pairs) == 0 or len(val_pairs) == 0:
        print("Error: No data loaded. Please check dataset loading.")
        return

    # Train model
    print("\nStarting knowledge distillation training...")
    student_model, teacher_model, val_lr, val_hr = train_student_model(train_pairs, val_pairs, epochs=5)

    # Final evaluation
    print("\nFinal Evaluation:")
    final_ssim = evaluate_model(student_model, val_lr, val_hr)
    print(f"Final SSIM Score: {final_ssim:.4f} ({final_ssim * 100:.2f}%)")

    # Show results
    print("\nSample Results:")
    for i in range(min(3, len(val_pairs))):
        print(f"\nSample {i+1}:")
        show_results(student_model, val_lr, val_hr, index=i)
        print("MOS Rating (1-5): ______")  # Manual evaluation

    # Benchmark FPS
    print("\nFPS Benchmarking:")
    benchmark_fps(student_model, resolution=(1080, 1920))

    # Save model
    torch.save(student_model.state_dict(), "student_sharpening_model.pt")
    print("\n🎉 Student model saved as 'student_sharpening_model.pt'")

    return student_model

if __name__ == "__main__":
    main()