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
def show_results(student_model, val_lr, val_hr, index=0):
    """Display comparison results"""
    student_model.eval()

    with torch.no_grad():
        lr_img = val_lr[index].cpu().permute(1, 2, 0)
        hr_img = val_hr[index].cpu().permute(1, 2, 0)
        pred_img = student_model(val_lr[index:index+1].to(device)).squeeze(0).cpu().permute(1, 2, 0)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(lr_img.clamp(0, 1))
    axs[0].set_title('Low Resolution Input')
    axs[0].axis('off')

    axs[1].imshow(pred_img.clamp(0, 1))
    axs[1].set_title('Student Model Output')
    axs[1].axis('off')

    axs[2].imshow(hr_img.clamp(0, 1))
    axs[2].set_title('Ground Truth HR')
    axs[2].axis('off')

    plt.tight_layout()
    plt.show()

    # Calculate metrics
    psnr = calculate_psnr(pred_img, hr_img)
    print(f"PSNR: {psnr:.2f} dB")
