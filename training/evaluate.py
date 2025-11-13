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
def evaluate_model(model, val_lr, val_hr):
    """Evaluate model performance"""
    model.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    with torch.no_grad():
        val_lr = val_lr.to(device)
        val_hr = val_hr.to(device)
        pred = model(val_lr)
        ssim_score = ssim_metric(pred, val_hr).item()

    return ssim_score

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
