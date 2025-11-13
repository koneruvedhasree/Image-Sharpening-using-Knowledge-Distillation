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
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class TeacherSharpeningModel(nn.Module):
    """Teacher model based on ResNet34 encoder-decoder"""
    def __init__(self):
        super().__init__()
        base = models.resnet34(weights='DEFAULT')
        # Remove the last two layers (avgpool and fc) to keep spatial dimensions
        self.encoder = nn.Sequential(*list(base.children())[:-2])  # 512 channels, 8x8 for 256x256 input

        # Add adaptive pooling to ensure consistent size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        self.decoder = nn.Sequential(
            # From 8x8 to 16x16
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # From 16x16 to 32x32
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # From 32x32 to 64x64
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # From 64x64 to 128x128
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # From 128x128 to 256x256
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Debug: print input shape
        # print(f"Input shape: {x.shape}")

        x = self.encoder(x)
        # print(f"After encoder: {x.shape}")

        x = self.adaptive_pool(x)
        # print(f"After adaptive pool: {x.shape}")

        x = self.decoder(x)
        # print(f"Output shape: {x.shape}")

        return x