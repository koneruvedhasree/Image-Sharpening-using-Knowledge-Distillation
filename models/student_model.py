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
class StudentSharpeningModel(nn.Module):
    """Lightweight student model"""
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Downsample to 128x128
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            # Upsample from 128x128 to 256x256
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Final output
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Debug: print shapes
        # print(f"Student input: {x.shape}")

        x = self.encoder(x)
        # print(f"Student after encoder: {x.shape}")

        x = self.decoder(x)
        # print(f"Student output: {x.shape}")

        return x
