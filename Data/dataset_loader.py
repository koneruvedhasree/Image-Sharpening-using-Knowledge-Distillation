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

def load_div2k_dataset(num_samples=200):
    """Load and preprocess DIV2K dataset"""
    print("Loading DIV2K dataset...")

    try:
        # Try regular loading first
        ds = load_dataset("eugenesiow/Div2k", "bicubic_x2")
        dataset_source = ds["train"]
        use_streaming = False
        print("Using regular dataset loading")
    except Exception as e:
        print(f"Regular loading failed: {e}")
        try:
            # Fallback to streaming
            ds = load_dataset("eugenesiow/Div2k", "bicubic_x2", streaming=True)
            dataset_source = ds["train"]
            use_streaming = True
            print("Using streaming dataset loading")
        except Exception as e2:
            print(f"Streaming also failed: {e2}")
            print("Creating synthetic dataset for demonstration...")
            return create_synthetic_dataset(num_samples)

    # Define transforms
    hr_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    lr_tf = transforms.Compose([
        transforms.Resize((128, 128), interpolation=Image.BICUBIC),
        transforms.Resize((256, 256), interpolation=Image.BICUBIC),
        transforms.ToTensor()
    ])

    # Build dataset sample
    train_pairs = []
    val_pairs = []

    try:
        for i, sample in enumerate(dataset_source):
            if i >= num_samples:
                break

            try:
                # Handle different possible formats
                if isinstance(sample["hr"], str):
                    # If it's a path/URL, skip for now
                    continue
                elif hasattr(sample["hr"], 'convert'):
                    # PIL Image
                    hr = hr_tf(sample["hr"].convert("RGB"))
                    lr = lr_tf(sample["lr"].convert("RGB"))
                else:
                    # Skip unknown format
                    continue

                # Split into train/val (80/20)
                if i < int(0.8 * num_samples):
                    train_pairs.append((lr, hr))
                else:
                    val_pairs.append((lr, hr))

            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue

    except Exception as e:
        print(f"Dataset iteration failed: {e}")
        print("Creating synthetic dataset for demonstration...")
        return create_synthetic_dataset(num_samples)

    if len(train_pairs) == 0:
        print("No valid samples loaded, creating synthetic dataset...")
        return create_synthetic_dataset(num_samples)

    print(f"Loaded {len(train_pairs)} training pairs and {len(val_pairs)} validation pairs")
    return train_pairs, val_pairs

def create_synthetic_dataset(num_samples=200):
    """Create synthetic dataset for demonstration when real dataset fails"""
    print("Creating synthetic image dataset...")

    train_pairs = []
    val_pairs = []

    # Create synthetic images with patterns
    for i in range(num_samples):
        # Generate a high-resolution synthetic image with patterns
        hr_img = torch.zeros(3, 256, 256)

        # Add some patterns (stripes, gradients, noise)
        x = torch.linspace(0, 1, 256)
        y = torch.linspace(0, 1, 256)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        # Different patterns for different channels
        hr_img[0] = torch.sin(10 * X) * torch.cos(10 * Y) * 0.5 + 0.5
        hr_img[1] = (X + Y) / 2
        hr_img[2] = torch.sin(5 * (X + Y)) * 0.5 + 0.5

        # Add some noise for realism
        hr_img += torch.randn_like(hr_img) * 0.05
        hr_img = torch.clamp(hr_img, 0, 1)

        # Create low-resolution version by downsampling and upsampling
        lr_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128), interpolation=Image.BICUBIC),
            transforms.Resize((256, 256), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])

        lr_img = lr_transform(hr_img)

        # Split into train/val (80/20)
        if i < int(0.8 * num_samples):
            train_pairs.append((lr_img, hr_img))
        else:
            val_pairs.append((lr_img, hr_img))

    print(f"Created {len(train_pairs)} synthetic training pairs and {len(val_pairs)} validation pairs")
    return train_pairs, val_pairs
