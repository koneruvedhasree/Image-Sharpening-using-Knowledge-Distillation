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
def benchmark_fps(model, resolution=(1080, 1920)):
    """Benchmark model FPS at given resolution"""
    model.eval()

    # Create test input
    test_input = torch.randn(1, 3, resolution[0], resolution[1]).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(test_input)

    # Benchmark
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()

    with torch.no_grad():
        for _ in range(30):
            _ = model(test_input)

    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()

    fps = 30 / (end_time - start_time)
    print(f"FPS at {resolution[0]}x{resolution[1]}: {fps:.2f}")
    return fps
