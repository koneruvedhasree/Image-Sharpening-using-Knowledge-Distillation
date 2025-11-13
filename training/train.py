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
def train_student_model(train_pairs, val_pairs, epochs=5):
    """Train student model using knowledge distillation"""

    # Initialize models
    teacher_model = TeacherSharpeningModel().to(device)
    student_model = StudentSharpeningModel().to(device)

    # Set teacher to eval mode (no training)
    teacher_model.eval()

    # Convert pairs to tensors
    lr_imgs = torch.stack([x[0] for x in train_pairs])
    hr_imgs = torch.stack([x[1] for x in train_pairs])

    val_lr_imgs = torch.stack([x[0] for x in val_pairs])
    val_hr_imgs = torch.stack([x[1] for x in val_pairs])

    train_loader = DataLoader(TensorDataset(lr_imgs, hr_imgs), batch_size=8, shuffle=True)

    # Loss functions and optimizer
    mse_loss = nn.MSELoss()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    # Training loop
    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

        for lr, hr in progress_bar:
            lr = lr.to(device)
            hr = hr.to(device)

            # Generate teacher output (no gradients)
            with torch.no_grad():
                teacher_output = teacher_model(lr)

            # Student output
            student_output = student_model(lr)

            # Combined loss: distillation + ground truth
            loss_distill = mse_loss(student_output, teacher_output)
            loss_gt = mse_loss(student_output, hr)
            loss_ssim = 1 - ssim_metric(student_output, hr)

            # Weighted combination
            total_loss = 0.4 * loss_distill + 0.4 * loss_gt + 0.2 * loss_ssim

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            progress_bar.set_postfix({'loss': f'{total_loss.item():.4f}'})

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}] - Average Loss: {avg_loss:.4f}")

        # Validation
        if epoch % 2 == 0:
            val_ssim = evaluate_model(student_model, val_lr_imgs, val_hr_imgs)
            print(f"Validation SSIM: {val_ssim:.4f}")

    return student_model, teacher_model, val_lr_imgs, val_hr_imgs