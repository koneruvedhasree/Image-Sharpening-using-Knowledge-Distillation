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
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
# from datasets import load_dataset # Not needed for inference
from torchmetrics import StructuralSimilarityIndexMeasure # Not needed for inference
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
# from torch.utils.data import DataLoader, TensorDataset # Not needed for inference


# Ensure the device is set up (assuming CUDA is available, otherwise use CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Define the model class again (needed for loading the state_dict)
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
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Loads an image and preprocesses it for the model."""
    try:
        img = Image.open(image_path).convert("RGB")
        # Resize to the input size the model expects
        transform = transforms.Compose([
            transforms.Resize(target_size, interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        return transform(img).unsqueeze(0).to(device) # Add batch dimension
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to save the output image
def save_output_image(tensor, output_path="sharpened_output.png"):
    """Saves a tensor image to a file."""
    try:
        # Ensure tensor is on CPU and remove batch dimension
        img = transforms.ToPILImage()(tensor.squeeze(0).cpu())
        img.save(output_path)
        print(f"Sharpened image saved as {output_path}")
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")


# Load the saved student model
try:
    # Ensure the StudentSharpeningModel class is defined (it is in the preceding code)
    loaded_model = StudentSharpeningModel().to(device)
    # Use map_location to load model on CPU if CUDA is not available
    loaded_model.load_state_dict(torch.load("student_sharpening_model.pt", map_location=device))
    loaded_model.eval()
    print("🎉 Student model loaded successfully!")

    # --- VS Code modification: Get input image path from user input ---
    image_filename = input("Please enter the path to the image you want to sharpen: ")
    # -------------------------------------------------------------------


    if not image_filename:
        print("No image path entered. Exiting.")
    else:
        print(f"Processing image: {image_filename}")

        # Load and preprocess the uploaded image
        input_image_tensor = load_and_preprocess_image(image_filename, target_size=(256, 256)) # Model expects 256x256

        if input_image_tensor is not None:
            # Perform inference
            print("Performing inference...")
            start_time = time.time()
            with torch.no_grad():
                sharpened_output_tensor = loaded_model(input_image_tensor)
            end_time = time.time()
            print(f"Inference time: {end_time - start_time:.4f} seconds")

            # Save and display the output
            save_output_image(sharpened_output_tensor, "sharpened_output.png") # Changed output filename

            # Optional: Display comparison (requires original image tensor before resizing)
            # This requires modifying load_and_preprocess_image to also return the original resized image (not just the processed tensor)
            # For simplicity, we'll just show the sharpened output here.
            output_img_display = transforms.ToPILImage()(sharpened_output_tensor.squeeze(0).cpu())
            plt.imshow(output_img_display)
            plt.title("Sharpened Output")
            plt.axis('off')
            plt.show()

except FileNotFoundError:
    print("Error: 'student_sharpening_model.pt' not found. Please ensure you have run the training section first and the model file is in the correct directory.")
except Exception as e:
    print(f"An error occurred during model loading or inference: {e}")
