import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from rgbd_dataset import RGBDDataset
from rgb_model import RGBUNet
from rgbd_model import RGBDUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Load Dataset
# -----------------------------
dataset = RGBDDataset("processed/preprocessed/manifest.csv")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

# -----------------------------
# Load Models
# -----------------------------
rgb_model = RGBUNet().to(device)
rgb_model.load_state_dict(torch.load("best_rgb_model.pth", map_location=device))
rgb_model.eval()

rgbd_model = RGBDUNet().to(device)
rgbd_model.load_state_dict(torch.load("best_rgbd_model.pth", map_location=device))
rgbd_model.eval()

# -----------------------------
# Visualization Function
# -----------------------------
def overlay_mask(image, mask, color=(0,255,0), alpha=0.4):
    mask = mask.squeeze()
    mask = mask.cpu().numpy()

    image = image.permute(1,2,0).cpu().numpy()
    image = (image * 255).astype(np.uint8)

    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0.5] = color

    blended = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    return blended

# -----------------------------
# Show Samples
# -----------------------------
num_samples = 5

with torch.no_grad():
    for i, (rgb, depth, mask) in enumerate(loader):

        if i >= num_samples:
            break

        rgb = rgb.to(device)
        depth = depth.to(device)
        mask = mask.to(device)

        # Predictions
        rgb_pred = torch.sigmoid(rgb_model(rgb))
        rgbd_pred = torch.sigmoid(rgbd_model(rgb, depth))

        # Threshold
        rgb_pred = (rgb_pred > 0.5).float()
        rgbd_pred = (rgbd_pred > 0.5).float()

        # Overlay
        original = (rgb[0] * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)
        gt_overlay = overlay_mask(rgb[0], mask[0], color=(0,0,255))
        rgb_overlay = overlay_mask(rgb[0], rgb_pred[0], color=(0,255,0))
        rgbd_overlay = overlay_mask(rgb[0], rgbd_pred[0], color=(255,0,0))

        # Plot
        plt.figure(figsize=(12,8))

        plt.subplot(2,2,1)
        plt.title("Original RGB")
        plt.imshow(original)
        plt.axis("off")

        plt.subplot(2,2,2)
        plt.title("Ground Truth")
        plt.imshow(gt_overlay)
        plt.axis("off")

        plt.subplot(2,2,3)
        plt.title("RGB Prediction")
        plt.imshow(rgb_overlay)
        plt.axis("off")

        plt.subplot(2,2,4)
        plt.title("RGB-D Prediction")
        plt.imshow(rgbd_overlay)
        plt.axis("off")

        plt.tight_layout()
        plt.show()