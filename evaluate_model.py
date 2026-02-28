import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rgbd_dataset import RGBDDataset
from rgb_model import RGBUNet
from rgbd_model import RGBDUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = RGBDDataset("processed/preprocessed/manifest.csv")
loader = DataLoader(dataset, batch_size=4, shuffle=False)

def compute_metrics(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * masks).sum()
    union = (preds + masks - preds*masks).sum()

    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2 * intersection + 1e-6) / (preds.sum() + masks.sum() + 1e-6)

    return iou.item(), dice.item()

def evaluate(model, model_path, is_rgbd=True):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    total_iou = 0
    total_dice = 0

    with torch.no_grad():
        for rgb, depth, mask in loader:
            rgb = rgb.to(device)
            depth = depth.to(device)
            mask = mask.to(device)

            if is_rgbd:
                preds = model(rgb, depth)
            else:
                preds = model(rgb)

            iou, dice = compute_metrics(preds, mask)
            total_iou += iou
            total_dice += dice

    print("IoU:", total_iou / len(loader))
    print("Dice:", total_dice / len(loader))

# Evaluate RGB-D
print("Evaluating RGB-D Model")
evaluate(RGBDUNet(), "best_rgbd_model.pth", is_rgbd=True)

# Evaluate RGB
print("\nEvaluating RGB Model")
evaluate(RGBUNet(), "best_rgb_model.pth", is_rgbd=False)