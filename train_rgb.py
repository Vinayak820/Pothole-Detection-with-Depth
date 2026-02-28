import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from rgbd_dataset import RGBDDataset
from rgb_model import RGBUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

dataset = RGBDDataset("processed/preprocessed/manifest.csv")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = RGBUNet().to(device)

bce = nn.BCEWithLogitsLoss()

def dice_loss(preds, targets):
    preds = torch.sigmoid(preds)
    intersection = (preds * targets).sum()
    return 1 - (2. * intersection + 1e-6) / (preds.sum() + targets.sum() + 1e-6)

def compute_iou(preds, masks, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()

    intersection = (preds * masks).sum(dim=(1,2,3))
    union = (preds + masks - preds*masks).sum(dim=(1,2,3))

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 20
best_iou = 0

for epoch in range(epochs):

    # Training
    model.train()
    train_loss = 0

    for rgb, depth, mask in train_loader:
        rgb = rgb.to(device)
        mask = mask.to(device)

        preds = model(rgb)

        loss = bce(preds, mask) + dice_loss(preds, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    val_iou = 0

    with torch.no_grad():
        for rgb, depth, mask in val_loader:
            rgb = rgb.to(device)
            mask = mask.to(device)

            preds = model(rgb)

            loss = bce(preds, mask) + dice_loss(preds, mask)
            val_loss += loss.item()
            val_iou += compute_iou(preds, mask)

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)

    print(f"Epoch [{epoch+1}/{epochs}]")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val IoU:    {val_iou:.4f}")
    print("-" * 40)

    if val_iou > best_iou:
        best_iou = val_iou
        torch.save(model.state_dict(), "best_rgb_model.pth")
        print("🔥 Best RGB model saved!\n")

print("Training complete.")