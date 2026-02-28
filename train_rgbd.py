import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rgbd_dataset import RGBDDataset
from rgbd_model import RGBDUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = RGBDDataset("processed/preprocessed/manifest.csv")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = RGBDUNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 20

print("Starting training...")

for epoch in range(epochs):
    print(f"Epoch {epoch+1} starting...")
    model.train()
    total_loss = 0

    for rgb, depth, mask in loader:
        rgb, depth, mask = rgb.to(device), depth.to(device), mask.to(device)

        preds = model(rgb, depth)
        loss = criterion(preds, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")