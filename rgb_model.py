import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class RGBUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Single RGB encoder
        self.encoder = models.resnet18(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])

        # Decoder (same as RGB-D version)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, rgb):
        features = self.encoder(rgb)
        out = self.decoder(features)

        out = F.interpolate(
            out,
            size=(rgb.shape[2], rgb.shape[3]),
            mode='bilinear',
            align_corners=False
        )

        return out