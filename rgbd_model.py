import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class RGBDUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # RGB encoder
        self.rgb_encoder = models.resnet18(weights="IMAGENET1K_V1")
        self.rgb_encoder = nn.Sequential(*list(self.rgb_encoder.children())[:-2])

        # Depth encoder
        self.depth_encoder = models.resnet18(weights=None)
        self.depth_encoder.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.depth_encoder = nn.Sequential(*list(self.depth_encoder.children())[:-2])

        # Fusion
        self.fusion_conv = nn.Conv2d(512 * 2, 512, kernel_size=3, padding=1)

        # Decoder
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

    def forward(self, rgb, depth):
        rgb_feat = self.rgb_encoder(rgb)
        depth_feat = self.depth_encoder(depth)

        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        fused = self.fusion_conv(fused)

        out = self.decoder(fused)

        # Upsample to original image size (640x640)
        out = torch.nn.functional.interpolate(
            out,
            size=(rgb.shape[2], rgb.shape[3]),
            mode='bilinear',
            align_corners=False
        )

        return out