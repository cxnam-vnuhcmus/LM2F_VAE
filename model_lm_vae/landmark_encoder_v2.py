import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkNet(nn.Module):
    def __init__(self):
        super(LandmarkNet, self).__init__()
        # Define the CNN layers
        self.conv_layers = nn.Sequential(
            # Block 1: (B, 3, 256, 256) -> (B, 64, 128, 128)
            nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: (B, 64, 128, 128) -> (B, 128, 64, 64)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: (B, 128, 64, 64) -> (B, 256, 32, 32)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 4: (B, 256, 32, 32) -> (B, 512, 32, 32)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Final Conv Layer to get (B, 4, 32, 32)
        self.final_conv = nn.Conv2d(in_channels=512, out_channels=4, kernel_size=1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_conv(x)
        return x
