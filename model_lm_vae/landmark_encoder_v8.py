import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        self.up1 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up3 = DoubleConv(128 + 64, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Downsampling
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        
        # Upsampling
        u1 = F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.up1(u1)

        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=True)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2(u2)

        u3 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=True)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.up3(u3)

        return self.out_conv(u3)

class DiffusionModel(nn.Module):
    def __init__(self, timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.timesteps = timesteps
        self.unet = UNet(in_channels=8, out_channels=4)

    def forward_diffusion(self, x, t):
        """Add noise to the input at timestep t."""
        noise = torch.randn_like(x)
        return x * (1 - t / self.timesteps) + noise * (t / self.timesteps)

    def reverse_diffusion(self, x, t):
        """Denoise the input using the U-Net."""
        return self.unet(x)

    def forward(self, x):
        # Forward diffusion
        t = torch.randint(0, self.timesteps, (x.size(0),)).to(x.device)
        x_noisy = self.forward_diffusion(x, t)

        # Reverse diffusion (denoising)
        x_denoised = self.reverse_diffusion(x_noisy, t)

        return x_denoised

class LandmarkEncoder(nn.Module):
    def __init__(self, input_size=478, output_size=4):
        super(LandmarkEncoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size * 2, 1024),  # Increase the feature dimension
            nn.ReLU(inplace=True),
            nn.Linear(1024, 4096),  # Targeting 4 * 32 * 32 (i.e., 4096)
            nn.ReLU(inplace=True)
        )

    def forward(self, landmarks):
        # Flatten landmarks input to shape (B, 478 * 2)
        x = self.fc(landmarks.view(landmarks.size(0), -1))
        
        # Reshape output to (B, 4, 32, 32)
        return x.view(landmarks.size(0), 4, 32, 32)

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.diffusion_model = DiffusionModel()
        self.landmark_encoder = LandmarkEncoder()
        self.mse_loss = nn.MSELoss()

    def forward(self, image, landmarks):
        # Encode landmarks to a spatial tensor
        landmark_features = self.landmark_encoder(landmarks)
        
        # Concatenate landmark features with image along the channel dimension
        image_with_landmarks = torch.cat([image, landmark_features], dim=1)
        
        # Apply diffusion model to inpaint mouth region
        output = self.diffusion_model(image_with_landmarks)
        return output

    def loss_function(self, pred_image, gt_image):
        # Mean Squared Error (MSE) Loss
        mse_loss_value = self.mse_loss(pred_image, gt_image)
        
        # Combine losses (with a balancing factor)
        total_loss = mse_loss_value 
        return total_loss