import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class AttentionUNet(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, timesteps=1000):
        super(AttentionUNet, self).__init__()

        # U-Net components
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.down3 = DoubleConv(128, 256)
        self.down4 = DoubleConv(256, 512)
        
        self.up1 = DoubleConv(512 + 256, 256)
        self.up2 = DoubleConv(256 + 128, 128)
        self.up3 = DoubleConv(128 + 64, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)

        # MultiHeadAttention module
        # self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=num_heads, batch_first=True)

        # Embedding layer for timestep t
        self.time_embedding = nn.Linear(1, 512)  # Linear layer for time embedding

        # Ref image encoder (similar to downsampling path)
        self.ref_down1 = DoubleConv(1, 64)
        self.ref_down2 = DoubleConv(64, 128)
        self.ref_down3 = DoubleConv(128, 256)
        self.ref_down4 = DoubleConv(256, 512)

    def forward(self, x, t, ref_img):
        # Embedding for timestep t
        t = t.view(t.size(0),-1).float()
        t_emb = self.time_embedding(t)  # Shape: (B, 512)

        # Downsampling for input image x (noisy image)
        d1 = self.down1(x)
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))

        # Downsampling for ref image
        ref_d1 = self.ref_down1(ref_img)
        ref_d2 = self.ref_down2(self.pool(ref_d1))
        ref_d3 = self.ref_down3(self.pool(ref_d2))
        ref_d4 = self.ref_down4(self.pool(ref_d3))
        
        # Flatten the spatial dimensions for attention (input + ref)
        B, C, H, W = d4.shape
        d4_flat = d4.view(B, C, H * W).permute(0, 2, 1)  # Shape: (B, H*W, C)
        ref_flat = ref_d4.view(B, C, H * W).permute(0, 2, 1)  # Shape: (B, H*W, C)
        
        # Combine features from noisy image and ref image
        combined_flat = d4_flat + ref_flat + t_emb.unsqueeze(1)  # Broadcasting time embedding
        attn_output = combined_flat.permute(0, 2, 1).view(B, C, H, W)
        
        # Apply MultiHeadAttention
        # attn_output, attn_weights = self.attention(combined_flat, combined_flat, combined_flat)
        
        # Reshape back to (B, C, H, W)
        # attn_output = attn_output.permute(0, 2, 1).view(B, C, H, W)

        # Upsampling
        u1 = F.interpolate(attn_output, scale_factor=2, mode='bilinear', align_corners=True)
        u1 = torch.cat([u1, d3], dim=1)
        u1 = self.up1(u1)

        u2 = F.interpolate(u1, scale_factor=2, mode='bilinear', align_corners=True)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up2(u2)

        u3 = F.interpolate(u2, scale_factor=2, mode='bilinear', align_corners=True)
        u3 = torch.cat([u3, d1], dim=1)
        u3 = self.up3(u3)

        return self.out_conv(u3), None  # Return the attention weights for visualization

class DiffusionModel(nn.Module):
    def __init__(self, timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.timesteps = timesteps
        self.unet = AttentionUNet(in_channels=5, out_channels=1)

    def forward_diffusion(self, x, t):
        """Add noise to the input at timestep t."""
        noise = torch.randn_like(x)
        return x * (1 - t / self.timesteps) + noise * (t / self.timesteps)

    def reverse_diffusion(self, x, t, ref_img):
        """Denoise the input using U-Net with attention and ref image."""
        return self.unet(x, t, ref_img)

    def forward(self, x, ref_img):
        # Forward diffusion
        t = torch.randint(0, self.timesteps, (x.size(0),1,1,1)).to(x.device)
        
        x_noisy = self.forward_diffusion(x, t)

        # Reverse diffusion (denoising) with attention and ref image
        x_denoised, attn_weights = self.reverse_diffusion(x_noisy, t, ref_img)

        return x_denoised, attn_weights  # Return the attention weights for visualization

class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.diffusion_model = DiffusionModel()
        self.mse_loss = nn.MSELoss()
        

    def forward(self, image, landmarks, ref_image):
        # Concatenate landmark features with image along the channel dimension
        image_with_landmarks = torch.cat([image, landmarks], dim=1)
        
        # Apply diffusion model to inpaint mouth region with ref image as additional condition
        output, attn_weights = self.diffusion_model(image_with_landmarks, ref_image)

        return output

    def loss_function(self, pred_image, gt_image):
        # Calculate MSE loss for the entire image
        mse_loss_value = self.mse_loss(pred_image, gt_image)

        # # Focus on the masked region (e.g., mouth area)
        # mask_region_pred = pred_image[:, :, 4*4:7*4, 2*4:6*4]  # Extract mouth region from prediction
        # mask_region_gt = gt_image[:, :, 4*4:7*4, 2*4:6*4]  # Extract mouth region from ground truth
        
        # # Calculate MSE loss for the mask region
        # mask_loss = self.mse_loss(mask_region_pred, mask_region_gt)

        # Apply weighting to give more importance to the mask region
        weighted_loss = mse_loss_value #+ 5 * mask_loss  # Adjust weighting factor (5 in this case)

        return weighted_loss
