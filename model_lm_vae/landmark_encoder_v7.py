import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelWiseAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(ChannelWiseAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_channels)
        )
        
    def forward(self, x):
        # X.shape: (B, C, H, W)
        B, C, H, W = x.size()
        
        # Reshape for MLP
        x_flat = x.view(B, C, -1)  # Shape: (B, C, H*W)
        attention_scores = self.mlp(x_flat)  # Shape: (B, C, H*W)
        
        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # Shape: (B, C, H*W)
        
        # Apply attention weights to the input
        x_attended = (x_flat * attention_weights).view(B, C, H, W)  # Shape: (B, C, H, W)
        
        return x_attended, attention_weights


class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.channel_attention = ChannelWiseAttention(in_channels=32*32, hidden_dim=512)
        
        # Additional convolutional layers after attention
        self.conv1 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # (B, 16, 32, 32) -> (B, 32, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (B, 32, 32, 32) -> (B, 64, 32, 32)
        self.conv3 = nn.Conv2d(64, 4, kernel_size=3, padding=1)   # (B, 64, 32, 32) -> (B, 4, 32, 32)
        
        self.relu = nn.ReLU()

    def forward(self, pred_mask_img, pred_lm, ref_img, ref_lm):
        # Concatenate the inputs
        x = torch.cat((pred_mask_img, pred_lm, ref_img, ref_lm), dim=1)  # Shape: (B, 16, 32, 32)
        
        # Process through channel-wise attention
        x_attended, attention_map = self.channel_attention(x)
        
        # Apply further layers to generate pred_img
        x_out = self.conv1(x_attended)  # Shape: (B, 32, 32, 32)
        x_out = self.relu(x_out)

        x_out = self.conv2(x_out)  # Shape: (B, 64, 32, 32)
        x_out = self.relu(x_out)

        x_out = self.conv3(x_out)  # Shape: (B, 4, 32, 32)
        
        return x_out, attention_map

    def loss_function(self, pred_img, gt_img):
        # Calculate mean and std of the ground truth
        mean_gt = torch.mean(gt_img, dim=(0, 2, 3), keepdim=True)  # Shape: (1, C, 1, 1)
        std_gt = torch.std(gt_img, dim=(0, 2, 3), keepdim=True)   # Shape: (1, C, 1, 1)

        # Normalize the output
        normalized_output = pred_img * std_gt + mean_gt
        
        # Loss based on the normalized output
        return F.mse_loss(normalized_output, gt_img)

