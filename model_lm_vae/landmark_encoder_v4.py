import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallUNet(nn.Module):
    def __init__(self, in_channels=7, out_channels=4):
        super(SmallUNet, self).__init__()
        
        # Encoder path (Downsampling)
        self.enc_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Max pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Decoder path (Upsampling)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Decoder convolutions
        self.dec_conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1)  # skip connection
        self.dec_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Final output layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = F.relu(self.enc_conv1(x))   # (B, 64, 32, 32)
        enc2 = F.relu(self.enc_conv2(self.pool(enc1)))  # (B, 128, 16, 16)
        enc3 = F.relu(self.enc_conv3(self.pool(enc2)))  # (B, 256, 8, 8)
        
        # Decoder path
        up1 = F.relu(self.upconv1(enc3))  # (B, 128, 16, 16)
        up1 = torch.cat((up1, enc2), dim=1)  # Skip connection (B, 128 + 128, 16, 16)
        dec1 = F.relu(self.dec_conv1(up1))  # (B, 128, 16, 16)
        
        up2 = F.relu(self.upconv2(dec1))  # (B, 64, 32, 32)
        up2 = torch.cat((up2, enc1), dim=1)  # Skip connection (B, 64 + 64, 32, 32)
        dec2 = F.relu(self.dec_conv2(up2))  # (B, 64, 32, 32)
        
        # Final output
        output = self.final_conv(dec2)  # (B, 4, 32, 32)
        return output
    
class LandmarkToImageFeatureEncoder(nn.Module):
    def __init__(self):
        super(LandmarkToImageFeatureEncoder, self).__init__()
        self.unet = SmallUNet(in_channels=7, out_channels=4)

    def forward(self, landmark_sketch, img_feature, raw_img=None):
        # Extract features
        # downsampled_landmarks = F.interpolate(landmark_sketch, size=(32, 32), mode='bilinear', align_corners=False)
        
        # Concatenate features with img_feature
        concatenated = torch.cat([img_feature, landmark_sketch], dim=1)
        
        # Inpaint with U-Net
        output = self.unet(concatenated)
        
        return output