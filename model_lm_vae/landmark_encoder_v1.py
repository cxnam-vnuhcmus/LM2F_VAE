import torch
import torch.nn as nn
import torch.nn.functional as F



class LandmarkNet(nn.Module):
    def __init__(self):
        super(LandmarkNet, self).__init__()
        # Define the CNN layers
        self.conv_layers = nn.Sequential(
            # Block 1: (B, 3, 256, 256) -> (B, 64, 128, 128)
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
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

class LandmarkToImageFeatureEncoder(nn.Module):
    def __init__(self):
        super(LandmarkToImageFeatureEncoder, self).__init__()
        self.feature_extraction = LandmarkNet()
        # self.unet = UNet()
        self.attention = nn.MultiheadAttention(embed_dim=32*32, num_heads=8, dropout=0.1, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(32*32, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 32*32)
        )        

    def forward(self, landmark_sketch, img_feature, raw_img=None):
        lm_features = self.feature_extraction(landmark_sketch) # (B, 4, 32, 32)
        B, N, W, H = img_feature.shape
        lm_features = lm_features.reshape(B*N, W*H)
        img_feature = img_feature.reshape(B*N, W*H)
        
        output, _ = self.attention(img_feature, lm_features, lm_features)
        output = self.fc(output)
        output = output.reshape(B, N, W, H)
        
        return output