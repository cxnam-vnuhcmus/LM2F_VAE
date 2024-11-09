import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkNet(nn.Module):
    def __init__(self):
        super(LandmarkNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(5, 64)  # Input channels = 5 (from concatenated features)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        self.pool = nn.MaxPool2d(2)
        
        self.middle = self.conv_block(512, 1024)
        
        self.upconv4 = self.upconv(1024, 512)
        self.decoder4 = self.conv_block(1024, 512)  # 1024 from concatenation of upconv4 and encoder4
        
        self.upconv3 = self.upconv(512, 256)
        self.decoder3 = self.conv_block(512, 256)  # 512 from concatenation of upconv3 and encoder3
        
        self.upconv2 = self.upconv(256, 128)
        self.decoder2 = self.conv_block(256, 128)  # 256 from concatenation of upconv2 and encoder2
        
        self.upconv1 = self.upconv(128, 64)
        self.decoder1 = self.conv_block(128, 64)  # 128 from concatenation of upconv1 and encoder1
        
        self.out_conv = nn.Conv2d(64, 4, kernel_size=1)  # Output channels = 4

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    
    def upconv(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        e2 = self.pool(e1)
        e2 = self.encoder2(e2)
        e3 = self.pool(e2)
        e3 = self.encoder3(e3)
        e4 = self.pool(e3)
        e4 = self.encoder4(e4)
        
        # Middle
        m = self.pool(e4)
        m = self.middle(m)
        
        # Decoder
        d4 = self.upconv4(m)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)
        
        out = self.out_conv(d1)
        out = torch.sigmoid(out)
        
        return out
    
class LandmarkToImageFeatureEncoder(nn.Module):
    def __init__(self):
        super(LandmarkToImageFeatureEncoder, self).__init__()
        self.feature_extraction = LandmarkNet()
        self.unet = UNet()
        self.fc_mu = nn.Linear(4 * 32 * 32, 512)
        self.fc_logvar = nn.Linear(4 * 32 * 32, 512)
        self.decoder_fc = nn.Sequential(
            nn.Linear(512, 4 * 32 * 32),
            # nn.Sigmoid()
        )
        
    def reparameterize(self, mu, logvar):
        # Reparameterization trick to sample from the latent space
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, landmark_sketch, img_feature, raw_img=None):
        # Extract features
        features = self.feature_extraction(landmark_sketch)
        
        # Concatenate features with img_feature
        concatenated = torch.cat([img_feature, features], dim=1)
        
        concatenated = self.unet(concatenated)
        
        concatenated_flat = concatenated.view(concatenated.size(0), -1)
        
        mu = self.fc_mu(concatenated_flat)
        logvar = self.fc_logvar(concatenated_flat)
        
        z = self.reparameterize(mu, logvar)
        output = self.decoder_fc(z)
        output = output.reshape(output.shape[0], 4, 32, 32)
        
        
        return output