import torch
import torch.nn as nn
from diffusers import AutoencoderKL
import torchvision.transforms as T

class VisualEncoder(nn.Module):
    def __init__(self):
        super(VisualEncoder, self).__init__()
        self.transform = T.Compose([
            T.Resize((256, 256 )),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])   

        self.inv_normalize = T.Compose([
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]) 
        
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")
        for param in self.vae.parameters():
            param.requires_grad = False

    def encode(self, x):
        trans_image = self.transform(x)
        trans_image = trans_image.unsqueeze(0)
        trans_image = trans_image.to(self.vae.device)
        with torch.no_grad():
            latents = self.vae.encode(trans_image).latent_dist.sample()
        return latents
    
    def decode(self, latents):
        self.vae.eval()
        with torch.no_grad():
            samples = self.vae.decode(latents)
            reconstructed = []
            for tensor in samples.sample:
                min_value = tensor.min()
                max_value = tensor.max()
                tensor = (tensor - min_value)/(max_value - min_value)
                reconstructed.append(tensor)
            # reconstructed = [self.inv_normalize(tensor) for tensor in samples.sample]
            reconstructed = torch.stack(reconstructed)
        return reconstructed # (32, 3, 256, 256)
    
    def forward(self, x):
        latents = self.encode(x)    
        reconstructed = self.decode(latents)
        return reconstructed
    
    def unfreeze_decoder(self):
        for param in self.vae.decoder.parameters():
            param.requires_grad = True