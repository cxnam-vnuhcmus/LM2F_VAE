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
            T.Normalize(mean=[-1.0, -1.0, -1.0], std=[1.0/0.5, 1.0/0.5, 1.0/0.5]),
            T.ToPILImage()
        ]) 
        
        self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema")

    def encode(self, x):
        trans_image = self.transform(x)
        trans_image = trans_image.unsqueeze(0)
        trans_image = trans_image.to(self.vae.device)
        with torch.no_grad():
            latents = vae.encode(trans_image).latent_dist.sample()
        return latents
    
    def decode(self, latents):
        with torch.no_grad():
            samples = self.vae.decode(latents)
        reconstructed = samples.sample[0]
        reconstructed = self.inv_normalize(reconstructed)
        return reconstructed
    
    def forward(self, x):
        latents = self.encode(x)    
        reconstructed = self.decode(latents)
        return reconstructed