import torch
import torch.nn as nn
from diffusers import AutoencoderKL
import torchvision.transforms as T
from torch.cuda.amp import autocast


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
        # self.vae.eval()
        # with torch.no_grad():
        with autocast():
            latents = latents.float()
            B = latents.size(0)
            latents = latents.reshape(-1, 4, 32, 32)
            samples = self.vae.decode(latents)
            reconstructed = [self.inv_normalize(tensor) for tensor in samples.sample]
            reconstructed = torch.stack(reconstructed)
            reconstructed = torch.clamp(reconstructed, min=0., max=1.)
            reconstructed = torch.nan_to_num(reconstructed, nan=0.0, posinf=1.0, neginf=1.0)
            reconstructed = reconstructed.reshape(B, -1, 3, 256, 256)
        return reconstructed # (32, 3, 256, 256)
    
    def forward(self, x):
        latents = self.encode(x)    
        reconstructed = self.decode(latents)
        return reconstructed
    
    def unfreeze_decoder(self):
        for param in self.vae.decoder.parameters():
            param.requires_grad = True