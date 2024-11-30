import torch
import json
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import os
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
psnr = PeakSignalNoiseRatio(1)
ssim = StructuralSimilarityIndexMeasure(1)


def device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

root_path = './assets/samples/M003/samples_lm_vae_v8'
# Load the data from the JSON file
with open(f'{root_path}/tensor_data.json', 'r') as json_file:
    data = json.load(json_file)

# Convert lists back to tensors
gt_img_feature = torch.tensor(data["gt_img_feature"]).reshape(-1, 4, 32, 32)
pred_img_feature = torch.tensor(data["pred_img_feature"]).reshape(-1, 4, 32, 32)
# gt_img_path = data["gt_img_path"][0]
gt_img_path = [batch[0] for batch in data["gt_img_path"]]

# Print shapes to verify
print("gt_img_feature shape:", gt_img_feature.shape)
print("pred_img_feature shape:", pred_img_feature.shape)

from diffusers import AutoencoderKL

pred_features = pred_img_feature.to(device())
gt_features = gt_img_feature.to(device())

inv_normalize = T.Compose([
    # T.Normalize(mean=[-1.0, -1.0, -1.0], std=[1.0/0.5, 1.0/0.5, 1.0/0.5]),
    T.ToPILImage()
]) 

with torch.no_grad():
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device())
    os.makedirs(f'{root_path}/images', exist_ok=True)
    for i in tqdm(range(pred_features.shape[0])):
        pf = pred_features[i].unsqueeze(0)
        # mask = torch.zeros(pf.shape).to(device())
        # mask[:, :, 4*4:7*4, 2*4:6*4] = 1
        noise = torch.randn_like(pf).to(device()) * 0.7 # * mask 
        gt = gt_features[i].unsqueeze(0) + noise
        pf[:, :, 4*4:7*4, 2*4:6*4] = gt[:, :, 4*4:7*4, 2*4:6*4]
        

        samples = vae.decode(pf)
        output = samples.sample[0]
        min_value = output.min()
        max_value = output.max()
        pred_image = (output - min_value)/(max_value - min_value)         
        to_pil = T.ToPILImage()
        inv_image = to_pil(pred_image)
        inv_image.save(f'{root_path}/images/pred_{i:05d}.jpg')
        
        gt = gt_features[i].unsqueeze(0)
        samples = vae.decode(gt)
        output = samples.sample[0]
        min_value = output.min()
        max_value = output.max()
        gt_image = (output - min_value)/(max_value - min_value)  
        to_pil = T.ToPILImage()
        inv_image = to_pil(gt_image)
        inv_image.save(f'{root_path}/images/gt_{i:05d}.jpg')
        
        print(pred_image.shape)
        psnr_mark = psnr(pred_image.cpu(), gt_image.cpu())
        ssim_mark = ssim(pred_image.cpu().unsqueeze(0), gt_image.cpu().unsqueeze(0))
        print(f"{i} - PSRN: {psnr_mark}; SSIM: {ssim_mark}")

        # image_size = gt_image.size[0]
        # combined_image = np.ones((image_size, image_size * 2, 3), dtype=np.uint8) * 255
        # inv_image_np = np.array(inv_image)
        # gt_image_np = np.array(gt_image)
        # combined_image[:, :image_size, :] = gt_image_np
        # combined_image[:, image_size:image_size*2, :] = inv_image_np
        # combined_image_pil = Image.fromarray(combined_image)
        # combined_image_pil.save(f'{root_path}/images/img_{i:05d}.jpg')
        # break
        