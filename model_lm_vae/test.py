import torch
import json
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import os
import numpy as np

def device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

# Load the data from the JSON file
with open('./assets/samples/M003/samples_lm_vae_v3/tensor_data.json', 'r') as json_file:
    data = json.load(json_file)

# Convert lists back to tensors
gt_img_feature = torch.tensor(data["gt_img_feature"])
pred_img_feature = torch.tensor(data["pred_img_feature"])
gt_img_path = data["gt_img_path"]

# Print shapes to verify
print("gt_img_feature shape:", gt_img_feature.shape)
print("pred_img_feature shape:", pred_img_feature.shape)

from diffusers import AutoencoderKL

pred_features = pred_img_feature.to(device())

inv_normalize = T.Compose([
    T.Normalize(mean=[-1.0, -1.0, -1.0], std=[1.0/0.5, 1.0/0.5, 1.0/0.5]),
    T.ToPILImage()
]) 

with torch.no_grad():
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-ema").to(device())
    os.makedirs('./assets/samples/M003/samples_lm_vae_v3/images', exist_ok=True)
    for i in tqdm(range(pred_features.shape[0])):
        pf = pred_features[i].unsqueeze(0)
        # pf[:, :, 2*8:3*8, 1*8:3*8] = 0

        samples = vae.decode(pf)
        output = samples.sample[0]
        inv_image = inv_normalize(output)
        # inv_image.save(f'./assets/samples/M003/samples_lm_vae/images/pred_{i:05d}.jpg')
        
        gt_image = Image.open(gt_img_path[i])
        # gt_image.save(f'./assets/samples/M003/samples_lm_vae/images/gt_{i:05d}.jpg')

        image_size = gt_image.size[0]
        combined_image = np.ones((image_size, image_size * 2, 3), dtype=np.uint8) * 255
        inv_image_np = np.array(inv_image)
        gt_image_np = np.array(gt_image)
        combined_image[:, :image_size, :] = gt_image_np
        combined_image[:, image_size:image_size*2, :] = inv_image_np
        combined_image_pil = Image.fromarray(combined_image)
        combined_image_pil.save(f'./assets/samples/M003/samples_lm_vae_v3/images/img_{i:05d}.jpg')
        # break