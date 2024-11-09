import torch
from torch import nn
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import os
from typing import Union
import json
import torchvision.transforms as transforms

from .landmark_encoder_v9 import FusionModel
# from .visual_encoder import VisualEncoder
# from .discriminator import Discriminator
# from .loss_v1 import AdversarialLoss

#DS_V4
class Model(nn.Module):

    def __init__(self,
                 pretrained: Union[bool, str] = True,
                 infer_samples: bool = False
                 ):
        super().__init__()
        
        self.pretrained = pretrained
        self.infer_samples = infer_samples
        
        self.model = FusionModel(timesteps=32)
        
        # self.vsencoder = VisualEncoder()
        # for param in self.vsencoder.parameters():
        #     param.requires_grad = False
        
        # self.adversarial_loss = AdversarialLoss()
        
        # self.discriminator = Discriminator(in_channels=3, use_sigmoid=True)
        
        
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def forward(self,
                landmark,
                gt_img_feature,
                gt_img,
                training=True
                ):
        landmark = landmark.to(self.device)
        gt_img_feature = gt_img_feature.to(self.device)
        # gt_img = gt_img.to(self.device)
        
        mask_gt_img_feature = gt_img_feature.clone()        
        noise = torch.randn(gt_img_feature.shape).to(self.device)
        mask_gt_img_feature[:, :, 4*4:7*4, 2*4:6*4] = noise[:, :, 4*4:7*4, 2*4:6*4]
        
        ref_img_feature = gt_img_feature.clone()
        ref_img_feature = ref_img_feature[torch.randperm(ref_img_feature.size(0))]
        
        pred_noise, noise =  self.model(mask_gt_img_feature, landmark, ref_img_feature)        
        
        loss = self.model.loss_function(pred_noise, noise)
        
        pred_img_feature = None
        if not training:
            pred_img_feature = self.model.sample(mask_gt_img_feature, landmark, ref_img_feature)
        
        return (pred_img_feature, None), loss
        
        
    def training_step_imp(self, batch, device) -> torch.Tensor:
        landmark, gt_img_feature, gt_mask_img_feature, gt_img, _ = batch
        _, loss = self(
            landmark = landmark,
            gt_img_feature = gt_img_feature,
            gt_img = gt_img,
            training=True
        )
        
        return loss

    def eval_step_imp(self, batch, device):
        with torch.no_grad():
            landmark, gt_img_feature, gt_mask_img_feature, gt_img, _ = batch
            
            (img_feature, _), _ = self(
                landmark = landmark,
                gt_img_feature = gt_img_feature,
                gt_img = gt_img,
                training=False
            )
            
                
        return {"y_pred": img_feature, "y": gt_img_feature}
        
    def inference(self, batch, device, save_folder):
        with torch.no_grad():
            landmark, gt_img_feature, gt_mask_img_feature, gt_img, gt_img_path = batch
            
            # Forward pass with attention weights output
            (pred_img_feature, attn_weights), _ = self(
                landmark=landmark,
                gt_img_feature=gt_img_feature,
                gt_img=gt_img,
                training=False
            )
            
            gt_img_feature_list = gt_img_feature.tolist()
            pred_img_feature_list = pred_img_feature.tolist()
            
            data = {
                "gt_img_feature": gt_img_feature_list,
                "pred_img_feature": pred_img_feature_list,
                "gt_img_path": gt_img_path
            }
            
            os.makedirs(os.path.join(save_folder, "landmarks"), exist_ok=True)
            
            with open(os.path.join(save_folder, 'tensor_data.json'), 'w') as json_file:
                json.dump(data, json_file)
            
            gt_img_feature = gt_img_feature.permute(0, 2, 3, 1)
            pred_img_feature = pred_img_feature.permute(0, 2, 3, 1)

            for i in range(landmark.shape[0]):
                image_size = 32
                output_file = os.path.join(save_folder, f'landmarks/landmarks_{i}.png')
                
                gt_lm = gt_img_feature[i][:,:,:3]
                pred_lm = pred_img_feature[i][:,:,:3].detach().cpu()

                combined_image = np.ones((image_size, image_size * 2, 3), dtype=np.uint8)
                combined_image[:, :image_size, :] = gt_lm
                combined_image[:, image_size:image_size*2, :] = pred_lm

                if attn_weights is not None:
                    # Convert attention weights to a heatmap format (assuming attention is applied on a flattened 32x32 grid)
                    attn_weights_i = attn_weights[i].mean(dim=0)  # Average across all heads
                    attn_size = int(attn_weights_i.numel() ** 0.5)  # Compute square root of number of elements
                    attn_map = attn_weights_i.view(attn_size, attn_size).detach().cpu().numpy()

                    # Interpolate attention map to 32x32 if necessary
                    if attn_map.shape != (32, 32):
                        attn_map = cv2.resize(attn_map, (32, 32))


                # Tạo subplots với 3 axes
                fig, axes = plt.subplots(1, 3, figsize=(18, 4))

                # Phần 1: Ảnh Ground Truth
                axes[0].imshow(combined_image[:, :image_size, :])
                axes[0].set_title('Ground Truth')
                axes[0].axis('off')

                # Phần 2: Ảnh Prediction
                axes[1].imshow(combined_image[:, image_size:image_size*2, :])
                axes[1].set_title('Prediction')
                axes[1].axis('off')
                
                if attn_weights is not None:
                    # Phần 3: Attention Map (as a heatmap)
                    axes[2].imshow(attn_map, cmap='viridis')
                    axes[2].set_title('Attention Map')
                    axes[2].axis('off')

                # Lưu ảnh vào file
                plt.savefig(output_file, bbox_inches='tight')
                plt.close()
