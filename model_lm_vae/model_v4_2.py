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
import seaborn as sns
from .landmark_encoder_v4_2 import FusionModel

#DS_V7 LM_V4_2
class Model(nn.Module):

    def __init__(self,
                 pretrained: Union[bool, str] = True,
                 infer_samples: bool = False
                 ):
        super().__init__()
        
        self.pretrained = pretrained
        self.infer_samples = infer_samples
        
        self.model = FusionModel()
        
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def forward(self, batch):
        gt_lm_feature = batch["gt_lm_feature"].to(self.device)
        gt_img_feature = batch["gt_img_feature"].to(self.device)
        ref_lm_feature = batch["ref_lm_feature"].to(self.device)
        ref_img_feature = batch["ref_img_feature"].to(self.device)
        gt_mask_img_feature = batch["gt_mask_img_feature"].to(self.device)
                
        pred_img_feature =  self.model(gt_mask_img_feature, gt_lm_feature, ref_img_feature, ref_lm_feature)        
        
        loss = self.model.loss_function(pred_img_feature, gt_img_feature)
        
        return pred_img_feature, loss
        
        
    def training_step_imp(self, batch, device) -> torch.Tensor:
        _, loss = self(batch)
        
        return loss

    def eval_step_imp(self, batch, device):
        gt_img_feature = batch["gt_img_feature"]
        
        with torch.no_grad():
            pred_img_feature, _ = self(batch)
                            
        return {"y_pred": pred_img_feature, "y": gt_img_feature}
        
    def inference(self, batch, device, save_folder):
        with torch.no_grad():
            gt_img_feature = batch["gt_img_feature"]
            
            # Forward pass with attention weights output
            pred_img_feature, _ = self(batch)
            
            os.makedirs(os.path.join(save_folder, "landmarks"), exist_ok=True)
            
            gt_img_feature_list = gt_img_feature.tolist()
            pred_img_feature_list = pred_img_feature.tolist()            
            data = {
                "gt_img_feature": gt_img_feature_list,
                "pred_img_feature": pred_img_feature_list,
                "gt_img_path": batch["vs_path"]
            }
            with open(os.path.join(save_folder, 'tensor_data.json'), 'w') as json_file:
                json.dump(data, json_file)

            gt_img_feature = gt_img_feature[:,0]
            pred_img_feature = pred_img_feature[:, 0]            
            gt_img_feature = gt_img_feature.permute(0, 2, 3, 1)                
            pred_img_feature = pred_img_feature.permute(0, 2, 3, 1)

            for i in range(gt_img_feature.shape[0]):
                image_size = 32
                output_file = os.path.join(save_folder, f'landmarks/landmarks_{i}.png')
                
                gt_lm = gt_img_feature[i][:,:,:]
                pred_lm = pred_img_feature[i][:,:,:].detach().cpu()

                combined_image = np.ones((image_size, image_size * 2, 4), dtype=np.uint8)
                combined_image[:, :image_size, :] = gt_lm
                combined_image[:, image_size:image_size*2, :] = pred_lm

                # Tạo subplots
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))

                # Phần 1: Ảnh background + Ground Truth
                axes[0].imshow(combined_image[:, :image_size, :])
                axes[0].set_title('Ground Truth')
                axes[0].axis('off')

                # Phần 2: Ảnh background + Prediction
                axes[1].imshow(combined_image[:, image_size:image_size*2, :])
                axes[1].set_title('Prediction')
                axes[1].axis('off')
                
                # Lưu ảnh vào file
                plt.savefig(output_file, bbox_inches='tight')
                plt.close()



