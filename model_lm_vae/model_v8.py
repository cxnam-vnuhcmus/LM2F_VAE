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

from .landmark_encoder_v8 import FusionModel
# from .loss_v1 import CustomLoss

#DS_V0
class Model(nn.Module):

    def __init__(self,
                 pretrained: Union[bool, str] = True,
                 infer_samples: bool = False
                 ):
        super().__init__()
        
        self.pretrained = pretrained
        self.infer_samples = infer_samples
        
        self.model = FusionModel()
        
        # self.criterion = CustomLoss()
        
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def forward(self,
                landmark,
                gt_img_feature,
                gt_img
                ):
        landmark = landmark.to(self.device)
        gt_img_feature = gt_img_feature.to(self.device)
        gt_img = gt_img.to(self.device)
        
        landmark = landmark.reshape(landmark.size(0), -1)
        mask_gt_img_feature = gt_img_feature.clone()
        mask_gt_img_feature[:, :, 4*4:7*4, 2*4:6*4] = 1
        # noise = torch.randn(gt_img_feature.shape).to(self.device)
        # noise = gt_img_feature[0].clone().unsqueeze(0).repeat(32,1,1,1).to(self.device)
        
        pred_img_feature =  self.model(mask_gt_img_feature, landmark)        
        
        loss = self.model.loss_function(pred_img_feature, gt_img_feature)

        return (pred_img_feature), loss
        
    # def loss_fn(self, pred_features, gt_features, pred_lm, gt_lm):
    #     loss = self.criterion(pred_features, gt_features, pred_lm, gt_lm)

    #     return loss

        
    def training_step_imp(self, batch, device) -> torch.Tensor:
        landmark, gt_img_feature, gt_img, _ = batch
        _, loss = self(
            landmark = landmark,
            gt_img_feature = gt_img_feature,
            gt_img = gt_img
        )
        
        return loss

    def eval_step_imp(self, batch, device):
        with torch.no_grad():
            landmark, gt_img_feature, gt_img, _ = batch
            
            (img_feature), _ = self(
                landmark = landmark,
                gt_img_feature = gt_img_feature,
                gt_img = gt_img
            )
            
                
        return {"y_pred": img_feature, "y": gt_img_feature}
        
    def inference(self, batch, device, save_folder):
        with torch.no_grad():
            landmark, gt_img_feature, gt_img, gt_img_path = batch
            
            (pred_img_feature), _ = self(
                landmark = landmark,
                gt_img_feature = gt_img_feature,
                gt_img = gt_img
            )
            
            gt_img_feature_list = gt_img_feature.tolist()
            pred_img_feature_list = pred_img_feature.tolist()
            
            data = {
                "gt_img_feature": gt_img_feature_list,
                "pred_img_feature": pred_img_feature_list,
                "gt_img_path": gt_img_path
            }
            
            os.makedirs(os.path.join(save_folder, "landmarks"), exist_ok=True)
            
            with open(os.path.join(save_folder,'tensor_data.json'), 'w') as json_file:
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