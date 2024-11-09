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
from .landmark_encoder_v3 import FusionModel

#DS_V4 LM_V3
class Model(nn.Module):

    def __init__(self,
                 pretrained: Union[bool, str] = True,
                 infer_samples: bool = False
                 ):
        super().__init__()
        
        self.pretrained = pretrained
        self.infer_samples = infer_samples
        
        self.channel_idx = [0]
        
        self.model_list = []
        for i in self.channel_idx:
            self.model = FusionModel()
            self.model_list.append(self.model.to(self.device))
        
        
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

        pred_img_feature_list = gt_img_feature.clone()
        total_loss = 0.0
        for idx, i in enumerate(self.channel_idx):
            gt_img_feature_i = gt_img_feature[:,i,:,:].unsqueeze(1)
            gt_min, gt_max = gt_img_feature_i.min(), gt_img_feature_i.max()
            
            mask_gt_img_feature = gt_img_feature_i.clone()        
            noise = torch.randn(gt_img_feature_i.shape).to(self.device)
            mask_gt_img_feature[:, :, 4*4:7*4, 2*4:6*4] += noise[:, :, 4*4:7*4, 2*4:6*4]
            
            ref_img_feature = gt_img_feature_i.clone()
            ref_img_feature = ref_img_feature[torch.randperm(ref_img_feature.size(0))]
            
            pred_img_feature =  self.model_list[idx](mask_gt_img_feature, landmark, ref_img_feature)        
            # pred_min, pred_max = pred_img_feature.min(), pred_img_feature.max()
            # pred_norm = (pred_img_feature - pred_min) / (pred_max - pred_min + 1e-5)
            # pred_adjusted = pred_norm * (gt_max - gt_min) + gt_min
            # pred_adjusted = torch.clamp(pred_adjusted, gt_min, gt_max)

            pred_img_feature_list[:,i:i+1,:,:] = pred_img_feature
            
            loss = self.model_list[idx].loss_function(pred_img_feature, gt_img_feature_i)
            total_loss += loss
        
        return (pred_img_feature_list, None), total_loss
        
        
    def training_step_imp(self, batch, device) -> torch.Tensor:
        landmark, gt_img_feature, gt_mask_img_feature, gt_img, _ = batch
        _, loss = self(
            landmark = landmark,
            gt_img_feature = gt_img_feature,
            gt_img = gt_img
        )
        
        return loss

    def eval_step_imp(self, batch, device):
        with torch.no_grad():
            landmark, gt_img_feature, gt_mask_img_feature, gt_img, _ = batch
            
            gt_img_feature_org = gt_img_feature.clone()

            (img_feature, _), _ = self(
                landmark = landmark,
                gt_img_feature = gt_img_feature,
                gt_img = gt_img
            )
            
            gt_img_feature_org = gt_img_feature_org.to(device)
            gt_img_feature_list = []
            for i in self.module.channel_idx:
                gt_img_feature_list.append(gt_img_feature_org[:, i, :, :].unsqueeze(1))
            gt_img_feature_list = torch.cat(gt_img_feature_list, dim=1)
                
        return {"y_pred": img_feature, "y": gt_img_feature_list}
        
    def inference(self, batch, device, save_folder):
        with torch.no_grad():
            landmark, gt_img_feature, gt_mask_img_feature, gt_img, gt_img_path = batch
            
            # Forward pass with attention weights output
            (pred_img_feature, attn_weights), _ = self(
                landmark=landmark,
                gt_img_feature=gt_img_feature,
                gt_img=gt_img
            )
            
            os.makedirs(os.path.join(save_folder, "landmarks"), exist_ok=True)

            gt_img_feature_list = gt_img_feature.tolist()
            pred_img_feature_list = pred_img_feature.tolist()            
            data = {
                "gt_img_feature": gt_img_feature_list,
                "pred_img_feature": pred_img_feature_list,
                "gt_img_path": gt_img_path
            }
            with open(os.path.join(save_folder, 'tensor_data.json'), 'w') as json_file:
                json.dump(data, json_file)
            
            gt_img_feature = gt_img_feature.permute(0, 2, 3, 1)                
            pred_img_feature = pred_img_feature.permute(0, 2, 3, 1)

            # Chỉ lưu 5 ảnh từ batch
            num_to_save = min(2, gt_img_feature.shape[0])

            for i in range(num_to_save):
                output_file = os.path.join(save_folder, f'landmarks/landmarks_{i}.png')

                # Lấy từng channel của GT và Prediction
                gt_channels = [gt_img_feature[i][:, :, ch].detach().cpu().numpy() for ch in range(4)]
                pred_channels = [pred_img_feature[i][:, :, ch].detach().cpu().numpy() for ch in range(4)]
                
                # Tạo subplots với 4 axes (tương ứng với 4 kênh)
                fig, axes = plt.subplots(3, 4, figsize=(24, 12))

                for ch in range(4):
                    # Tính sự khác biệt giữa GT và Prediction cho từng kênh
                    diff = abs(pred_channels[ch] - gt_channels[ch])

                    # Normalize từng kênh để hiển thị heatmap trong khoảng [0, 1]
                    # gt_channels[ch] = (gt_channels[ch] - gt_channels[ch].min()) / (gt_channels[ch].max() - gt_channels[ch].min() + 1e-5)
                    # pred_channels[ch] = (pred_channels[ch] - pred_channels[ch].min()) / (pred_channels[ch].max() - pred_channels[ch].min() + 1e-5)
                    diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-5)

                    # Phần 1: GT (Heatmap của từng kênh)
                    sns.heatmap(gt_channels[ch], ax=axes[0, ch], cbar=True)
                    axes[0, ch].set_title(f'GT Channel {ch}')
                    axes[0, ch].axis('off')

                    # Phần 2: Prediction (Heatmap của từng kênh)
                    sns.heatmap(pred_channels[ch], ax=axes[1, ch],  cbar=True)
                    axes[1, ch].set_title(f'Pred Channel {ch}')
                    axes[1, ch].axis('off')

                    # Phần 3: Hiệu giữa Prediction và GT (Difference Heatmap)
                    sns.heatmap(diff, ax=axes[2, ch], vmin=0, vmax=1,cmap="Reds",  cbar=True)  # 'RdBu' sẽ làm rõ sự khác biệt
                    axes[2, ch].set_title(f'Difference Channel {ch}')
                    axes[2, ch].axis('off')

                # Lưu ảnh vào file
                plt.savefig(output_file, bbox_inches='tight')
                plt.close()




