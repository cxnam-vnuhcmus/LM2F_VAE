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
from .landmark_encoder_v6 import FusionModel
from .visual_encoder import VisualEncoder
from .loss import VGG19FeatureExtractor, perceptual_loss

#DS_V7 LM_V6
class Model(nn.Module):

    def __init__(self,
                 pretrained: Union[bool, str] = True,
                 infer_samples: bool = False
                 ):
        super().__init__()
        
        self.pretrained = pretrained
        self.infer_samples = infer_samples
        
        self.model = FusionModel(n_frame=3)
        
        self.vsencoder = VisualEncoder()
        for param in self.vsencoder.parameters():
            param.requires_grad = False
        for param in list(self.vsencoder.vae.parameters())[-2:]:
            param.requires_grad = True
        for param in list(self.vsencoder.vae.decoder.parameters())[-3:]: #108 midblock
            param.requires_grad = True
            
        self.vgg19 = VGG19FeatureExtractor(layers=18)
        for param in self.vgg19.parameters():
            param.requires_grad = False
            
        # self.pred_landmark = nn.Sequential(
        #     nn.Conv2d(4, 2, kernel_size=7, stride=1, padding=3),
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(2 * 32 * 32, 1 * 32 * 32),
        #     nn.ReLU(),
        #     nn.Linear(1 * 32 * 32, 131 * 2),
        #     nn.Sigmoid()
        # )
        
        self.kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.rec_loss_fn = nn.MSELoss()
                
    @property
    def device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device
    
    def forward(self,batch):
        gt_lm_feature = batch["gt_lm_feature"].to(self.device)
        gt_img_feature = batch["gt_img_feature"].to(self.device)
        ref_lm_feature = batch["ref_lm_feature"].to(self.device)
        ref_img_feature = batch["ref_img_feature"].to(self.device)
        gt_mask_img_feature = batch["gt_mask_img_feature"].to(self.device)
        gt_img = batch["gt_img"].to(self.device)
        # gt_lm = batch["gt_lm"].to(self.device)
        
        pred_img_feature =  self.model(gt_mask_img_feature, gt_lm_feature, ref_img_feature, ref_lm_feature)         
        latents = pred_img_feature.reshape(-1, 4, 32, 32)
        pred_img = self.vsencoder.decode(latents)
        gt_img = gt_img.reshape(-1, 3, 256, 256)
        
        # pred_lm = self.pred_landmark(latents)  #(B*N, 131*2)
        # gt_lm = gt_lm.view(gt_lm.size(0) * gt_lm.size(1), -1)  #(B*N, 131*2)
        
        loss = self.loss_function(pred_img_feature, gt_img_feature, pred_img, gt_img, None, None)
        
        return pred_img_feature, loss
    
    def loss_function(self, pred_feature, gt_feature, pred_img, gt_img, pred_lm, gt_lm):
        # pred_feature = pred_feature.reshape(-1, 4, 32, 32)
        # gt_feature = gt_feature.reshape(-1, 4, 32, 32)
        log_pred_feature = F.log_softmax(pred_feature, dim=2)
        log_gt_feature = F.softmax(gt_feature, dim=2)
        kd_loss = self.kd_loss_fn(log_pred_feature, log_gt_feature)
        rec_loss = self.rec_loss_fn(pred_img, gt_img)
        p_loss = perceptual_loss(pred_img, gt_img, self.vgg19)
        # lm_loss = nn.MSELoss()(pred_lm, gt_lm)
        
        loss = 0.0001 * kd_loss \
        + 10.0 * rec_loss \
        + 0.1 * p_loss \
        # + 50.0 * lm_loss
        
        return loss
    
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
