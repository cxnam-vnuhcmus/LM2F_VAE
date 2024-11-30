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
from .landmark_encoder_v7_1 import FusionModel
from .visual_encoder import VisualEncoder
from .loss import VGG19FeatureExtractor, perceptual_loss
from .discriminator import Discriminator
from .loss import AdversarialLoss

#DS_V8 LM_V7_1 SPADERefImg GanLoss
class Model(nn.Module):

    def __init__(self,
                 pretrained: Union[bool, str] = True,
                 infer_samples: bool = False
                 ):
        super().__init__()
        
        self.pretrained = pretrained
        self.infer_samples = infer_samples
        
        self.model = FusionModel()
        
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
            
        self.adversarial_loss = AdversarialLoss(type='lsgan')               
        self.discriminator = Discriminator(in_channels=3, use_sigmoid=True)
        
        self.kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.rec_loss_fn = nn.MSELoss()
        self.feat_loss_fn = nn.MSELoss()
                
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

        pred_img_feature =  self.model(gt_mask_img_feature, gt_lm_feature, ref_img_feature, ref_lm_feature)         
        latents = pred_img_feature.reshape(-1, 4, 32, 32)
        pred_img = self.vsencoder.decode(latents)
        gt_img = gt_img.reshape(-1, 3, 256, 256)
        
        loss_G, loss_D = self.loss_function(pred_img_feature, gt_img_feature, pred_img, gt_img, None, None)

        return pred_img_feature, (loss_G, loss_D), pred_img, gt_img
    
    def loss_function(self, pred_feature, gt_feature, pred_img, gt_img, pred_lm, gt_lm):
        # pred_feature = pred_feature.reshape(-1, 4, 32, 32)
        # gt_feature = gt_feature.reshape(-1, 4, 32, 32)
        # log_pred_feature = F.log_softmax(pred_feature, dim=2)
        # log_gt_feature = F.softmax(gt_feature, dim=2)
        # kd_loss = self.kd_loss_fn(log_pred_feature, log_gt_feature)
        rec_loss = self.rec_loss_fn(pred_img, gt_img)
        p_loss = perceptual_loss(pred_img, gt_img, self.vgg19)
        # lm_loss = nn.MSELoss()(pred_lm, gt_lm)
        feat_loss = self.feat_loss_fn(pred_feature, gt_feature)

        # discriminator
        pred_img_gan = pred_img.clone().detach()
        pred_img_gan = pred_img_gan.squeeze(1)
        gt_img_gan = gt_img.clone()
        gt_img_gan = gt_img_gan.squeeze(1)
        pred_fake, _ = self.discriminator(pred_img_gan)
        loss_D_fake = self.adversarial_loss(pred_fake, False, is_disc=True)
        # Real Detection and Loss
        pred_real, _ = self.discriminator(gt_img_gan.clone().detach())
        loss_D_real = self.adversarial_loss(pred_real, True, is_disc=True)
        loss_D = (loss_D_fake + loss_D_real).mean() * 0.5
        #
        # GAN loss
        pred_fake, _ = self.discriminator(gt_img_gan)
        loss_G_GAN = self.adversarial_loss(pred_fake, True, is_disc=False).mean()
        
        loss_G = 5.0 * feat_loss \
        # 0.0001 * kd_loss \
        + 5.0 * rec_loss \
        + 4.0 * p_loss \
        + 2.5 * loss_G_GAN
        
        return loss_G, loss_D
    
    def training_step_imp(self, batch, device) -> torch.Tensor:
        _, loss, _, _ = self(batch)
        
        return loss

    def eval_step_imp(self, batch, device):
        gt_img_feature = batch["gt_img_feature"]
        
        with torch.no_grad():
            pred_img_feature, _, _, _ = self(batch)
                            
        return {"y_pred": pred_img_feature, "y": gt_img_feature}
        
    def inference(self, batch, device, save_folder):
        with torch.no_grad():
            gt_img_feature = batch["gt_img_feature"]
            
            pred_img_feature, _, pred_img, gt_img = self(batch)
            
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
            pred_img = pred_img.permute(0, 2, 3, 1).detach().cpu().numpy()
            gt_img = gt_img.permute(0, 2, 3, 1).detach().cpu().numpy()

            for i in range(gt_img_feature.shape[0]):
                image_size = 32
                output_file = os.path.join(save_folder, f'landmarks/landmarks_{i}.png')
                
                gt_lm = gt_img_feature[i][:,:,:]
                pred_lm = pred_img_feature[i][:,:,:].detach().cpu()

                combined_image = np.ones((image_size, image_size * 2, 4), dtype=np.uint8)
                combined_image[:, :image_size, :] = gt_lm
                combined_image[:, image_size:image_size*2, :] = pred_lm

                # Tạo subplots
                fig, axes = plt.subplots(1, 4, figsize=(12, 4))

                # Phần 1: Ảnh background + Ground Truth
                axes[0].imshow(combined_image[:, :image_size, :])
                axes[0].set_title('Ground Truth')
                axes[0].axis('off')

                # Phần 2: Ảnh background + Prediction
                axes[1].imshow(combined_image[:, image_size:image_size*2, :])
                axes[1].set_title('Prediction')
                axes[1].axis('off')

                axes[2].imshow(gt_img[i])  # For color image, no cmap is needed
                axes[2].set_title('Groudtruth Image')
                axes[2].axis('off')

                axes[3].imshow(pred_img[i])  # For color image, no cmap is needed
                axes[3].set_title('Predicted Image')
                axes[3].axis('off')
                
                # Lưu ảnh vào file
                plt.savefig(output_file, bbox_inches='tight')
                plt.close()
