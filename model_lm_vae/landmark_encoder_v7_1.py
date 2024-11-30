import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from .blocks import SPADE, AdaIN, convert_flow_to_deformation, make_coordinate_grid, warping


class FusionModel(nn.Module):
    def __init__(self, n_target_frame=5, n_ref_frame=3):
        super(FusionModel, self).__init__()
        self.n_target_frame = n_target_frame
        self.n_ref_frame = n_ref_frame
        self.spade_layer_1 = SPADE(8, 4*n_target_frame, 32)
        self.spade_layer_2 = SPADE(8, 4*n_target_frame, 32)
        self.spade_layer_3 = SPADE(8, 4*n_target_frame, 32)
        self.conv_4 = torch.nn.Conv2d(8, 2, kernel_size=7, stride=1, padding=3)
        self.conv_5= nn.Sequential(torch.nn.Conv2d(8, 4, kernel_size=7, stride=1, padding=3),
                                   torch.nn.ReLU(),
                                   torch.nn.Conv2d(4, 1, kernel_size=7, stride=1, padding=3),
                                   torch.nn.Sigmoid(),
                                   )#predict weight
        
        self.spade_layer_4 = SPADE(24, 8, 32)
        self.adain_layer_4 = SPADE(24, 4*self.n_ref_frame, 32)
        self.spade_layer_5 = SPADE(24, 8, 32)
        self.adain_layer_5 = SPADE(24, 4*self.n_ref_frame, 32)
        self.spade_layer_6 = SPADE(24, 4, 32)
        
        self.leaky_relu = torch.nn.LeakyReLU()
        self.conv_last = torch.nn.Conv2d(in_channels=24, out_channels=4, kernel_size=7, stride=1, padding=3, bias=False)
        
    def forward(self, mask_gt_img_feature, gt_lm_feature, ref_img_feature, ref_lm_feature): #(B,N,4,32,32)
        B, N_img, C, H, W = mask_gt_img_feature.shape
        N_lm = gt_lm_feature.size(1)
        N_ref = ref_img_feature.size(1)
        mask_gt_img_feature = mask_gt_img_feature.reshape(B, N_img*C, H, W) #(B,1*4,32,32)
        gt_lm_feature = gt_lm_feature.reshape(B, N_lm*C, H, W) #(B,5*4=20,32,32)
        target_input = torch.cat([mask_gt_img_feature, gt_lm_feature], dim=1) #(B,24,32,32)
        ref_input = torch.cat([ref_img_feature, ref_lm_feature], dim=2) #(B,3,8,32,32)
        
        driving_sketch = gt_lm_feature
        
        wrapped_spade_sum, wrapped_ref_sum=0.,0.
        softmax_denominator=0.
        for ref_idx in range(ref_input.size(1)):
            ref_input_by_idx = ref_input[:,ref_idx] #(B,8,32,32)
            ref_img_feature_by_idx = ref_img_feature[:, ref_idx] #(B,4,32,32)
            
            spade_output_1 = self.spade_layer_1(ref_input_by_idx, driving_sketch)
            spade_output_2 = self.spade_layer_2(spade_output_1, driving_sketch)
            spade_output_3 = self.spade_layer_3(spade_output_2, driving_sketch)
            
            output_flow = self.conv_4(spade_output_3)
            output_weight = self.conv_5(spade_output_3)
            
            deformation=convert_flow_to_deformation(output_flow)
            wrapped_spade = warping(spade_output_3, deformation)  #(32,8,32,32)
            wrapped_ref = warping(ref_img_feature_by_idx, deformation)  #(32,4,32,32)
            
            softmax_denominator += output_weight
            wrapped_spade_sum += wrapped_spade * output_weight
            wrapped_ref_sum += wrapped_ref * output_weight
            
        softmax_denominator += 0.00001
        wrapped_spade_sum = wrapped_spade_sum/softmax_denominator
        wrapped_ref_sum = wrapped_ref_sum / softmax_denominator
        
        ref_img_feature = ref_img_feature.reshape(B, N_ref * C, H, W)
        target_input = target_input.reshape(B,-1,H,W)
        x = self.spade_layer_4(target_input, wrapped_spade_sum) #(B, 24, 32, 32)
        x = self.adain_layer_4(x, ref_img_feature)
        x = self.spade_layer_5(x, wrapped_spade_sum)
        x = self.adain_layer_5(x, ref_img_feature)
        x = self.spade_layer_6(x, wrapped_ref_sum)
        
        x = self.leaky_relu(x)
        x = self.conv_last(x)  #(B, 4, 32, 32)
        x = x.reshape(B,N_img,C,H,W) #(B, 1, 4, 32, 32)
        return x
