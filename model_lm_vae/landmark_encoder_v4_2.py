import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from .blocks import SPADE, AdaIN, convert_flow_to_deformation, make_coordinate_grid, warping


class FusionModel(nn.Module):
    def __init__(self):
        super(FusionModel, self).__init__()
        self.spade_layer_1 = SPADE(8, 20, 32)
        self.spade_layer_2 = SPADE(8, 20, 32)
        self.spade_layer_3 = SPADE(8, 20, 32)
        self.conv_4 = torch.nn.Conv2d(8, 2, kernel_size=7, stride=1, padding=3)
        self.conv_5= nn.Sequential(torch.nn.Conv2d(8, 4, kernel_size=7, stride=1, padding=3),
                                   torch.nn.ReLU(),
                                   torch.nn.Conv2d(4, 1, kernel_size=7, stride=1, padding=3),
                                   torch.nn.Sigmoid(),
                                   )#predict weight
        
        self.spade_layer_4 = SPADE(40, 8, 32)
        # self.adain_layer_4 = AdaIN(20, 8)
        self.spade_layer_5 = SPADE(40, 8, 32)
        # self.adain_layer_5 = AdaIN(20, 8)
        self.spade_layer_6 = SPADE(40, 4, 32)
        
        self.leaky_relu = torch.nn.LeakyReLU()
        self.conv_last = torch.nn.Conv2d(in_channels=8, out_channels=4, kernel_size=7, stride=1, padding=3, bias=False)
        
    def forward(self, mask_gt_img_feature, gt_lm_feature, ref_img_feature, ref_lm_feature): #(B,N,4,32,32)
        target_input = torch.cat([mask_gt_img_feature, gt_lm_feature], dim=2)
        ref_input = torch.cat([ref_img_feature, ref_lm_feature], dim=2)
        
        driving_sketch = gt_lm_feature.reshape(gt_lm_feature.size(0),-1,32,32) #(B,20,32,32)
        
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
        
        target_input = target_input.reshape(target_input.size(0),-1,32,32)
        x = self.spade_layer_4(target_input, wrapped_spade_sum)
        x = self.spade_layer_5(x, wrapped_spade_sum)
        x = self.spade_layer_6(x, wrapped_ref_sum)
        
        x = x.reshape(-1,8,32,32)
        x = self.leaky_relu(x)
        x = self.conv_last(x)
        x = x.reshape(target_input.size(0),5,4,32,32)
        
        return x

    def loss_function(self, pred_image, gt_image):
        # Calculate MSE loss for the entire image
        mse_loss_value = nn.MSELoss()(pred_image, gt_image)

        return mse_loss_value
