from ignite.metrics import Metric
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        # self.mse_loss = nn.MSELoss()
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.kd_loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target, recon_img=None, gt_img=None):
        assert pred.shape == target.shape, f"Shape mismatch: pred shape {pred.shape}, target shape {target.shape}"

        total_loss = 0
        for i in range(pred.shape[1]):  
            total_loss += F.mse_loss(pred[:, i, :, :], target[:, i, :, :])
            
        log_pred_features = torch.log_softmax(pred, dim=-1)
        log_target_features = torch.softmax(target, dim=-1)
        kd_loss = self.kd_loss_fn(log_pred_features, log_target_features)
        total_loss += kd_loss
        
        return total_loss

class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def forward(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss
