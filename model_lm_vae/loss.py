from ignite.metrics import Metric
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from .utils import FACEMESH_ROI_IDX, FACEMESH_LIPS_IDX, FACEMESH_FACES_IDX
import numpy as np
import mediapipe as mp
import cv2

mapped_lips_indices = [FACEMESH_ROI_IDX.index(i) for i in FACEMESH_LIPS_IDX]
mapped_faces_indices = [FACEMESH_ROI_IDX.index(i) for i in FACEMESH_FACES_IDX]

# Custom metric class
class CustomMetric(Metric):
    def __init__(self, output_transform=lambda x: x, device=None):
        self._sum_loss = None
        self._num_examples = None
        self.customLoss = CustomLoss()
        super(CustomMetric, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._sum_loss = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output[0].cpu() , output[1].cpu() 
        
        loss = self.customLoss(y_pred, y)
        self._sum_loss += loss.item()
        self._num_examples += y_pred.shape[0]
        

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomMetric must have at least one example before it can be computed')
        output = (self._sum_loss / self._num_examples)
        return output
        

class CustomLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.kd_loss_fn = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred, target):
        pred = pred.cpu()
        target = target.cpu()
        
        mse_loss = self.mse_loss(pred, target)
        # bce_loss = self.bce_loss(pred, target)
        
        log_pred_features = torch.log_softmax(pred, dim=-1)
        log_target_features = torch.softmax(target, dim=-1)
        kd_loss = self.kd_loss_fn(log_pred_features, log_target_features)
        
        total_loss = self.alpha * mse_loss + self.beta * kd_loss
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

class VGG19FeatureExtractor(nn.Module):
    def __init__(self, layers=None):
        super(VGG19FeatureExtractor, self).__init__()
        
        # Load pretrained VGG19 model
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        # Only keep the specified layers for feature extraction
        self.layers = nn.Sequential(*list(vgg19.children())[:layers])
    
    def forward(self, x):
        return self.layers(x)

# Define perceptual loss function using VGG19
def perceptual_loss(generated, target, vgg19_model):
    # Make sure both images are in the same format (B, C, H, W)
    generated = generated.unsqueeze(0) if generated.dim() == 3 else generated
    target = target.unsqueeze(0) if target.dim() == 3 else target
    
    # Normalize images to match ImageNet statistics
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    generated = preprocess(generated)
    target = preprocess(target)
    
    vgg19_model.eval()
    
    # Extract features from the VGG model
    with torch.no_grad():
        generated_features = vgg19_model(generated)
        target_features = vgg19_model(target)
    
    # Compute L2 loss between features of the generated and target images
    loss = nn.MSELoss()(generated_features, target_features)
    return loss
