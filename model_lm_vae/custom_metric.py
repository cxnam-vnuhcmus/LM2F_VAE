from ignite.metrics import Metric
import torch
from torch import nn
import numpy as np

# Custom metric class
class CustomMetric(Metric):
    def __init__(self, output_transform=lambda x: x, device=None):
        self._sum_loss = None
        self._num_examples = None
        self.mse = nn.MSELoss(reduction='mean')
        super(CustomMetric, self).__init__(output_transform=output_transform, device=device)

    def reset(self):
        self._sum_loss = 0.0
        self._num_examples = 0

    def update(self, output):
        y_pred, y = output[0].cpu() , output[1].cpu() 
        
        loss = self.mse(y_pred, y)
        self._sum_loss += loss.item()
        self._num_examples += y_pred.shape[0] * y_pred.shape[1]
        

    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError('CustomMetric must have at least one example before it can be computed')
        output = (self._sum_loss / self._num_examples)
        return output
        