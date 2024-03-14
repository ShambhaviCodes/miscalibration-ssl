
import torch
import torch.nn as nn
import torch.nn.functional as F

class Calibration_Penalty(nn.Module):
    """Add marginal penalty to logits:
        max(0, max(l^n) - l^n - margin)
    """
    def __init__(self):
        super().__init__()

    def get_diff(self, inputs):
        max_values = inputs.max(dim=1)
        max_values = max_values.values.unsqueeze(dim=1).repeat(1, inputs.shape[1])
        diff = max_values - inputs
        return diff

    def forward(self, inputs, targets, margin_hyperparam):
        diff = self.get_diff(inputs)
        loss_margin = F.relu(diff - margin_hyperparam).mean() 
        return loss_margin
