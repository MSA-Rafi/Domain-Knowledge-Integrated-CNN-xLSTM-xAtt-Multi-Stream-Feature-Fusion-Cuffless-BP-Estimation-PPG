import torch
import torch.nn as nn

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0, reduction='mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, input, target):
        error = input - target
        abs_error = torch.abs(error)
        loss = torch.where(abs_error < self.delta,
                           0.5 * error ** 2,
                           self.delta * (abs_error - 0.5 * self.delta))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def pearson_corr(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    cov_xy = torch.sum((x - x_mean) * (y - y_mean))
    std_x = torch.sqrt(torch.sum((x - x_mean) ** 2))
    std_y = torch.sqrt(torch.sum((y - y_mean) ** 2))
    return cov_xy / (std_x * std_y)