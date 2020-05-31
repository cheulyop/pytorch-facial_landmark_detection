import math
import torch
import torch.nn as nn


class WingLoss(nn.Module):
    def __init__(self, w, eps, reduction='mean'):
        super(WingLoss, self).__init__()
        self.w = w
        self.eps = eps
        self.reduction = reduction

    def forward(self, outputs, labels):
        w, eps = self.w, self.eps
        C = w - w * math.log(1 + w/eps)

        x = torch.abs(outputs.view(-1) - labels.view(-1))
        x_small = x[x < w]
        x_large = x[x >= w]

        wing_x = torch.sum(w * torch.log(1 + x_small / eps)) + torch.sum(x_large - C)

        if self.reduction == 'mean':
            return wing_x / len(x)
        elif self.reduction == 'sum':
            return wing_x


if __name__ == "__main__":
    criterion = WingLoss(w=10, eps=2)
    test_outputs = torch.randn(200, 136)
    test_labels = torch.randn(200, 136)
    test_outputs.requires_grad_(True)
    loss = criterion(test_outputs, test_labels)
    loss.backward()
    print(loss)
