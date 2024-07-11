import torch
import torch.nn as nn
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure as MSSIM

import torch.nn.functional as F

from piqa import SSIM
from piqa import MS_SSIM

class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)
    

class MSSIMLoss(MS_SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)
    

class BerHULoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, x, y):

        threshold = torch.max(torch.abs((x - y)), 2)[0]
        threshold = 0.2*torch.max(threshold, 2)[0]
        #threshold

        #I obtained a mask for thresholding pixel-wise
        mask = (torch.abs((x - y)) <= threshold[:,:,None,None]).int().float()

        threshold = threshold[:, None].expand(threshold.shape[0],x.shape[2],x.shape[3])[:,None,:]

        l1 = torch.abs((x - y))*mask
        l2 = (((x - y)**2 + threshold**2)/2*threshold)*(1. - mask)
        
        loss = torch.mean(l1 + l2, dim=(2, 3)).squeeze()

        if self.reduction=="mean":
            loss = torch.mean(loss)

        return loss
    
class SOBELLoss(nn.Module):
    def __init__(self, device, reduction="mean"):
        super().__init__()
        self.Sx = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])[None,None, :].to(device)
        self.Sy = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])[None,None, :].to(device)
        self.reduction = reduction


    def forward(self, x, y):
        
        lossx = torch.abs(F.conv2d(x, self.Sx, padding=1) - F.conv2d(y, self.Sx, padding=1))
        lossy = torch.abs(F.conv2d(x, self.Sy, padding=1) - F.conv2d(y, self.Sy, padding=1))
        loss = torch.mean(lossx + lossy, dim=(2, 3)).squeeze()

        if self.reduction=="mean":
            loss = torch.mean(loss)

        return loss