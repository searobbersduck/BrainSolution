import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.feature import local_binary_pattern
import numpy as np

class LBPLoss(nn.Module):
    def __init__(self, radius=1, n_points=8):
        super(LBPLoss, self).__init__()
        self.radius = radius
        self.n_points = n_points

    def forward(self, input, target):
        '''
        input: gpu variable tensor, (n,c,d,h,w)
        target: gpu tensor, (n,c,d,h,w)
        '''
        # flat
        input = input.detach().cpu().numpy() # n,c,d,h,w
        input = np.reshape(input, [-1, input.shape[-1]])
        target = target.detach().cpu().numpy()
        target = np.reshape(target, [-1, input.shape[-1]])
        lbp_input = local_binary_pattern(input, self.n_points, self.radius)
        lbp_target = local_binary_pattern(target, self.n_points, self.radius)
        loss = torch.nn.L1Loss()(torch.from_numpy(lbp_input), torch.from_numpy(lbp_target))
        return loss

class MaskLBPLoss(nn.Module):
    def __init__(self, radius=1, n_points=8):
        super(MaskLBPLoss, self).__init__()
        self.radius = radius
        self.n_points = n_points

    def forward(self, input, target, mask):
        '''
        input: gpu variable tensor, (n,c,d,h,w)
        target: gpu tensor, (n,c,d,h,w)
        mask: numpy arr, (n,c,d,h,w)
        '''
        # flat
        input = input.detach().cpu().numpy() # n,c,d,h,w
        input = np.reshape(input, [-1, input.shape[-1]])
        target = target.detach().cpu().numpy()
        target = np.reshape(target, [-1, input.shape[-1]])
        mask = np.reshape(mask, [-1, input.shape[-1]])
        input = input*mask
        target = target*mask
        lbp_input = local_binary_pattern(input, self.n_points, self.radius)
        lbp_target = local_binary_pattern(target, self.n_points, self.radius)
        loss = torch.nn.L1Loss()(torch.from_numpy(lbp_input), torch.from_numpy(lbp_target))
        return loss