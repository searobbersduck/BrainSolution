import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.feature import local_binary_pattern
import numpy as np
from torchvision.models.vgg import vgg16

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


# ref: https://github.com/leftthomas/SRGAN/blob/master/loss.py
class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss
        tv_loss = self.tv_loss(out_images)
        return image_loss + 0.001 * adversarial_loss + 0.006 * perception_loss + 2e-8 * tv_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

