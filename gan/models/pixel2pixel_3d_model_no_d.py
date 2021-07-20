'''
@Description: 
@Version: 1.0
@Autor: searobbersanduck
@Date: 2020-03-30 15:44:31
LastEditors: searobbersanduck
LastEditTime: 2020-10-22 14:08:48
@License : (C)Copyright 2020-2021, MIT
'''

import os
import sys

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn import init
import functools
from torch.optim import lr_scheduler

import os
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, 'external_lib/pytorch-CycleGAN-and-pix2pix/models'))

import networks


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        
        p = 1
        conv_block += [nn.Conv3d(dim, dim//2, kernel_size=1, padding=0, bias=use_bias), norm_layer(dim//2), nn.ReLU(True)]
        conv_block += [nn.Conv3d(dim//2, dim//2, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim//2), nn.ReLU(True)]
        conv_block += [nn.Conv3d(dim//2, dim, kernel_size=1, padding=0, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        
        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

        
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='zero'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        model = [nn.Conv3d(input_nc, ngf, kernel_size=7, padding=3, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 3
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=3)]
        # model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class Pix2PixModel():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device('cuda', self.local_rank)

        self.netG_cpu = ResnetGenerator(1,1, 32, n_blocks=6)

        if opt.netG_model_path is not None:
            self.netG_cpu.load_state_dict(torch.load(opt.netG_model_path))

        self.isTrain = True
        
        if self.isTrain:
            self.criterionL1 = torch.nn.L1Loss()

        self.netG = self.netG_cpu.to(self.device)
        self.netG = torch.nn.parallel.DistributedDataParallel(self.netG, device_ids=[self.local_rank], output_device=self.local_rank)

        if self.isTrain:
            self.optimizer_G = torch.optim.Adam(self.netG_cpu.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.scaler = torch.cuda.amp.GradScaler()


    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].cuda()
        self.real_B = input['B' if AtoB else 'A'].cuda()
        if 'mask' in input:
            self.mask = input['mask'].cuda()
        else:
            self.mask = None
        if 'lbp_mask' in input:
            self.lbp_mask = input['lbp_mask']
        else:
            self.lbp_mask = None
        self.image_paths = input['A_paths' if AtoB else 'B_paths']


    def optimize_parameters(self):
        with torch.cuda.amp.autocast():
            self.fake_B = self.netG(self.real_A)
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
            self.loss_G = self.loss_G_L1
        self.scaler.scale(self.loss_G).backward()
        self.scaler.step(self.optimizer_G)
        self.scaler.update()

        self.optimizer_G.zero_grad()


    def reduce_tensor(self, tensor:torch.Tensor):
        rt = tensor.clone()
        torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
        rt /= torch.distributed.get_world_size()
        return rt


    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        # loss_G = self.reduce_tensor(self.loss_G).detach().cpu().numpy()
        # loss_D = self.reduce_tensor(self.loss_D).detach().cpu().numpy()
        
        netG_out_model_file = 'pixel2pixel_netG_epoch_{}_loss_{:.4f}.pth'.format(epoch, self.loss_G.detach().cpu().numpy())
        torch.save(self.netG.module.state_dict(), 
            os.path.join(self.save_dir, netG_out_model_file))
        # netD_out_model_file = 'pixel2pixel_netD_epoch_{}_loss_{:.4f}.pth'.format(epoch, self.loss_D.detach().cpu().numpy())    
        # torch.save(self.netD.module.state_dict(), 
        #     os.path.join(self.save_dir, netD_out_model_file))

        print('====> save model:\t{}'.format(netG_out_model_file))
        # print('====> save model:\t{}'.format(netD_out_model_file))