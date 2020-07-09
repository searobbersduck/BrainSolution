'''
@Description: 
@Version: 1.0
@Autor: searobbersanduck
@Date: 2020-03-30 15:44:31
@LastEditors: searobbersanduck
@LastEditTime: 2020-07-03 18:07:27
@License : (C)Copyright 2020-2021, MIT
'''

import os
import sys

import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from torch.autograd import Variable


import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import numpy as np

import sys
sys.path.append('../external_lib/pytorch-CycleGAN-and-pix2pix/models')
sys.path.append('../../external_lib/pytorch-CycleGAN-and-pix2pix/models')
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
from gan.losses.lbp_loss import LBPLoss, MaskLBPLoss

import networks

Tensor = torch.cuda.FloatTensor

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

class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm3d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        out = self.net(input)
        out = torch.nn.MaxPool3d(kernel_size=[out.shape[2], out.shape[3], out.shape[4]])(out).squeeze(-1).squeeze(-1).squeeze(-1)
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
            # model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
            #                              kernel_size=3, stride=2,
            #                              padding=1, output_padding=1,
            #                              bias=use_bias),
            #           norm_layer(int(ngf * mult / 2)),
            #           nn.ReLU(True)]
            model += [nn.Upsample(scale_factor=2, mode='trilinear'), 
                nn.Conv3d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias), 
                norm_layer(int(ngf * mult / 2)),nn.ReLU(True)]

            

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
        self.netG_cpu = ResnetGenerator(1,1, 32, n_blocks=6)
        self.netD_cpu = PixelDiscriminator(2,8)
        if self.opt.patch_D:
            self.netD_P_cpu = PixelDiscriminator(2,8)

        if opt.netG_model_path is not None:
            self.netG_cpu.load_state_dict(torch.load(opt.netG_model_path))
        if opt.netD_model_path is not None:
            self.netD_cpu.load_state_dict(torch.load(opt.netD_model_path))
        
        self.isTrain = True
        self.optimizers=[]
        if self.isTrain:
            # define loss functions
            # self.criterionGAN = networks.GANLoss(opt.gan_mode).cuda()
            self.criterionGAN = torch.nn.BCELoss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMaskLBP = MaskLBPLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG_cpu.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD_cpu.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            if self.opt.patch_D:
                self.optimizer_D_P = torch.optim.Adam(self.netD_P.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.netD_P = torch.nn.DataParallel(self.netD_P).cuda()
        self.netG = torch.nn.DataParallel(self.netG_cpu).cuda()
        self.netD = torch.nn.DataParallel(self.netD_cpu).cuda()


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
        
    def forward(self):
        self.netG = torch.nn.DataParallel(self.netG_cpu).cuda()
        self.netD = torch.nn.DataParallel(self.netD_cpu).cuda()
        # self.netG.train()
        # self.netD.train()
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)

        # add patch(block) discriminator calculate 
        # related params: patch_D(bool), num_patches_D(int), patch_size_D([z,y,x])
        if self.opt.patch_D:
            self.fake_B_patch = []
            self.real_B_patch = []
            self.real_A_patch = []
            d = self.real_A.size(2)
            h = self.real_A.size(3)
            w = self.real_A.size(4)

            for i in range(self.opt.num_patches_D):
                max_end_d = d - self.opt.patch_size_D[0]
                max_end_h = h - self.opt.patch_size_D[1]
                max_end_w = w - self.opt.patch_size_D[2]

                # print('d:{}\th:{}\tw:{}'.format(d,h,w))
                # print('max_end_d:{}\tmax_end_h:{}\tmax_end_w:{}'.format(max_end_d,max_end_h,max_end_w))
                start_d = np.random.randint(0, max_end_d)
                start_h = np.random.randint(0, max_end_h)
                start_w = np.random.randint(0, max_end_w)

                end_d = start_d + self.opt.patch_size_D[0]
                end_h = start_h + self.opt.patch_size_D[1]
                end_w = start_w + self.opt.patch_size_D[2]

                self.fake_B_patch.append(self.fake_B[:,:,start_d:end_d, start_h:end_h, start_w:end_w])
                self.real_B_patch.append(self.real_B[:,:,start_d:end_d, start_h:end_h, start_w:end_w])
                self.real_A_patch.append(self.real_A[:,:,start_d:end_d, start_h:end_h, start_w:end_w])

        
    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if self.mask is None:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        else:
            fake_AB = torch.cat((self.real_A*self.mask, self.fake_B*self.mask), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        valid = Variable(Tensor(self.real_A.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(self.real_A.size(0), 1).fill_(0.0), requires_grad=False)
        pred_fake = self.netD(fake_AB.detach())
        print('pred_fake\t', pred_fake.shape)
        print('fake\t', fake.shape)
        self.loss_D_fake = self.criterionGAN(pred_fake, fake)
        # Real
        if self.mask is None:
            real_AB = torch.cat((self.real_A, self.real_B), 1)
        else:
            real_AB = torch.cat((self.real_A*self.mask, self.real_B*self.mask), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, valid)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()


    def backward_D_P(self):
        self.loss_D_fake_patch = 0
        self.loss_D_real_patch = 0
        #from .networks import cal_gradient_penalty
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B

        for i in range(self.opt.num_patches_D):
            fake_AB_patch = torch.cat((self.real_A_patch[i], self.fake_B_patch[i]), 1)
            pred_fake_AB_patch = self.netD_P(fake_AB_patch.detach())
            self.loss_D_fake_patch += self.criterionGAN(pred_fake_AB_patch, False)

        for i in range(self.opt.num_patches_D):
            real_AB_patch = torch.cat((self.real_A_patch[i], self.real_B_patch[i]), 1)
            pred_real_AB_patch = self.netD_P(real_AB_patch.detach())
            self.loss_D_real_patch += self.criterionGAN(pred_real_AB_patch, True)

        self.loss_D_fake_patch = self.loss_D_fake_patch/self.opt.num_patches_D
        self.loss_D_real_patch = self.loss_D_real_patch/self.opt.num_patches_D
        self.loss_D_patch = (self.loss_D_fake_patch + self.loss_D_real_patch)/2
        self.loss_D_patch.backward(retain_graph=False)



    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.mask is None:
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        else:
            fake_AB = torch.cat((self.real_A*self.mask, self.fake_B*self.mask), 1)
        pred_fake = self.netD(fake_AB)
        valid = Variable(Tensor(self.real_A.size(0), 1).fill_(1.0), requires_grad=False)
        self.loss_G_GAN = self.criterionGAN(pred_fake, valid)
        # Second, G(A) = B
        # if self.mask is None and self.lbp_mask is None:
        #     self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # elif self.lbp_mask is not None:
        #     self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1 + self.criterionMaskLBP(self.fake_B, self.real_B, self.lbp_mask).float()
        # else:
        #     self.loss_G_L1 = self.criterionL1(self.fake_B*self.mask, self.real_B*self.mask) * self.opt.lambda_L1
        # combine loss and calculate gradients
        # self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G = self.loss_G_GAN
        self.loss_G.backward()
        
    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        
        # update D patches
        if self.opt.patch_D:
            self.set_requires_grad(self.netD_P, True)  # enable backprop for D
            self.optimizer_D_P.zero_grad()  # set D's gradients to zero
            self.backward_D_P()  # calculate gradients for D
            self.optimizer_D_P.step()  # update D's weights
        
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        if self.opt.patch_D:
            self.set_requires_grad(self.netD_P, False)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
        
        # print(self.loss_G)
        # print(self.loss_D)
        # print('\n')


    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        netG_out_model_file = 'pixel2pixel_netG_epoch_{}_loss_{:.4f}.pth'.format(epoch, self.loss_G.detach().cpu().numpy())
        torch.save(self.netG_cpu.cpu().state_dict(), 
            os.path.join(self.save_dir, netG_out_model_file))
        netD_out_model_file = 'pixel2pixel_netD_epoch_{}_loss_{:.4f}.pth'.format(epoch, self.loss_D.detach().cpu().numpy())    
        torch.save(self.netD_cpu.cpu().state_dict(), 
            os.path.join(self.save_dir, netD_out_model_file))

        print('====> save model:\t{}'.format(netG_out_model_file))
        print('====> save model:\t{}'.format(netD_out_model_file))



if __name__ == '__main__':
    print('pixel to pixel model')