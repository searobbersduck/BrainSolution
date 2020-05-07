import os
import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np

from datasets.ncct_gan_dataset import NCCT_GAN_MASK_DS, NCCT_GAN_PREDICT_UTILS
from torch.utils.data import DataLoader, Dataset

from models.pixel2pixel_3d_model import Pix2PixModel, ResnetGenerator

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

from tqdm import tqdm

import SimpleITK as sitk
import time

class Options():
    def __init__(self):
        self.lr = 2e-4
        self.beta1 = 0.5
        self.gan_mode = 'lsgan'
        self.direction = 'AtoB'
        self.lambda_L1 = 2
        self.epochs = 1000
        self.num_workers = 8
        self.batch_size = 2
        self.pin_memory = True
        self.display = 2
        self.save_interval = 10
        self.model_save_interval = 50
        self.intermidiate_result_root = '../../data/gan/hospital_4_2/experiment_registration2/8.out/train_result/intermidiate_result_{}'.format(__file__.split('.')[0])
        self.save_dir = '../../data/gan/hospital_4_2/experiment_registration2/9.model_out/model_{}'.format(__file__.split('.')[0])
        # add patch discriminator
        self.patch_D = False
        self.num_patches_D = 5
        self.patch_size_D = [64, 64, 64]
        # crop_size
        self.crop_size = [128, 224, 224]
        # self.crop_size = [8, 8, 8]

        self.root_dir = '../../data/gan/hospital_4_2/experiment_registration2/8.out'
        self.config_file = '../../data/gan/hospital_4_2/experiment_registration2/8.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt'
        self.check_point = None
        self.netG_model_path = '../../data/gan/hospital_4/experiment_registration2/9.model_out/model_train_ncct_to_dwi_bxxx_hospital4_1_nonmask_20200506/pixel2pixel_netG_epoch_175_loss_207.5621.pth'
        # self.netD_model_path = '../../data/gan/ncct2dwi/experiment_registration2/9.model_out/model_train_ncct_to_dwi_bxxx_20200421/pixel2pixel_netD_epoch_100_loss_0.2630.pth'
        # self.netG_model_path = None
        self.netD_model_path = None

def predict(infile, outdir):
    opt = Options()
    # ds = NCCT_GAN_MASK_DS(opt.root_dir, 
    # opt.config_file, 
    # 'train', opt.crop_size, opt.crop_size, debug=False)
    
    predict_utils = NCCT_GAN_PREDICT_UTILS()
    crop_size = [128, 224, 224]
    image_tensors, d_cnt, h_cnt, w_cnt = predict_utils.get_image_tensors(infile, crop_size)

    # gan_model = Pix2PixModel(opt)

    netG_cpu = ResnetGenerator(1,1, 32, n_blocks=6)
    netG_cpu.load_state_dict(torch.load(opt.netG_model_path))

    net_g = torch.nn.DataParallel(netG_cpu).cuda()

    def set_requires_grad(nets, requires_grad=False):
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
    
    set_requires_grad(net_g)
    # net_g.eval()

    out_arr = []
    for image_tensor in image_tensors:
        out = net_g(image_tensor.cuda())
        sub_arr = out.detach().cpu()[0][0].numpy()
        out_arr.append(sub_arr)

    dst_arr = predict_utils.compose_arrays_to_image(out_arr, [d_cnt, h_cnt, w_cnt], crop_size)
    
    os.makedirs(outdir, exist_ok=True)
    outname = os.path.join(outdir, os.path.basename(infile))
    sitk_img = sitk.GetImageFromArray(dst_arr)
    sitk.WriteImage(sitk_img, outname)
    print('hello world')

if __name__ == '__main__':
    predict('../../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct/114093_first_BS_NCCT.nii.gz', '../../data/gan/hospital_4/experiment_registration2/10.predict')
