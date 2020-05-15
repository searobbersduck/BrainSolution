import os
import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np

from datasets.ncct_gan_dataset import NCCT_GAN_MASK_DS, NCCT_GAN_DS
from torch.utils.data import DataLoader, Dataset

from models.pixel2pixel_3d_model import Pix2PixModel

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
        self.model_save_interval = 25
        self.intermidiate_result_root = '../../data/gan/hospital_6/experiment_registration2/8.2.out/train_result/intermidiate_result_{}'.format(__file__.split('.')[0])
        self.save_dir = '../../data/gan/hospital_6/experiment_registration2/9.2.model_out/model_{}'.format(__file__.split('.')[0])
        # add patch discriminator
        self.patch_D = False
        self.num_patches_D = 5
        self.patch_size_D = [64, 64, 64]
        # crop_size
        self.crop_size = [32, 416, 416]
        # self.crop_size = [8, 8, 8]

        self.root_dir = '../../data/gan/hospital_6/experiment_registration2/8.2.out'
        self.config_file = '../../data/gan/hospital_6/experiment_registration2/8.2.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt'
        self.check_point = None
        # self.netG_model_path = '../../data/gan/hospital_6/experiment_registration2/9.2.model_out/model_train_cta_to_dwi_bxxx_hospital6_nonmask_20200514/pixel2pixel_netG_epoch_25_loss_55.2654.pth'
        # self.netD_model_path = '../../data/gan/ncct2dwi/experiment_registration2/9.model_out/model_train_ncct_to_dwi_bxxx_20200421/pixel2pixel_netD_epoch_100_loss_0.2630.pth'
        self.netG_model_path = None
        self.netD_model_path = None

def train():
    opt = Options()
    ds = NCCT_GAN_MASK_DS(opt.root_dir, 
    opt.config_file, 
    'train', opt.crop_size, opt.crop_size, debug=False)
    dataloader = DataLoader(ds, num_workers=opt.num_workers, batch_size=opt.batch_size, pin_memory=True, shuffle=True)

    gan_model = Pix2PixModel(opt)

    for epoch_i in range(opt.epochs):
        for index, (src_imgs, dst_imgs, mask_imgs, src_names, dst_names) in enumerate(dataloader):
            input = {}
            input['A'] = src_imgs
            input['B'] = dst_imgs
            # input['mask'] = mask_imgs
            input['A_paths'] = 'A'
            input['B_paths'] = 'B'


            gan_model.set_input(input)
            gan_model.optimize_parameters()

            if index%opt.display == 0:
                print('====> epochs:[{}][{:4d}/{:4d}]\tgan loss:[{:.3f}]\tdiscriminator loss:[{:.3f}]\ttarget files:[{}]'.format(
                    epoch_i, index, len(dataloader), gan_model.loss_G.detach().cpu().numpy(), 
                    gan_model.loss_D.detach().cpu().numpy(), dst_names
                ))

            if (index%opt.display == 0 and epoch_i%opt.save_interval == 0) or (epoch_i != 0 and epoch_i%100 == 0):
                os.makedirs(opt.intermidiate_result_root, exist_ok=True)

                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(opt.intermidiate_result_root, 'epoch_{}_index_{}_src_{}.nii.gz'.format(epoch_i, index, src_names[0].split('.')[0])))
                writer.Execute(sitk.GetImageFromArray(src_imgs.detach().cpu()[0][0].numpy()))

                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(opt.intermidiate_result_root, 'epoch_{}_index_{}_dst_real_{}.nii.gz'.format(epoch_i, index, dst_names[0].split('.')[0])))
                writer.Execute(sitk.GetImageFromArray(dst_imgs.detach().cpu()[0][0].numpy()))
                
                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(opt.intermidiate_result_root, 'epoch_{}_index_{}_dst_fake_{}.nii.gz'.format(epoch_i, index, dst_names[0].split('.')[0])))
                writer.Execute(sitk.GetImageFromArray(gan_model.fake_B.detach().cpu()[0][0].numpy()))
            
        if (epoch_i%opt.model_save_interval == 0):
            gan_model.save_networks(epoch_i)

if __name__ == '__main__':
    train()
