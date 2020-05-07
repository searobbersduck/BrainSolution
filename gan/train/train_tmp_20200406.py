import os
import sys
sys.path.append('../')
import numpy as np

from datasets.cta_to_dwi_dataset import CTA2DWI_GAN_DS
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
        self.display = 10
        self.save_interval = 10
        self.intermidiate_result_root = '../data/gan/cta2dwi/case_178_forHuang_rescale/train_result/intermidiate_result_{}'.format(__file__.split('.')[0])
        # add patch discriminator
        self.patch_D = False
        self.num_patches_D = 5
        self.patch_size_D = [64, 64, 64]
        # crop_size
        self.crop_size = [160, 224, 288]

def train():
    opt = Options()
    ds = CTA2DWI_GAN_DS('../data/gan/cta2dwi/case_178_forHuang_rescale', 
    '../data/gan/cta2dwi/case_178_forHuang_rescale/config/config_file_1.txt', 
    'train', opt.crop_size, opt.crop_size, debug=True)
    dataloader = DataLoader(ds, num_workers=opt.num_workers, batch_size=opt.batch_size, pin_memory=True, shuffle=True)

    gan_model = Pix2PixModel(opt)

    for epoch_i in range(opt.epochs):
        for index, (ct_imgs, dwi_imgs, ct_names, dwi_names) in enumerate(dataloader):
            input = {}
            input['A'] = ct_imgs
            input['B'] = dwi_imgs
            input['A_paths'] = 'A'
            input['B_paths'] = 'B'

            gan_model.set_input(input)
            gan_model.optimize_parameters()

            if index%opt.display == 0:
                print('====> epochs:[{}][{:4d}/{:4d}]\tgan loss:[{:.3f}]\tdiscriminator loss:[{:.3f}]'.format(
                    epoch_i, index, len(dataloader), gan_model.loss_G.detach().cpu().numpy(), 
                    gan_model.loss_D.detach().cpu().numpy()
                ))

            if (index%opt.display == 0 and epoch_i%opt.save_interval == 0) or (epoch_i != 0 and epoch_i%100 == 0):
                os.makedirs(opt.intermidiate_result_root, exist_ok=True)

                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(opt.intermidiate_result_root, 'epoch_{}_index_{}_ct_{}.nii.gz'.format(epoch_i, index, ct_names[0].split('.')[0])))
                writer.Execute(sitk.GetImageFromArray(ct_imgs.detach().cpu()[0][0].numpy()))

                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(opt.intermidiate_result_root, 'epoch_{}_index_{}_dwi_real_{}.nii.gz'.format(epoch_i, index, dwi_names[0].split('.')[0])))
                writer.Execute(sitk.GetImageFromArray(dwi_imgs.detach().cpu()[0][0].numpy()))
                
                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(opt.intermidiate_result_root, 'epoch_{}_index_{}_dwi_fake_{}.nii.gz'.format(epoch_i, index, dwi_names[0].split('.')[0])))
                writer.Execute(sitk.GetImageFromArray(gan_model.fake_B.detach().cpu()[0][0].numpy()))
            

if __name__ == '__main__':
    train()
