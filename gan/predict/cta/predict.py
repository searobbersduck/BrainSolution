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

import shutil
from glob import glob

import cv2

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
        self.crop_size = [32, 512, 512]
        # self.crop_size = [8, 8, 8]

        self.root_dir = '../../data/gan/hospital_4_2/experiment_registration2/8.out'
        self.config_file = '../../data/gan/hospital_4_2/experiment_registration2/8.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt'
        self.check_point = None
        # self.netG_model_path = '../../data/gan/hospital_4_2/experiment_registration2/9.model_out/model_train_ncct_to_dwi_bxxx_hospital4_2_nonmask_20200506/pixel2pixel_netG_epoch_50_loss_69.7196.pth'
        # self.netG_model_path = '../../data/gan/hospital_4/experiment_registration2/9.2.model_out/model_train_ncct_to_dwi_bxxx_hospital4_nonmask_20200508/pixel2pixel_netG_epoch_175_loss_11.4341.pth'
        # self.netG_model_path = '../../data/gan/hospital_4_2/experiment_registration2/9.2.model_out/model_train_ncct_to_dwi_bxxx_hospital4_2_nonmask_20200508/pixel2pixel_netG_epoch_400_loss_16.9547.pth' #ncct还可以
        # self.netG_model_path = '../../data/gan/hospital_6/experiment_registration2/9.2.model_out/model_train_cta_to_dwi_bxxx_hospital6_nonmask_20200508/pixel2pixel_netG_epoch_300_loss_7.5488.pth'
        self.netG_model_path = '../../data/gan/hospital_6/experiment_registration2/9.2.model_out/model_train_cta_to_dwi_bxxx_hospital6_nonmask_20200508/pixel2pixel_netG_epoch_950_loss_9.0383.pth'
        # self.netG_model_path = '../../data/gan/hospital_6/experiment_registration2/9.2.model_out/model_train_cta_to_dwi_bxxx_hospital6_nonmask_skip_20200520/pixel2pixel_netG_epoch_975_loss_7.7722.pth'
        self.netG_model_path = '../../data/gan/hospital_6_crop/experiment_registration2/9.2.model_out/model_train_cta_to_dwi_bxxx_hospital6_nonmask_skip_20200729/pixel2pixel_netG_epoch_5075_loss_3.4252.pth'
        self.netG_model_path = '../../data/gan/hospital_6_crop/experiment_registration2/9.2.model_out/model_train_cta_to_dwi_bxxx_hospital6_nonmask_skip_20200729/pixel2pixel_netG_epoch_3900_loss_4.9019.pth'
        # self.netG_model_path = '../../data/gan/hospital_6_crop/experiment_registration2/9.2.model_out/model_train_cta_to_dwi_bxxx_hospital6_nonmask_skip_lbp_loss_20200615/pixel2pixel_netG_epoch_2500_loss_17.1864.pth'
        # self.netD_model_path = '../../data/gan/ncct2dwi/experiment_registration2/9.model_out/model_train_ncct_to_dwi_bxxx_20200421/pixel2pixel_netD_epoch_100_loss_0.2630.pth'
        # self.netG_model_path = None
        self.netD_model_path = None

def predict(infile, outdir):
    opt = Options()
    # ds = NCCT_GAN_MASK_DS(opt.root_dir, 
    # opt.config_file, 
    # 'train', opt.crop_size, opt.crop_size, debug=False)
    
    predict_utils = NCCT_GAN_PREDICT_UTILS()
    crop_size = [32, 512, 512]
    image_tensors, d_cnt, h_cnt, w_cnt = predict_utils.get_image_tensors(infile, crop_size)

    # gan_model = Pix2PixModel(opt)

    netG_cpu = ResnetGenerator(1,1, 32, n_blocks=6)
    netG_cpu.load_state_dict(torch.load(opt.netG_model_path))

    # net_g = torch.nn.DataParallel(netG_cpu).cuda()
    net_g = netG_cpu.to(torch.device("cuda"))

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
    with torch.no_grad():
        for image_tensor in image_tensors:
            out = net_g(image_tensor.cuda())
            sub_arr = out.detach().cpu()[0][0].numpy()
            torch.cuda.empty_cache() 
            out_arr.append(sub_arr)

    dst_arr = predict_utils.compose_arrays_to_image(out_arr, [d_cnt, h_cnt, w_cnt], crop_size)
    
    os.makedirs(outdir, exist_ok=True)
    outname = os.path.join(outdir, os.path.basename(infile).replace('.nii.gz', '_fake.nii.gz'))
    sitk_img = sitk.GetImageFromArray(dst_arr)
    raw_img = sitk.ReadImage(infile)
    sitk_img.SetOrigin(raw_img.GetOrigin())
    sitk_img.SetDirection(raw_img.GetDirection())
    sitk_img.SetSpacing(raw_img.GetSpacing())
    sitk.WriteImage(sitk_img, outname)
    print('hello world')

def batch_predict_cta(root_dir, config_file, outdir):
    '''
    batch_predict_cta('../../data/gan/hospital_6/experiment_registration2/8.2.out', '../../data/gan/hospital_6/experiment_registration2/8.2.out/config/anno_ncct_to_dwi_bxxx_test_config_file.txt', '../../data/gan/hospital_6/experiment_registration2/10.predict')
    '''
    os.makedirs(outdir, exist_ok=True)
    ct_list = []
    gt_dwi_list = []
    with open(config_file) as f:
        for line in f.readlines():
            if line is None or len(line) == 0:
                continue
            ss = line.split('\t')
            ct_file = ss[0]
            ct_file = os.path.join(root_dir, ct_file)
            if not os.path.isfile(ct_file):
                continue
            gt_dwi_file = ss[1]
            gt_dwi_file = os.path.join(root_dir, gt_dwi_file)
            if not os.path.isfile(gt_dwi_file):
                continue
            ct_list.append(ct_file)
            gt_dwi_list.append(gt_dwi_file)
    
    for i in tqdm(range(len(ct_list))):
        ct_file = ct_list[i]
        gt_dwi_file = gt_dwi_list[i]
        predict(ct_file, outdir)
        dst_gt_dwi_file = os.path.join(outdir, os.path.basename(gt_dwi_file))
        dst_ct_file = os.path.join(outdir, os.path.basename(ct_file))
        shutil.copyfile(gt_dwi_file, dst_gt_dwi_file)
        shutil.copyfile(ct_file, dst_ct_file)

def save_ct_img(in_arr, out_file, ww=150, wl=50):
    min_v = wl-ww//2
    max_v = wl+ww//2
    out_arr = np.clip(in_arr, min_v, max_v)
    out_arr = (out_arr-min_v)/ww*255
    cv2.imwrite(out_file, out_arr)


def batch_convert_niigz_jpg(indir, outdir):
    os.makedirs(outdir, exist_ok=True)
    ct_pattern = '_first_BS_NCCT.nii.gz'
    fake_pattern = '_first_BS_NCCT_fake.nii.gz'
    real_pattern = '_first_FU_DWI_BXXX.nii.gz'
    fake_list = glob(os.path.join(indir, '*{}*'.format(fake_pattern)))
    patient_ids = [os.path.basename(i).split('_')[0] for i in fake_list]
    for patient_id in tqdm(patient_ids):
        ct_file = os.path.join(indir, '{}{}'.format(patient_id, ct_pattern))
        fake_file = os.path.join(indir, '{}{}'.format(patient_id, fake_pattern))
        real_file = os.path.join(indir, '{}{}'.format(patient_id, real_pattern))
        sub_outdir = os.path.join(outdir, patient_id)
        os.makedirs(sub_outdir, exist_ok=True)
        fake_img = sitk.ReadImage(fake_file)
        real_img = sitk.ReadImage(real_file)
        ct_img = sitk.ReadImage(ct_file)
        fake_arr = sitk.GetArrayFromImage(fake_img)
        real_arr = sitk.GetArrayFromImage(real_img)
        ct_arr = sitk.GetArrayFromImage(ct_img)
        z_len = real_arr.shape[0]
        for j in range(z_len):
            sub_fake_file = os.path.join(sub_outdir, '{}_{}_fake.jpg'.format(patient_id, j))
            sub_real_file = os.path.join(sub_outdir, '{}_{}_real.jpg'.format(patient_id, j))
            sub_ct_file = os.path.join(sub_outdir, '{}_{}_ct.jpg'.format(patient_id, j))

            # cv2.imwrite(sub_fake_file, fake_arr[j])
            # cv2.imwrite(sub_real_file, real_arr[j])
            # cv2.imwrite(sub_ct_file, ct_arr[j])
            save_ct_img(fake_arr[j], sub_fake_file, 400, 200)
            save_ct_img(real_arr[j], sub_real_file, 400, 200)
            save_ct_img(ct_arr[j], sub_ct_file)



if __name__ == '__main__':
    # predict('../../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct/114093_first_BS_NCCT.nii.gz', '../../data/gan/hospital_4/experiment_registration2/10.predict')
    # predict('../../data/gan/hospital_4/experiment_registration2/8.out/NCCT/448646_first_BS_NCCT.nii.gz', '../../data/gan/hospital_4/experiment_registration2/10.predict')
    # predict('../../data/gan/hospital_4_2/experiment_registration2/8.out/NCCT/402830_first_BS_NCCT.nii.gz', '../../data/gan/hospital_4/experiment_registration2/10.predict')
    # predict('../../data/gan/hospital_4_2/experiment_registration2/8.out/NCCT/441128_first_BS_NCCT.nii.gz', '../../data/gan/hospital_4/experiment_registration2/10.predict')
    # predict('../../data/gan/hospital_4/experiment_registration2/8.2.out/NCCT/439856_first_BS_NCCT.nii.gz', '../../data/gan/hospital_4/experiment_registration2/10.predict')
    # predict('../../data/gan/hospital_6/experiment_registration2/8.2.out/NCCT/4692992_first_BS_NCCT.nii.gz', '../../data/gan/hospital_6/experiment_registration2/10.predict')
    # batch_predict_cta('../../data/gan/hospital_6/experiment_registration2/8.2.out', '../../data/gan/hospital_6/experiment_registration2/8.2.out/config/anno_mask_ncct_to_dwi_bxxx_test_config_file.txt', '../../data/gan/hospital_6/experiment_registration2/10.xx_predict')
    # batch_convert_niigz_jpg('../../data/gan/hospital_6/experiment_registration2/10.xx_predict', '../../data/gan/hospital_6/experiment_registration2/10.xx_predict_jpg')

    # batch_predict_cta('../../data/gan/hospital_6/experiment_registration3/8.2.out', '../../data/gan/hospital_6/experiment_registration3/8.2.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt', '../../data/gan/hospital_6/experiment_registration3/10.predict')
    # batch_predict_cta('../../data/gan/hospital_6/experiment_registration3/8.2.out', '../../data/gan/hospital_6/experiment_registration3/8.2.out/config/mask_ncct_to_dwi_bxxx_test_config_file.txt', '../../data/gan/hospital_6/experiment_registration3/10.predict')
    # batch_convert_niigz_jpg('../../data/gan/hospital_6/experiment_registration3/10.predict', '../../data/gan/hospital_6/experiment_registration3/10.predict_jpg')
    
    batch_predict_cta('../../data/gan/hospital_6_crop/experiment_registration2/8.2.out', '../../data/gan/hospital_6_crop/experiment_registration2/8.2.out/config/anno_mask_ncct_to_dwi_bxxx_test_config_file.txt', '../../data/gan/hospital_6_crop/experiment_registration2/10.predict_4.9019')
    # batch_convert_niigz_jpg('../../data/gan/hospital_6/experiment_registration3/10.predict', '../../data/gan/hospital_6/experiment_registration3/10.predict_jpg')

    # batch_predict_cta('../../data/gan/hospital_6/experiment_registration_neg/8.2.out', '../../data/gan/hospital_6/experiment_registration_neg/8.2.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt', '../../data/gan/hospital_6/experiment_registration_neg/10.predict')
    # batch_predict_cta('../../data/gan/hospital_6/experiment_registration_neg/8.2.out', '../../data/gan/hospital_6/experiment_registration_neg/8.2.out/config/mask_ncct_to_dwi_bxxx_test_config_file.txt', '../../data/gan/hospital_6/experiment_registration_neg/10.predict')
    # batch_convert_niigz_jpg('../../data/gan/hospital_6/experiment_registration_neg/10.predict', '../../data/gan/hospital_6/experiment_registration_neg/10.predict_jpg')



