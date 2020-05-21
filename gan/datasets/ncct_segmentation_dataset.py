import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))

from common.utils.crop_utils import CropUtils, CroppedBoundary

from glob import glob
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

global_ncct_error_list = ['398774', '448646', '458192']

class NCCTDWISegmentationDS(Dataset):
    def __init__(self, root_dir, config_file, phase, corp_size, flag_index, mask_pattern):
        super().__init__()
        self.root_dir = root_dir
        self.phase = phase
        self.config_file = config_file
        self.crop_size = corp_size
        self.ncct_pattern = '_first_BS_NCCT.nii.gz'
        self.mask_pattern = mask_pattern
        self.flag_index = flag_index
        # element in list is patient id
        self.pos_list = []
        self.neg_list = []
        self.pos_images_list = []
        self.pos_masks_list = []
        self.neg_images_list = []
        self.neg_masks_list = []
        with open(self.config_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                if len(ss) != 3:
                    continue
                pid = ss[0]
                if pid in global_ncct_error_list:
                    continue
                image_file = os.path.join(self.root_dir, '{}{}'.format(pid, self.ncct_pattern))
                mask_file = os.path.join(self.root_dir, '{}{}'.format(pid, self.mask_pattern))
                if not os.path.isfile(image_file):
                    continue
                if not os.path.isfile(mask_file):
                    continue
                if ss[self.flag_index] == 'True':
                    self.pos_images_list.append(image_file)
                    self.pos_masks_list.append(mask_file)
                else:
                    self.neg_images_list.append(image_file)
                    self.neg_masks_list.append(mask_file)  

    def __len__(self):
        return len(self.pos_images_list)

    def __getitem__(self, idx):
        if self.phase == 'train':
            src_file = self.pos_images_list[idx]
            mask_file = self.neg_masks_list[idx]
            src_img = sitk.ReadImage(src_file)
            src_data = sitk.GetArrayFromImage(src_img)
            mask_img = sitk.ReadImage(mask_file)
            mask_data = sitk.GetArrayFromImage(mask_img)
            [d,w,h] = src_data.shape
            cropped_boundary = CroppedBoundary(0,d-1,0,h-1,0,w-1)
            Z_min, Z_max, Y_min, Y_max, X_min, X_max = CropUtils.get_region_3d_random_crop(self.crop_size, cropped_boundary)
            cropped_src = src_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]
            cropped_mask = mask_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]
            
            cropped_src = torch.from_numpy(cropped_src).float()
            cropped_src = torch.unsqueeze(cropped_src, axis=0)

            return cropped_src, cropped_mask, os.path.basename(src_file), os.path.basename(mask_file)



class NCCTCoreInfarctSegmentation(NCCTDWISegmentationDS):
    
    def __init__(self, root_dir, config_file, phase, corp_size):
        mask_pattern = '_first_FU_DWI_INFARCT_MASK.nii.gz'
        flag_index = 1
        super().__init__(root_dir, config_file, phase, corp_size, flag_index, mask_pattern)




def test_NCCTDWISegmentationDS():
    '''
    config file format as follows:
    pid     是否包含核心梗死区  是否包含缺血半暗带
    456926  False   False
    392112  False   False
    270200  False   False
    124639  False   False
    450950  False   False
    447276  False   False
    452468  False   False
    316872  False   False
    446273  False   True
    '''

    class Options():
        def __init__(self):
            self.lr = 2e-4
            self.beta1 = 0.5
            self.gan_mode = 'lsgan'
            self.direction = 'AtoB'
            self.lambda_L1 = 2
            self.epochs = 1000
            self.num_workers = 2
            self.batch_size = 2
            self.pin_memory = True
            self.display = 2
            self.save_interval = 10
            self.intermidiate_result_root = '../data/gan/hospital_6/experiment_registration2/8.1.out/train_result/intermidiate_result_{}'.format(__file__.split('.')[0])
            # add patch discriminator
            self.patch_D = False
            self.num_patches_D = 5
            self.patch_size_D = [64, 64, 64]
            # crop_size
            self.crop_size = [32, 448, 448]

            self.root_dir = '../data/gan/hospital_6/experiment_registration2/8.1.out'
            self.config_file = '../data/gan/hospital_6/experiment_registration2/8.1.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt'
            self.check_point = None
    opt = Options()
    root_dir = '../data/gan/hospital_4_2/experiment_registration3/5 dwi_rigid_align_ncct'
    config_file = '../data/gan/hospital_4_2/1.rapid/config.txt'
    phase = 'train'
    crop_size = [64,64,64]
    ds = NCCTCoreInfarctSegmentation(root_dir, config_file, phase, crop_size)
    data_loader = DataLoader(ds, num_workers=opt.num_workers, batch_size=opt.batch_size, pin_memory=True, shuffle=True)
    for i, (srcs, masks, _, _) in tqdm(enumerate(data_loader)):
        print(srcs.shape)


if __name__ == '__main__':
    test_NCCTDWISegmentationDS()
