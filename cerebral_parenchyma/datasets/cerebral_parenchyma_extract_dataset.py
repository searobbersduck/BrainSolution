import os
import sys

from glob import glob
from tqdm import tqdm
import numpy as np

import SimpleITK as sitk

from torch.utils.data import Dataset, DataLoader
import nibabel
from scipy import ndimage
import time
import torch
import torch.nn as nn

import fire

def ncct_convert_3d_to_2d_single(image_file, mask_file, outdir):
    os.makedirs(outdir, exist_ok=True)
    image = sitk.ReadImage(image_file)
    mask = sitk.ReadImage(mask_file)
    image_data = sitk.GetArrayFromImage(image)
    mask_data = sitk.GetArrayFromImage(mask)
    assert image_data.shape[0] == mask_data.shape[0]
    prefix = os.path.basename(image_file).replace('_CT.nii.gz', '')
    for z in range(image_data.shape[0]):
        if np.sum(mask_data[z]) > 10:
            out_image_file = os.path.join(outdir, '{}_image_{}_xxx.npy'.format(prefix, z))
            out_mask_file = os.path.join(outdir, '{}_mask_{}_xxx.npy'.format(prefix, z))
        else:
            out_image_file = os.path.join(outdir, '{}_image_{}_yyy.npy'.format(prefix, z))
            out_mask_file = os.path.join(outdir, '{}_mask_{}_yyy.npy'.format(prefix, z))
        np.save(out_image_file, image_data[z])
        np.save(out_mask_file, mask_data[z]) 
    

def ncct_convert_3d_2d(indir, outdir, train_ratio=0.7, val_ratio=0.1):
    '''
    indir = '../data/ncct/brain-NCCT-with-mask'
    outdir = '../data/ncct/slice_2d'

    cmd: python cerebral_parenchyma_extract_dataset.py ncct_convert_3d_2d ../data/ncct/brain-NCCT-with-mask ../data/ncct/slice_2d
    debug cmd: ncct_convert_3d_2d('../data/ncct/brain-NCCT-with-mask', '../data/ncct/slice_2d')
    '''

    os.makedirs(outdir, exist_ok=True)
    image_pattern = '*_CT.nii.gz'
    mask_pattern = '*_brain_mask.nii.gz'

    image_files = glob(os.path.join(indir, image_pattern))
    
    train_pos = int(len(image_files) * train_ratio)
    val_pos = int(len(image_files) * (train_ratio+val_ratio))

    train_image_files = image_files[:train_pos]
    val_image_files = image_files[train_pos:val_pos]
    test_image_files = image_files[val_pos:]

    for image_file in tqdm(train_image_files):
        mask_file = image_file.replace('_CT.nii.gz', '_brain_mask.nii.gz')
        if not os.path.isfile(image_file):
            continue
        if not os.path.isfile(mask_file):
            continue
        sub_out_dir = os.path.join(outdir, 'train')
        ncct_convert_3d_to_2d_single(image_file, mask_file, sub_out_dir)

    for image_file in tqdm(val_image_files):
        mask_file = image_file.replace('_CT.nii.gz', '_brain_mask.nii.gz')
        if not os.path.isfile(image_file):
            continue
        if not os.path.isfile(mask_file):
            continue
        sub_out_dir = os.path.join(outdir, 'val')
        ncct_convert_3d_to_2d_single(image_file, mask_file, sub_out_dir)

    for image_file in tqdm(test_image_files):
        mask_file = image_file.replace('_CT.nii.gz', '_brain_mask.nii.gz')
        if not os.path.isfile(image_file):
            continue
        if not os.path.isfile(mask_file):
            continue
        sub_out_dir = os.path.join(outdir, 'test')
        ncct_convert_3d_to_2d_single(image_file, mask_file, sub_out_dir)



def cta_convert_3d_to_2d_single(image_file, mask_file, outdir):
    os.makedirs(outdir, exist_ok=True)
    reader = sitk.ImageSeriesReader()
    dicomfilenames = reader.GetGDCMSeriesFileNames(image_file)
    reader.SetFileNames(dicomfilenames)
    image = reader.Execute()
    mask = sitk.ReadImage(mask_file)
    image_data = sitk.GetArrayFromImage(image)
    mask_data = sitk.GetArrayFromImage(mask)
    assert image_data.shape[0] == mask_data.shape[0]
    prefix = os.path.basename(image_file)
    for z in range(image_data.shape[0]):
        if np.sum(mask_data[z]) > 10:
            out_image_file = os.path.join(outdir, '{}_image_{}_xxx.npy'.format(prefix, z))
            out_mask_file = os.path.join(outdir, '{}_mask_{}_xxx.npy'.format(prefix, z))
        else:
            out_image_file = os.path.join(outdir, '{}_image_{}_yyy.npy'.format(prefix, z))
            out_mask_file = os.path.join(outdir, '{}_mask_{}_yyy.npy'.format(prefix, z))
        np.save(out_image_file, image_data[z])
        np.save(out_mask_file, mask_data[z]) 

def cta_convert_3d_2d(indir, outdir, train_ratio=0.7, val_ratio=0.1):
    '''
    indir = '../data/cta'
    outdir = '../data/cta/slice_2d'

    cmd: python cerebral_parenchyma_extract_dataset.py cta_convert_3d_2d ../data/cta ../data/cta/slice_2d
    debug cmd: cta_convert_3d_2d('../data/cta', '../data/cta/slice_2d')


    indir结构
    tree -L 2
    ├── image
    │   └── dicom
    │       ├── 1.3.12.2.1107.5.1.4.60320.30000011010400322303100012211
    │       ├── 1.3.12.2.1107.5.1.4.60320.30000011091900223568700002300
    └── mask
        └── Ori_nii
        ├── 1.3.12.2.1107.5.1.4.60320.30000011010400322303100012211.nii.gz
        ├── 1.3.12.2.1107.5.1.4.60320.30000011091900223568700002300.nii.gz

    '''
    image_dir = os.path.join(indir, 'image', 'dicom')
    mask_dir = os.path.join(indir, 'mask', 'Ori_nii')

    image_files = [os.path.join(image_dir, i) for i in os.listdir(image_dir)]


    train_pos = int(len(image_files) * train_ratio)
    val_pos = int(len(image_files) * (train_ratio+val_ratio))

    train_image_files = image_files[:train_pos]
    val_image_files = image_files[train_pos:val_pos]
    test_image_files = image_files[val_pos:]

    for image_file in tqdm(train_image_files):
        if not os.path.isdir(image_file):
            continue
        mask_file = os.path.join(mask_dir, '{}.nii.gz'.format(os.path.basename(image_file)))
        if not os.path.isfile(mask_file):
            continue
        sub_out_dir = os.path.join(outdir, 'train')
        try:
            cta_convert_3d_to_2d_single(image_file, mask_file, sub_out_dir)
        except:
            print('error file:\t{}'.format(image_file))

    for image_file in tqdm(val_image_files):
        if not os.path.isdir(image_file):
            continue
        mask_file = os.path.join(mask_dir, '{}.nii.gz'.format(os.path.basename(image_file)))
        if not os.path.isfile(mask_file):
            continue
        sub_out_dir = os.path.join(outdir, 'val')
        try:
            cta_convert_3d_to_2d_single(image_file, mask_file, sub_out_dir)
        except:
            print('error file:\t{}'.format(image_file))

    for image_file in tqdm(test_image_files):
        if not os.path.isdir(image_file):
            continue
        mask_file = os.path.join(mask_dir, '{}.nii.gz'.format(os.path.basename(image_file)))
        if not os.path.isfile(mask_file):
            continue
        sub_out_dir = os.path.join(outdir, 'test')
        try:
            cta_convert_3d_to_2d_single(image_file, mask_file, sub_out_dir)
        except:
            print('error file:\t{}'.format(image_file))

def generate_config_file(in_2d_dir, out_config_dir, postfix=''):
    '''
    debug cmd: generate_config_file('../data/cta/slice_2d/train', '../data/cta/config', 'train')

    cmd: python cerebral_parenchyma_extract_dataset.py generate_config_file ../data/cta/slice_2d/train ../data/cta/config train
    cmd: python cerebral_parenchyma_extract_dataset.py generate_config_file ../data/cta/slice_2d/val ../data/cta/config val
    cmd: python cerebral_parenchyma_extract_dataset.py generate_config_file ../data/cta/slice_2d/test ../data/cta/config test
    cmd: python cerebral_parenchyma_extract_dataset.py generate_config_file ../data/ncct/slice_2d/train ../data/ncct/config train
    cmd: python cerebral_parenchyma_extract_dataset.py generate_config_file ../data/ncct/slice_2d/val ../data/ncct/config val
    cmd: python cerebral_parenchyma_extract_dataset.py generate_config_file ../data/ncct/slice_2d/test ../data/ncct/config test
    '''
    os.makedirs(out_config_dir, exist_ok=True)
    xxx_pairs = []
    xxx_image_list = glob(os.path.join(in_2d_dir, '*_image_*_xxx.npy'))
    for image_file in xxx_image_list:
        mask_file = image_file.replace('_image_', '_mask_')
        if not os.path.isfile(image_file):
            continue
        if not os.path.isfile(mask_file):
            continue
        xxx_pairs.append('{}\t{}'.format(os.path.basename(image_file), os.path.basename(mask_file)))
    
    yyy_pairs = []
    yyy_image_list = glob(os.path.join(in_2d_dir, '*_image_*_yyy.npy'))
    for image_file in yyy_image_list:
        mask_file = image_file.replace('_image_', '_mask_')
        if not os.path.isfile(image_file):
            continue
        if not os.path.isfile(mask_file):
            continue
        yyy_pairs.append('{}\t{}'.format(os.path.basename(image_file), os.path.basename(mask_file)))
    with open(os.path.join(out_config_dir, 'config_2d_cerebral_parenchyma_xxx_{}.txt'.format(postfix)), 'w') as f:
        f.write('\n'.join(xxx_pairs))

    with open(os.path.join(out_config_dir, 'config_2d_cerebral_parenchyma_yyy_{}.txt'.format(postfix)), 'w') as f:
        f.write('\n'.join(yyy_pairs))

def test_generate_config_file():
    generate_config_file('../data/cta/slice_2d/train', '../data/cta/config', 'train')
    generate_config_file('../data/cta/slice_2d/val', '../data/cta/config', 'val')
    generate_config_file('../data/cta/slice_2d/test', '../data/cta/config', 'test')
    generate_config_file('../data/ncct/slice_2d/train', '../data/ncct/config', 'train')
    generate_config_file('../data/ncct/slice_2d/val', '../data/ncct/config', 'val')
    generate_config_file('../data/ncct/slice_2d/test', '../data/ncct/config', 'test')

class CerebralParenchymaSegmentDS(Dataset):
    def __init__(self, root_dirs, config_xxx_files, config_yyy_files, phase,crop_size, scale_size):
        self.root_dirs = root_dirs
        self.config_xxx_files = config_xxx_files
        self.config_yyy_files = config_yyy_files
        self.phase = phase
        self.crop_size = crop_size
        self.scale_size = scale_size

        self.xxx_info_list = []
        self.yyy_info_list = []

        self.xxx_images_list = []
        self.xxx_masks_list = []

        for i in range(len(config_xxx_files)):
            with open(config_xxx_files[i]) as f:
                for line in f.readlines():
                    line = line.strip()
                    if line is None or len(line) == 0:
                        continue
                    ss = line.split('\t')
                    if len(ss) != 2:
                        continue
                    self.xxx_images_list.append(os.path.join(root_dirs[i], ss[0]))
                    self.xxx_masks_list.append(os.path.join(root_dirs[i], ss[1]))

        assert len(self.xxx_images_list) == len(self.xxx_masks_list)
        self.xxx_images_list = self.xxx_images_list

        self.yyy_images_list = []
        self.yyy_masks_list = []

        for i in range(len(config_yyy_files)):
            with open(config_yyy_files[i]) as f:
                for line in f.readlines():
                    line = line.strip()
                    if line is None or len(line) == 0:
                        continue
                    ss = line.split('\t')
                    if len(ss) != 2:
                        continue
                    self.yyy_images_list.append(os.path.join(root_dirs[i], ss[0]))
                    self.yyy_masks_list.append(os.path.join(root_dirs[i], ss[1]))

        assert len(self.yyy_images_list) == len(self.yyy_masks_list)

    def __len__(self):
        return len(self.xxx_images_list)


    def __getitem__(self, idx):
        if self.phase == 'train':
            if np.random.rand() < 0.8:
                image_path = self.xxx_images_list[idx]
                mask_path = self.xxx_masks_list[idx]
            else:
                rand_idx = np.random.randint(0, len(self.yyy_images_list))
                image_path = self.yyy_images_list[rand_idx]
                mask_path = self.yyy_masks_list[rand_idx]
            
            image_data = np.load(image_path)
            mask_data = np.load(mask_path)

            image_tensor = torch.from_numpy(image_data).float()
            image_tensor = torch.unsqueeze(image_tensor, axis=0)

            mask_data = np.array(mask_data, dtype=np.int32)
            mask_tensor = torch.from_numpy(mask_data).float()
            mask_tensor = torch.unsqueeze(mask_tensor, axis=0)

            return image_tensor, mask_tensor, image_path, mask_path
        

def test_CerebralParenchymaSegmentDS():
    root_dirs = ['../data/ncct/slice_2d/train']
    config_xxx_files = ['../data/ncct/config/config_2d_cerebral_parenchyma_xxx_train.txt']
    config_yyy_files = ['../data/ncct/config/config_2d_cerebral_parenchyma_yyy_train.txt']
    crop_size = [512, 512]
    ds = CerebralParenchymaSegmentDS(root_dirs, config_xxx_files, config_yyy_files, 'train', crop_size, crop_size)
    # data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=pin_memory)
    data_loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=1, pin_memory=False)
    for i, (images, masks, _, _) in tqdm(enumerate(data_loader)):
        print(images.shape)
        print('hello world')
        break



if __name__ == '__main__':
    # fire.Fire()
    # ncct_convert_3d_2d('../data/ncct/brain-NCCT-with-mask', '../data/ncct/slice_2d')
    # generate_config_file('../data/ncct/slice_2d/train', '../data/ncct/config', 'train')
    test_CerebralParenchymaSegmentDS()