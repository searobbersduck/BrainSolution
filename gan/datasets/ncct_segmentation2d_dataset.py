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
    mask_data[mask_data>0.05] = 1
    mask_data[mask_data<=0.05] = 0
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


def ncct_convert_3d_2d(indir, outdir, rapid_config_file, image_postfix='_BS_NCCT.nii.gz', mask_postfix='_FU_DWI_INFARCT_MASK.nii.gz', train_ratio=0.99, val_ratio=0.01):
    '''
    indir = '../data/ncct/brain-NCCT-with-mask'
    outdir = '../data/ncct/slice_2d'

    cmd: python cerebral_parenchyma_extract_dataset.py ncct_convert_3d_2d '../data/gan/hospital_4_2/experiment_registration3/5 dwi_rigid_align_ncct' '../data/gan/hospital_4_2/experiment_seg_2d/infarct' '../data/gan/hospital_4_2/experiment_registration3/1.rapid/config.txt'
    debug cmd: ncct_convert_3d_2d('../data/gan/hospital_4_2/experiment_registration3/5 dwi_rigid_align_ncct', '../data/gan/hospital_4_2/experiment_seg_2d/infarct', '../data/gan/hospital_4_2/experiment_registration3/1.rapid/config.txt')
    '''

    os.makedirs(outdir, exist_ok=True)
    image_pattern = '*{}'.format(image_postfix)
    mask_pattern = '*{}'.format(mask_postfix)

    infarct_pids = []
    penumbra_pids = []
    positive_pids = []
    with open(rapid_config_file) as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            ss = line.split('\t')
            if ss[1] == 'True':
                infarct_pids.append(ss[0])
            if ss[2] == 'True':
                penumbra_pids.append(ss[0])
            if ss[1] == 'True' or ss[2] == 'True':
                positive_pids.append(ss[0])

    image_files = glob(os.path.join(indir, image_pattern))

    image_files = [i for i in image_files if os.path.basename(i).split('_')[0] in infarct_pids]
    
    train_pos = int(len(image_files) * train_ratio)
    val_pos = int(len(image_files) * (train_ratio+val_ratio))

    train_image_files = image_files[:train_pos]
    val_image_files = image_files[train_pos:val_pos]
    test_image_files = image_files[val_pos:]

    for image_file in tqdm(train_image_files):
        mask_file = image_file.replace(image_postfix, mask_postfix)
        if not os.path.isfile(image_file):
            continue
        if not os.path.isfile(mask_file):
            continue
        sub_out_dir = os.path.join(outdir, 'train')
        ncct_convert_3d_to_2d_single(image_file, mask_file, sub_out_dir)

    for image_file in tqdm(val_image_files):
        mask_file = image_file.replace(image_postfix, mask_postfix)
        if not os.path.isfile(image_file):
            continue
        if not os.path.isfile(mask_file):
            continue
        sub_out_dir = os.path.join(outdir, 'val')
        ncct_convert_3d_to_2d_single(image_file, mask_file, sub_out_dir)

    for image_file in tqdm(test_image_files):
        mask_file = image_file.replace(image_postfix, mask_postfix)
        if not os.path.isfile(image_file):
            continue
        if not os.path.isfile(mask_file):
            continue
        sub_out_dir = os.path.join(outdir, 'test')
        ncct_convert_3d_to_2d_single(image_file, mask_file, sub_out_dir)


def generate_config_file(in_2d_dir, out_config_dir, postfix=''):
    '''
    debug cmd: generate_config_file('../data/gan/hospital_4_2/experiment_seg_2d/infarct/train', '../data/gan/hospital_4_2/experiment_seg_2d/infarct/config', 'train')

    cmd: python cerebral_parenchyma_extract_dataset.py generate_config_file '../data/gan/hospital_4_2/experiment_seg_2d/infarct/train' '../data/gan/hospital_4_2/experiment_seg_2d/infarct/config' train

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


if __name__ == '__main__':
    fire.Fire()
    # ncct_convert_3d_2d('../data/gan/hospital_4_2/experiment_registration3/5 dwi_rigid_align_ncct', '../data/gan/hospital_4_2/experiment_seg_2d/infarct', '../data/gan/hospital_4_2/experiment_registration3/1.rapid/config.txt')
    # generate_config_file('../data/gan/hospital_4_2/experiment_seg_2d/infarct/train', '../data/gan/hospital_4_2/experiment_seg_2d/infarct/config', 'train')
    # generate_config_file('../data/gan/hospital_4_2/experiment_seg_2d/infarct/val', '../data/gan/hospital_4_2/experiment_seg_2d/infarct/config', 'val')