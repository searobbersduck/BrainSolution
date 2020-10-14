import os
import sys

from glob import glob
from tqdm import tqdm
import numpy as np

import SimpleITK as sitk

from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
import time
import torch
import torch.nn as nn

import fire

class CTA_GAN_DS(Dataset):
    def __init__(self, data_root, config_file, phase, crop_size, scale_size):
        self.data_root = data_root
        self.config_file = config_file
        self.phase = phase
        self.crop_size = crop_size
        self.scale_size = scale_size

        self.files_a = []
        self.files_b = []
        with open(config_file) as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                file_a = os.path.join(data_root, ss[0])
                file_b = os.path.join(data_root, ss[1])
                self.files_a.append(file_a)
                self.files_b.append(file_b)

    def __len__(self):
        return len(self.files_a)

    
    def __getitem__(self, item):
        file_a = self.files_a[item]
        file_b = self.files_b[item]

        data_a = np.load(file_a)
        data_b = np.load(file_b)

        tensor_a = torch.from_numpy(data_a).float()
        tensor_a = torch.unsqueeze(tensor_a, axis=0)

        tensor_b = torch.from_numpy(data_b).float()
        tensor_b = torch.unsqueeze(tensor_b, axis=0)

        return tensor_a, tensor_b, file_a, file_b


def test_CTA_GAN_DS():
    data_root = '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/train'
    config_file = '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/config/cta_to_dwi_2d_train.txt'
    crop_size = [512, 512]
    ds = CTA_GAN_DS(data_root, config_file, 'train', crop_size, crop_size)
    data_loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=1, pin_memory=False)
    for i, (image_a, image_b, _, _) in tqdm(enumerate(data_loader)):
        print(image_a.shape)
        # break



if __name__ == '__main__':
    test_CTA_GAN_DS()