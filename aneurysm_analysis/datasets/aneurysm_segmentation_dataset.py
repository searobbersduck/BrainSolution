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

import numpy as np
import pydicom


class AneurysmSegmentationDS(Dataset):
    def __init__(self, root_dir, config_file, phase, corp_size):
        self.root_dir = root_dir
        self.phase = phase
        self.config_file = config_file
        self.crop_size = corp_size
        self.images = []
        self.masks = []
        self.images_list = []
        self.masks_list = []
        self.center_points_list = []
        with open(config_file) as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                series_uid = ss[0]
                points = ss[1:]
                points_num = len(points)//3
                center_points = []
                for i in range(points_num):
                    center_points.append([int(points[3*i]), int(points[3*i+1]), int(points[3*i+2])])
                self.center_points_list.append(center_points)
                self.images_list.append(os.path.join(root_dir, '{}_image.nii.gz'.format(series_uid)))
                self.masks_list.append(os.path.join(root_dir, '{}_mask.nii.gz'.format(series_uid)))

        # self.images_list = self.images_list[:1]
        # self.masks_list = self.masks_list[:1]

        for image_file in self.images_list:
            sitk_image = sitk.ReadImage(image_file)
            image_arr = sitk.GetArrayFromImage(sitk_image)
            self.images.append(image_arr)

        for mask_file in self.masks_list:
            sitk_mask = sitk.ReadImage(mask_file)
            mask_arr = sitk.GetArrayFromImage(sitk_mask)
            self.masks.append(mask_arr)

        
        


    def __random_crop_data(self, volume, mask, size):
        [img_d, img_h, img_w] = volume.shape
        [input_d, input_h, input_w] = size
        z_min_upper = img_d - input_d
        y_min_upper = img_h - input_h
        x_min_upper = img_w - input_w
        Z_min = np.random.randint(0, z_min_upper)
        Y_min = np.random.randint(0, y_min_upper)
        X_min = np.random.randint(0, x_min_upper)

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w
        return volume[Z_min: Z_max, Y_min: Y_max, X_min: X_max], mask[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __random_center_crop_data(self, volume, mask, size, crop_center):
        # [img_d, img_h, img_w] = volume.shape
        [img_d, img_h, img_w] = crop_center
        [input_d, input_h, input_w] = size
        z_min_upper = img_d - input_d
        y_min_upper = img_h - input_h
        x_min_upper = img_w - input_w
        Z_min = np.random.randint(0, z_min_upper)
        Y_min = np.random.randint(0, y_min_upper)
        X_min = np.random.randint(0, x_min_upper)

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w
        return volume[Z_min: Z_max, Y_min: Y_max, X_min: X_max], mask[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __len__(self):
        return len(self.images_list)


    def __getitem__(self, idx):
        # if self.phase == 'train':
        #     volume_path = self.images_list[idx]
        #     mask_path = self.masks_list[idx]
        image_data = self.images[idx]
        mask_data = self.masks[idx]

        if np.random.random() < 0.4:
            index = np.random.randint(0, len(self.center_points_list[idx]))
            crop_center = self.center_points_list[idx][index]
            cropped_volume, cropped_mask = self.__random_center_crop_data(image_data, mask_data, self.crop_size, crop_center)
        else:
            cropped_volume, cropped_mask = self.__random_crop_data(image_data, mask_data, self.crop_size)

        cropped_volume = torch.from_numpy(cropped_volume).float()
        cropped_volume = torch.unsqueeze(cropped_volume, axis=0)
        return cropped_volume, cropped_mask

            
def test_AneurysmBlockLocationDS():
    root_dir = '../data/source_img/seg/WD_fir/train'
    config_file = '../data/source_img/seg/WD_fir/train/config.txt'
    ds = AneurysmSegmentationDS(root_dir, config_file, 'train', [128,128,128])
    # data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=pin_memory)
    data_loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=1, pin_memory=False)
    for i, (images, labels, _) in tqdm(enumerate(data_loader)):
        print(images.shape)
        print('hello world')
        break

if __name__ == '__main__':
    test_AneurysmBlockLocationDS()