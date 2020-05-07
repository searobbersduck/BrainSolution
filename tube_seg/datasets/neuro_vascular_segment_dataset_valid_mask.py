import math
import os
import random

import numpy as np
from torch.utils.data import Dataset, DataLoader
import nibabel
from scipy import ndimage
import time
import torch
import torch.nn as nn

class NeuroVascularSegmentDS(Dataset):
    def __init__(self, root_dir, img_list, phase,crop_size, scale_size):
        self.series_list = []
        self.images_list = []
        self.masks_list = []
        self.phase = phase
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.ranges = []
        with open(img_list, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                if len(ss) != 8:
                    continue
                image_file = os.path.join(root_dir, ss[0])
                if not os.path.isfile(image_file):
                    continue
                mask_file = os.path.join(root_dir, ss[1])
                if not os.path.isfile(mask_file):
                    continue
                self.images_list.append(image_file)
                self.masks_list.append(mask_file)
                min_z = int(ss[2])
                max_z = int(ss[3])
                min_y = int(ss[4])
                max_y = int(ss[5])
                min_x = int(ss[6])
                max_x = int(ss[7])
                self.ranges.append([min_z, max_z, min_y, max_y, min_x, max_x])

    def __extract_valid_range__(self, label):
        """
        Cut off the invalid area
        """
        zero_value = 0
        non_zeros_idx = np.where(label != zero_value)
        
        [max_z, max_h, max_w] = np.max(np.array(non_zeros_idx), axis=1)
        [min_z, min_h, min_w] = np.min(np.array(non_zeros_idx), axis=1)
        
        return min_z, min_h, min_w, max_z, max_h, max_w
                
    def __random_crop_data(self, volume, mask, size, img_range):
        # note: range:[min_z, max_z, min_y, max_y, min_x, max_x]
        # [img_d, img_h, img_w] = volume.shape
        [img_d, img_h, img_w] = img_range[1], img_range[3], img_range[5]
        left_d, left_h, left_w = img_range[0], img_range[2], img_range[4]
        [input_d, input_h, input_w] = size
        z_min_upper = img_d - input_d
        y_min_upper = img_h - input_h
        x_min_upper = img_w - input_w
        Z_min = np.random.randint(left_d, z_min_upper)
        Y_min = np.random.randint(left_h, y_min_upper)
        X_min = np.random.randint(left_w, x_min_upper)

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w
#         print('x:[{}-{}]\ty:[{}-{}]\tz:[{}-{}]'.format(X_min, X_max, Y_min, Y_max, Z_min, Z_max))
#         print(mask.dtype)
#         print(mask.max())
        return volume[Z_min: Z_max, Y_min: Y_max, X_min: X_max], mask[Z_min: Z_max, Y_min: Y_max, X_min: X_max]
    
    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        if self.phase == 'train':
            volume_path = self.images_list[idx]
            mask_path = self.masks_list[idx]
            # print(mask_path)
            # with open(volume_path, 'rb') as f:
            #         volume_data = np.load(f)
            # with open(mask_path, 'rb') as f:
            #         mask_data = np.load(f)
            volume_data = np.load(volume_path)
            mask_data = np.load(mask_path)
            cropped_volume, cropped_mask = self.__random_crop_data(volume_data, mask_data, self.crop_size, self.ranges[idx])
            cropped_volume = torch.from_numpy(cropped_volume).float()
            cropped_volume = torch.unsqueeze(cropped_volume, axis=0)
            return cropped_volume, cropped_mask, volume_path, mask_path

    
if __name__ == '__main__':
    print('NeuroVascularSegmentDS')
