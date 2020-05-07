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

class AneurysmBlockLocationDS(Dataset):
    def __init__(self, root_dir, config_file, phase):
        self.root_dir = root_dir
        self.phase = phase
        self.config_file = []
        self.images_list = []
        self.labels_list = []
        self.pos_list = []
        self.neg_list = []
        with open(config_file) as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                if len(ss) != 2:
                    continue
                self.images_list.append(os.path.join(self.root_dir, ss[0]))
                if (int(ss[1])) > 0:
                    self.labels_list.append(1)
                    self.pos_list.append(os.path.join(self.root_dir, ss[0]))
                else:
                    self.labels_list.append(0)
                    self.neg_list.append(os.path.join(self.root_dir, ss[0]))
        self.pos_len = len(self.pos_list)
        self.neg_len = len(self.neg_list)
        # if self.phase == 'val':
        #     self.images_list = self.pos_list[:int(0.1*self.pos_len)] + self.neg_list[:int(0.1*self.neg_len)]
        #     self.labels_list = [1] * int(0.1*self.pos_len) + [0] * int(0.1*self.neg_len)

    def __len__(self):
        return len(self.images_list)
        
    def __getitem__(self, idx):
        # image_file = self.images_list[idx]
        if self.phase == 'train':
            if np.random.rand() < 0.4:
                image_file = self.pos_list[idx%self.pos_len]
                label = 1
            else:
                image_file = np.random.choice(self.neg_list)
                label = 0
        else:
            image_file = self.images_list[idx]
            label = self.labels_list[idx]

        image_data = np.load(image_file)

        image_data = np.array(image_data, dtype=np.float32)

        image_tensor = torch.from_numpy(image_data).float()
        image_tensor = torch.unsqueeze(image_tensor, axis=0)

        return image_tensor, label, image_file

def test_AneurysmBlockLocationDS():
    root_dir = '../data/block_pairs/WD_fir'
    config_file = '../data/block_pairs/WD_fir/config.txt'
    ds = AneurysmBlockLocationDS(root_dir, config_file, 'train')
    # data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=pin_memory)
    data_loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=1, pin_memory=False)
    for i, (images, labels, _) in tqdm(enumerate(data_loader)):
        print(images.shape)
        print('hello world')
        break


if __name__ == '__main__':
    test_AneurysmBlockLocationDS()