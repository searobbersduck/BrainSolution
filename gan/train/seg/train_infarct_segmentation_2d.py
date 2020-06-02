import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
import time

import SimpleITK as sitk

import fire

import sys
print(__file__)
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))

from cerebral_parenchyma.datasets.cerebral_parenchyma_extract_dataset import CerebralParenchymaSegmentDS

print(os.path.abspath(os.path.curdir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'cerebral_parenchyma/external_lib/brain-segmentation-pytorch'))
from logger import Logger
from loss import DiceLoss
from transform import transforms
from unet import UNet
from utils import log_images, dsc


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training U-Net model for segmentation of brain MRI"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    parser.add_argument(
        "--images", type=str, default="./kaggle_3m", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    return parser.parse_args()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Options():
    def __init__(self):
        self.lr = 1e-3
        self.beta1 = 0.9
        self.gan_mode = 'lsgan'
        self.direction = 'AtoB'
        self.lambda_L1 = 4
        self.epochs = 1000
        self.num_workers = 4
        self.phase = 'train'
        self.batch_size = 8
        self.pin_memory = True
        self.display = 100
        self.save_interval = 10
        self.intermidiate_result_root = '../../data/gan/ncct2dwi/experiment_registration2/8.out/train_result/intermidiate_result_{}'.format(__file__.split('.')[0])
        # add patch discriminator
        self.patch_D = False
        self.num_patches_D = 5
        self.patch_size_D = [64, 64, 64]
        # crop_size
        self.crop_size = [160, 224, 288]
        # self.crop_size = [8, 8, 8]

        self.root_dir = '../../data/gan/ncct2dwi/experiment_registration2/8.out'
        self.config_file = '../../data/gan/ncct2dwi/experiment_registration2/8.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt'
        self.check_point = None #'./model/extract_cerebral_parenchyma/extract_cerebral_parenchyma_0000_best_loss_0.017.pth'
        self.model_dir = './model/extract_cerebral_parenchyma'
        self.phase = 'train'


def train(train_dataloader, model, criterion, optimizer, epoch, display):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    logger = []
    for num_iter, (images, masks, image_names, mask_names) in enumerate(train_dataloader):
        data_time.update(time.time()-end)
        out = model(Variable(images.cuda()))
        loss = criterion(out, masks.cuda())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time()-end)
        end = time.time()
        losses.update(loss.data.cpu().numpy(), len(images))
        # print('loss:[{:.3f}]\tmin:[{:.3f}]\tmax:[{:.3f}]\tmin_mask:[{:.3f}]\tmax_mask:[{:.3f}]\tmask_names:[{}]'.format(
        #     loss, out.min(), out.max(), masks.min(), masks.max(), mask_names))
        if (num_iter+1) % display == 0:
            print_info = 'Epoch:[{}][{}/{}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\t'\
                'Data {data_time.avg:.3f}\t''Loss {loss.avg:.4f}'.format(
                    epoch, num_iter, len(train_dataloader), 
                    batch_time=batch_time, data_time=data_time, loss=losses)
            print(print_info)
            logger.append(print_info)
    return losses.avg, logger

def val(train_dataloader, model, criterion, optimizer, epoch, display):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    end = time.time()
    logger = []
    for num_iter, (images, masks, image_names, mask_names) in enumerate(train_dataloader):
        data_time.update(time.time()-end)
        out = model(Variable(images.cuda()))
        loss = criterion(out, masks.cuda())
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
        batch_time.update(time.time()-end)
        end = time.time()
        losses.update(loss.data.cpu().numpy(), len(images))
        # print('loss:[{:.3f}]\tmin:[{:.3f}]\tmax:[{:.3f}]\tmin_mask:[{:.3f}]\tmax_mask:[{:.3f}]\tmask_names:[{}]'.format(
        #     loss, out.min(), out.max(), masks.min(), masks.max(), mask_names))
        if (num_iter+1) % display == 0:
            print_info = 'Epoch:[{}][{}/{}]\tTime {batch_time.val:3f} ({batch_time.avg:.3f})\t'\
                'Data {data_time.avg:.3f}\t''Loss {loss.avg:.4f}'.format(
                    epoch, num_iter, len(train_dataloader), 
                    batch_time=batch_time, data_time=data_time, loss=losses)
            print(print_info)
            logger.append(print_info)
    return losses.avg, logger


def main():

    sets = Options()

    unet = UNet(in_channels=1, out_channels=1)
    if sets.check_point is not None:
        unet.load_state_dict(torch.load(sets.check_point))
    model = torch.nn.DataParallel(unet).cuda()
    
    dsc_loss = DiceLoss()
    best_validation_dsc = 0.0

    optimizer = optim.Adam(unet.parameters(), lr=sets.lr)

    if sets.phase == 'train':
        root_dirs = ['../../data/gan/hospital_4_2/experiment_seg_2d/infarct/train']
        config_xxx_files = ['../../data/gan/hospital_4_2/experiment_seg_2d/infarct/config/config_2d_cerebral_parenchyma_xxx_train.txt']
        config_yyy_files = ['../../data/gan/hospital_4_2/experiment_seg_2d/infarct/config/config_2d_cerebral_parenchyma_yyy_train.txt']
        train_ds = CerebralParenchymaSegmentDS(root_dirs, config_xxx_files, config_yyy_files, 'train', sets.crop_size, sets.crop_size)
        data_loader = DataLoader(train_ds, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=True)

        root_dirs = ['../../data/gan/hospital_4_2/experiment_seg_2d/infarct/val']
        config_xxx_files = ['../../data/gan/hospital_4_2/experiment_seg_2d/infarct/config/config_2d_cerebral_parenchyma_xxx_val.txt']
        config_yyy_files = ['../../data/gan/hospital_4_2/experiment_seg_2d/infarct/config/config_2d_cerebral_parenchyma_yyy_val.txt']
        val_ds = CerebralParenchymaSegmentDS(root_dirs, config_xxx_files, config_yyy_files, 'train', sets.crop_size, sets.crop_size)
        val_dataloader = DataLoader(val_ds, batch_size=sets.batch_size, shuffle=False, num_workers=sets.num_workers, pin_memory=False)

        best_loss = 5e-1
        for epoch in range(sets.epochs):
            train(data_loader, torch.nn.DataParallel(unet).cuda(), dsc_loss, optimizer, epoch, sets.display)
            val_loss, val_log = val(val_dataloader, torch.nn.DataParallel(unet).cuda(), dsc_loss, optimizer, epoch, 10)
            if val_loss < best_loss:
                best_loss = val_loss
                print('\ncurrent best loss is: {}\n'.format(val_loss))
                os.makedirs(sets.model_dir, exist_ok=True)
                saved_model_name = os.path.join(sets.model_dir, 'extract_cerebral_parenchyma_{:04d}_best_loss_{:.3f}.pth'.format(epoch, val_loss))
                torch.save(unet.cpu().state_dict(), saved_model_name)
                print('====> save model:\t{}'.format(saved_model_name))


if __name__ == '__main__':
    main()

