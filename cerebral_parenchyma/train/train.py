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

from cerebral_parenchyma.datasets.cerebral_parenchyma_extract_dataset import CerebralParenchymaSegmentDS

print(os.path.abspath(os.path.curdir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, 'external_lib/brain-segmentation-pytorch'))
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
        self.check_point = './model/extract_cerebral_parenchyma/extract_cerebral_parenchyma_0000_best_loss_0.017.pth'
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
        root_dirs = ['../data/ncct/slice_2d/train', '../data/cta/slice_2d/train']
        config_xxx_files = ['../data/ncct/config/config_2d_cerebral_parenchyma_xxx_train.txt', '../data/cta/config/config_2d_cerebral_parenchyma_xxx_train.txt']
        config_yyy_files = ['../data/ncct/config/config_2d_cerebral_parenchyma_yyy_train.txt', '../data/cta/config/config_2d_cerebral_parenchyma_yyy_train.txt']
        train_ds = CerebralParenchymaSegmentDS(root_dirs, config_xxx_files, config_yyy_files, 'train', sets.crop_size, sets.crop_size)
        data_loader = DataLoader(train_ds, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=True)

        root_dirs = ['../data/ncct/slice_2d/val', '../data/cta/slice_2d/val']
        config_xxx_files = ['../data/ncct/config/config_2d_cerebral_parenchyma_xxx_val.txt', '../data/cta/config/config_2d_cerebral_parenchyma_xxx_val.txt']
        config_yyy_files = ['../data/ncct/config/config_2d_cerebral_parenchyma_yyy_val.txt', '../data/cta/config/config_2d_cerebral_parenchyma_yyy_val.txt']
        val_ds = CerebralParenchymaSegmentDS(root_dirs, config_xxx_files, config_yyy_files, 'train', sets.crop_size, sets.crop_size)
        val_dataloader = DataLoader(val_ds, batch_size=sets.batch_size, shuffle=False, num_workers=sets.num_workers, pin_memory=False)

        best_loss = 5e-1
        for epoch in range(sets.epochs):
            train(data_loader, torch.nn.DataParallel(unet).cuda(), dsc_loss, optimizer, epoch, sets.display)
            val_loss, val_log = val(val_dataloader, torch.nn.DataParallel(unet).cuda(), dsc_loss, optimizer, epoch, sets.display)
            if val_loss < best_loss:
                best_loss = val_loss
                print('\ncurrent best loss is: {}\n'.format(val_loss))
                os.makedirs(sets.model_dir, exist_ok=True)
                saved_model_name = os.path.join(sets.model_dir, 'extract_cerebral_parenchyma_{:04d}_best_loss_{:.3f}.pth'.format(epoch, val_loss))
                torch.save(unet.cpu().state_dict(), saved_model_name)
                print('====> save model:\t{}'.format(saved_model_name))


def get_maximal_connected_region(sitk_mask):
    """
    only support for one class label output (0 and 1 for back/fore-ground
    :param sitk_mask:
    :return:
    """
    sitk_mask = sitk.Cast(sitk_mask, sitk.sitkUInt8)
    # get the connected components
    sitk_connect_componet = sitk.ConnectedComponent(sitk_mask)

    # statistic analysis of these components
    statsFilter = sitk.LabelIntensityStatisticsImageFilter()
    statsFilter.Execute(sitk_connect_componet, sitk_mask)
    areas = []
    labels = []
    for label in statsFilter.GetLabels():
        bbox = statsFilter.GetBoundingBox(label)
        area = statsFilter.GetNumberOfPixels(label)
        labels.append(label)
        areas.append(area)

    # get the max region
    id = areas.index(max(areas))
    label_value = labels[id]
    np_connecte_component = sitk.GetArrayFromImage(sitk_connect_componet)

    # 0 for background and 1 for forground
    a = np_connecte_component
    a[a != label_value] = 0
    a[a == label_value] = 1

    sitk_maximal_region = sitk.GetImageFromArray(a)
    sitk_maximal_region.SetSpacing(sitk_mask.GetSpacing())
    sitk_maximal_region.SetDirection(sitk_mask.GetDirection())
    sitk_maximal_region.SetOrigin(sitk_mask.GetOrigin())

    return sitk_maximal_region


def fill_hole(in_mask):
    filter = sitk.VotingBinaryHoleFillingImageFilter()
    filter.SetBackgroundValue(0)
    filter.SetForegroundValue(1)
    filter.SetRadius(8)
    out_mask = filter.Execute(in_mask)
    return out_mask


def inference(infile, model_pth, outfile, mask_file=None, is_dcm=False, mask_thres=0.5):
    '''
    cmd: inference('../data/cta/image/dicom/1.3.12.2.1107.5.1.4.60320.30000012022300460003100013565', '../data/cta/predict/1.3.12.2.1107.5.1.4.60320.30000012022300460003100013565.nii.gz', './model/extract_cerebral_parenchyma/extract_cerebral_parenchyma_0000_best_loss_0.017.pth', mask_file='../data/cta/mask/Ori_nii/1.3.12.2.1107.5.1.4.60320.30000012022300460003100013565.nii.gz', is_dcm=True, mask_thres=0.5)
    '''
    # load image
    if is_dcm:
        series_reader = sitk.ImageSeriesReader()
        dicomfilenames = series_reader.GetGDCMSeriesFileNames(infile)
        series_reader.SetFileNames(dicomfilenames)

        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()

        image = series_reader.Execute()
    else:
        image = sitk.ReadImage(infile)
    # load model
    unet = UNet(in_channels=1, out_channels=1)
    unet.load_state_dict(torch.load(model_pth))
    model = torch.nn.DataParallel(unet).cuda()
    model.eval()
    # 3d to 2d
    arr = sitk.GetArrayFromImage(image)
    mask = np.zeros(arr.shape)
    out_mask3d = np.zeros(arr.shape, dtype=np.uint8)
    print(arr.shape)
    for z in tqdm(range(arr.shape[0])):
        in_img = arr[z]
        in_img = np.array(in_img, dtype=np.float32)
        in_tensor = torch.from_numpy(in_img).float()
        in_tensor = torch.unsqueeze(in_tensor, axis=0)
        in_tensor = torch.unsqueeze(in_tensor, axis=0)
        out_mask = model(Variable(in_tensor.cuda()))
        out_mask = out_mask.detach().cpu().numpy()
        out_mask = np.squeeze(out_mask)
        mask[z] = out_mask
    mask_index = np.where(mask>mask_thres)
    out_mask3d[mask_index] = 1
    sitk_mask = sitk.GetImageFromArray(out_mask3d)
    sitk_mask.CopyInformation(image)
    sitk_mask = get_maximal_connected_region(sitk_mask)
    sitk_mask = fill_hole(sitk_mask)
    if outfile is not None:
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(outfile)
        writer.Execute(sitk_mask)
        print('====> generate mask file:\t{}'.format(outfile))

    if mask_file is not None:
        gt_mask_image = sitk.ReadImage(mask_file)
        gt_arr = sitk.GetArrayFromImage(gt_mask_image)
        gt_arr = np.array(gt_arr, dtype=np.uint8)
        pred_arr = out_mask3d.reshape(-1)
        gt_arr = gt_arr.reshape(-1)
        intersection = (pred_arr * gt_arr).sum()
        smooth = 1
        dice = (2. * intersection + smooth) / (
            pred_arr.sum() + gt_arr.sum() + smooth
        )
        print('dice:\t{:.4f}'.format(dice))

    return sitk_mask

def extract_region_by_mask(image_file, mask_file, default_value=-1024, out_image_file=None):
    image = sitk.ReadImage(image_file)
    mask = sitk.ReadImage(mask_file)
    maskfilter = sitk.MaskImageFilter()
    maskfilter.SetOutsideValue(default_value)
    src_img = sitk.Cast(image, sitk.sitkInt16)
    mask_img = sitk.Cast(mask, sitk.sitkInt16)
    out_img = maskfilter.Execute(src_img, mask_img)

    if out_image_file is not None: 
        os.makedirs(os.path.dirname(out_image_file), exist_ok=True)
        writerfilter = sitk.ImageFileWriter()
        writerfilter.SetFileName(out_image_file)
        writerfilter.Execute(out_img)

    return out_img

def extract_region_by_mask1(image, mask, default_value=-1024, out_image_file=None):
    maskfilter = sitk.MaskImageFilter()
    maskfilter.SetOutsideValue(default_value)
    src_img = sitk.Cast(image, sitk.sitkInt16)
    mask_img = sitk.Cast(mask, sitk.sitkInt16)
    out_img = maskfilter.Execute(src_img, mask_img)

    if out_image_file is not None: 
        os.makedirs(os.path.dirname(out_image_file), exist_ok=True)
        writerfilter = sitk.ImageFileWriter()
        writerfilter.SetFileName(out_image_file)
        writerfilter.Execute(out_img)

    return out_img



if __name__ == '__main__':
    fire.Fire()
    # main()
    # inference('../data/cta/image/dicom/1.3.12.2.1107.5.1.4.60320.30000012022300460003100013565', '../data/cta/predict/1.3.12.2.1107.5.1.4.60320.30000012022300460003100013565.nii.gz', './model/extract_cerebral_parenchyma/extract_cerebral_parenchyma_0002_best_loss_0.012.pth', mask_file='../data/cta/mask/Ori_nii/1.3.12.2.1107.5.1.4.60320.30000012022300460003100013565.nii.gz', is_dcm=True, mask_thres=0.5)
    # inference('../data/cta/image/dicom/1.3.12.2.1107.5.1.4.60320.30000012022300460003100013565', '../data/cta/predict/1.3.12.2.1107.5.1.4.60320.30000012022300460003100013565.nii.gz', './model/extract_cerebral_parenchyma/extract_cerebral_parenchyma_0056_best_loss_0.011.pth', mask_file='../data/cta/mask/Ori_nii/1.3.12.2.1107.5.1.4.60320.30000012022300460003100013565.nii.gz', is_dcm=True, mask_thres=0.5)
    # 20180712234344
    # inference('../data/ncct/brain-NCCT-with-mask/20180712234344_Recon_2__5mm_10mm_3_CT.nii.gz', '../data/ncct/predict/20180712234344_Recon_2__5mm_10mm_3_CT.nii.gz', './model/extract_cerebral_parenchyma/extract_cerebral_parenchyma_0002_best_loss_0.012.pth', mask_file='../data/ncct/brain-NCCT-with-mask/20180712234344_Recon_2__5mm_10mm_3_brain_mask.nii.gz', is_dcm=False, mask_thres=0.5)
    # inference('../data/ncct/brain-NCCT-with-mask/20180712234344_Recon_2__5mm_10mm_3_CT.nii.gz', '../data/ncct/predict/20180712234344_Recon_2__5mm_10mm_3_CT.nii.gz', './model/extract_cerebral_parenchyma/extract_cerebral_parenchyma_0056_best_loss_0.011.pth', mask_file='../data/ncct/brain-NCCT-with-mask/20180712234344_Recon_2__5mm_10mm_3_brain_mask.nii.gz', is_dcm=False, mask_thres=0.5)
    # inference('/ssd2/zhangwd/data/brain/gan/ncct2dwi/experiment_registration2/2.nii_file_ori/469298_first_FU_DWI_B0.nii.gz', '/ssd2/zhangwd/data/brain/cerebral_parenchyma/other/predict/469298_first_FU_DWI_B0.nii.gz', './model/extract_cerebral_parenchyma/extract_cerebral_parenchyma_0056_best_loss_0.011.pth')
