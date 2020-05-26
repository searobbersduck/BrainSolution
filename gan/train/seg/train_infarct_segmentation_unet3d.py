import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))

from gan.datasets.ncct_segmentation_dataset import NCCTCoreInfarctSegmentation

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir, 'tube_seg/external_lib/pytorch-3dunet'))
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.utils import get_logger, get_tensorboard_formatter
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric


import torch
import numpy as np
import importlib
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tube_seg.external_lib.MedicalNet.utils.logger import log
from tube_seg.utils.focalloss import FocalLoss

import time
import SimpleITK as sitk

def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = config['optimizer']
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config['weight_decay']
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    return optimizer


def _create_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)
    if lr_config is None:
        # use ReduceLROnPlateau as a default scheduler
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    else:
        class_name = lr_config.pop('name')
        m = importlib.import_module('torch.optim.lr_scheduler')
        clazz = getattr(m, class_name)
        # add optimizer to the config
        lr_config['optimizer'] = optimizer
        return clazz(**lr_config)


def train(model_cpu, data_loader, model, optimizer, scheduler, loss_criterion, total_epochs, save_interval, save_folder, sets):
    # settings
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    # loss_seg = nn.CrossEntropyLoss(ignore_index=-1)
    loss_seg = FocalLoss()

    print("Current setting is:")
    print(sets)
    print("\n\n")

    loss_seg = loss_seg.cuda()

    model.train()

    train_time_sp = time.time()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))

        scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))

        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label_masks, image_files, mask_files = batch_data
            volumes = Variable(volumes.cuda())
            out_masks = model(volumes)
            label_masks = torch.unsqueeze(label_masks, axis=1).long().cuda()
            # loss_value_seg = loss_seg(out_masks, label_masks)
            loss_value_seg = loss_criterion(out_masks[:,1:,:,:,:], Variable(label_masks.int().cuda()))
            print(loss_value_seg)
            loss = loss_value_seg
            loss.backward()                
            optimizer.step()
            if (epoch+1)%100 == 0 and loss_value_seg.detach().cpu().numpy() < 1.7:
                outdir = '../../data/gan/hospital_4_2/experiment_registration3/8.2.out/train_result'
                os.makedirs(outdir, exist_ok=True)
                pred_mask1 = nn.Sigmoid()(out_masks).cpu().detach().numpy()[0][1]
                pred_mask1[pred_mask1 > 0.5] = 1
                pred_mask = np.array(pred_mask1, dtype=np.uint8)
                out_pred_file = os.path.join(outdir, os.path.basename(mask_files[0]).replace('.nii.gz', '_fake.nii.gz'))
                out_gt_file = os.path.join(outdir, os.path.basename(mask_files[0]).replace('.nii.gz', '_real.nii.gz'))
                out_ct_file = os.path.join(outdir, os.path.basename(image_files[0]))

                sitk_pred = sitk.GetImageFromArray(pred_mask)
                sitk.WriteImage(sitk_pred, out_pred_file)

                gt_arr = label_masks.detach().cpu().int()[0][0].numpy()
                sitk_gt = sitk.GetImageFromArray(gt_arr)
                sitk.WriteImage(sitk_gt, out_gt_file)

                ct_arr = volumes.detach().cpu()[0][0].numpy()
                sitk_ct = sitk.GetImageFromArray(ct_arr)
                sitk.WriteImage(sitk_ct, out_ct_file)


def main():
    # Load and log experiment configuration
    logger = get_logger('UNet3DTrain')
    config, sets = load_config()
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info('Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model1 = get_model(config)
    model = torch.nn.DataParallel(model1).cuda()

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)
    # Create the optimizer
    optimizer = _create_optimizer(config, model)
    # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(config, optimizer)

    root_dir = '../../data/gan/hospital_4_2/experiment_registration3/5 dwi_rigid_align_ncct'
    config_file = '../../data/gan/hospital_4_2/1.rapid/config.txt'
    phase = 'train'
    crop_size = [128,256,256]
    ds = NCCTCoreInfarctSegmentation(root_dir, config_file, phase, crop_size)
    pin_memory = True
    data_loader = DataLoader(ds, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=pin_memory)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    train(model1, data_loader, model, optimizer, scheduler, loss_criterion, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets) 



if __name__ == '__main__':
    main()
