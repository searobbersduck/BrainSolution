import os
import sys
sys.path.append('../')

abs_dir = os.getcwd()
work_dir = os.path.abspath(os.path.join(abs_dir,os.path.pardir))
sys.path.append(work_dir)
work_dir = os.path.abspath(os.path.join(abs_dir,os.path.pardir,os.path.pardir))
sys.path.append(work_dir)

# third lib medical net

from tube_seg.external_lib.MedicalNet.setting import parse_opts
from tube_seg.external_lib.MedicalNet.utils.logger import log


# third lib unet3d
# sys.path.append('../external_lib/pytorch-3dunet/pytorch3dunet')
sys.path.append('../../tube_seg/external_lib/pytorch-3dunet')
import importlib
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch3dunet.unet3d.model import get_model
from pytorch3dunet.unet3d.config import load_config
from pytorch3dunet.unet3d.utils import get_logger, get_tensorboard_formatter
from pytorch3dunet.unet3d.losses import get_loss_criterion
from pytorch3dunet.unet3d.metrics import get_evaluation_metric

import torch
import numpy as np
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

from datasets.aneurysm_segmentation_dataset import AneurysmSegmentationDS
# from datasets.neuro_vascular_segment_dataset import NeuroVascularSegmentDS

from tube_seg.utils.focalloss import FocalLoss

import time
from scipy import ndimage
import nibabel as nib

import SimpleITK as sitk

logger = get_logger('UNet3DTrain')

class MedicalNetSets():
    def __init__(self):
        super().__init__()
        self.input_W = 128

        self.n_epochs = 200
        self.save_intervals = 20
        self.model = 'unet3d'
        self.model_depth = 50
        self.ci_test = False
        self.save_folder = "./trails/models/{}_{}".format(self.model, self.model_depth)

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
            volumes, label_masks = batch_data

            volumes = Variable(volumes.cuda())
            out_masks = model(volumes)

            label_masks = torch.unsqueeze(label_masks, axis=1).int()

            # label_masks = Variable(label_masks.long().cuda())
            # loss_value_seg = loss_seg(out_masks, label_masks)

            # pred_masks_prob = torch.max(out_masks, 1)[0]
            loss_value_seg = loss_criterion(out_masks, Variable(label_masks.int().cuda()))

            if epoch%10 == 0 and loss_value_seg.detach().cpu().numpy() < 1.7:
                outdir = './tmp/tube_seg/unet3d/{}_{}_{}'.format(sets.input_W, sets.input_H, sets.input_D)
                os.makedirs(outdir, exist_ok=True)
                pred_mask1 = nn.Sigmoid()(out_masks).cpu().detach().numpy()[0][0]
                pred_mask1[pred_mask1 > 0.5] = 1
                pred_mask = np.array(pred_mask1, dtype=np.uint8)
                mask_img = nib.Nifti1Image(pred_mask, affine=np.eye(4))
                test_out_file = os.path.join(outdir, 'epoch_{}_index_{}_mask.nii'.format(epoch, batch_id))
                nib.save(mask_img, test_out_file)
                
                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(outdir, 'epoch_{}_index_{}_volume.nii'.format(epoch, batch_id)))
                writer.Execute(sitk.GetImageFromArray(volumes.detach().cpu()[0][0].numpy()))
                print('\n====> save to tmp file:\t{}\n'.format(test_out_file))


            loss = loss_value_seg
            loss.backward()                
            optimizer.step()
            
            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            log.info(
                    'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}'\
                    .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_seg.item(), avg_batch_time))
          
            if not sets.ci_test:
                # save model
                if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                #if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
                    model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
                    model_save_dir = os.path.dirname(model_save_path)
                    if not os.path.exists(model_save_dir):
                        os.makedirs(model_save_dir)
                    
                    log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
                    torch.save({
                                'ecpoch': epoch,
                                'batch_id': batch_id,
                                'state_dict': model_cpu.state_dict(),
                                'optimizer': optimizer.state_dict()},
                                model_save_path)

def main():
    # Load and log experiment configuration
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
    

    # sets = MedicalNetSets()

    # img_list = "../data/brain_henan/algo_mask/config/neuro_vascular_seg_mix_0_1.txt"
    # data_root = "../data/brain_henan/algo_mask/brain_seg_by_threshold_preprocessed"

    # crop_size = [sets.input_D, sets.input_H, sets.input_W]

    root_dir = '../data/source_img/seg/WD_fir/train'
    config_file = '../data/source_img/seg/WD_fir/train/config.txt'
    ds = AneurysmSegmentationDS(root_dir, config_file, 'train', [128,128,128])

    pin_memory = True

    # training_dataset = NeuroVascularSegmentDS(sets.data_root, sets.img_list, sets.phase, crop_size, crop_size)

    data_loader = DataLoader(ds, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=pin_memory)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    train(model1, data_loader, model, optimizer, scheduler, loss_criterion, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets) 

    print('pause')

if __name__ == '__main__':
    main()
    print('hello world!')