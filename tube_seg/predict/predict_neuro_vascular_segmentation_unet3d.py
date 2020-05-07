import os
import sys
sys.path.append('../')
sys.path.append('../../external_model/lib/MedicalNet')
# third lib medical net
from setting import parse_opts 
from utils.logger import log

# third lib unet3d
# sys.path.append('../external_lib/pytorch-3dunet/pytorch3dunet')
sys.path.append('../external_lib/pytorch-3dunet')
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

from datasets.neuro_vascular_segment_dataset import NeuroVascularSegmentDS

import time
from scipy import ndimage

import nibabel as nib

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

def train(model_cpu, loss_criterion, data_loader, model, optimizer, scheduler, total_epochs, save_interval, save_folder, sets):
    # settings
    batches_per_epoch = len(data_loader)
    log.info('{} epochs in total, {} batches per epoch'.format(total_epochs, batches_per_epoch))
    loss_seg = nn.CrossEntropyLoss(ignore_index=-1)

    print("Current setting is:")
    print(sets)
    print("\n\n")

    loss_seg = loss_seg.cuda()

    model.eval()

    train_time_sp = time.time()
    for epoch in range(total_epochs):
        log.info('Start epoch {}'.format(epoch))

        # scheduler.step()
        log.info('lr = {}'.format(scheduler.get_lr()))

        for batch_id, batch_data in enumerate(data_loader):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, label_masks = batch_data

            # volumes = Variable(volumes.cuda())
            volumes = volumes.cuda()
            out_masks = model(volumes)

            pred_masks1 = torch.max(out_masks, 1)[0]
            pred_masks = pred_masks1.detach().cpu().numpy()
            pred_masks[pred_masks>0.5] = 1
            max_v = np.max(pred_masks)
            print(max_v)
            if (max_v == 1):
                pred_mask = np.array(pred_masks[0], dtype=np.uint8)
                mask_img = nib.Nifti1Image(pred_mask, affine=np.eye(4))
                nib.save(mask_img, 'test.nii')
            

            # label_masks = torch.unsqueeze(label_masks, axis=1).int()

            label_masks = Variable(label_masks.long().cuda())
            # loss_value_seg1 = loss_criterion(pred_masks1, Variable(label_masks.int().cuda()))
            loss_value_seg = loss_seg(out_masks, label_masks)
            
            loss = loss_value_seg
            # loss.backward()                
            # optimizer.step()
            
            # avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)
            # log.info(
            #         'Batch: {}-{} ({}), loss = {:.3f}, loss_seg = {:.3f}, avg_batch_time = {:.3f}'\
            #         .format(epoch, batch_id, batch_id_sp, loss.item(), loss_value_seg.item(), avg_batch_time))
          
            # if not sets.ci_test:
            #     # save model
            #     if batch_id == 0 and batch_id_sp != 0 and batch_id_sp % save_interval == 0:
            #     #if batch_id_sp != 0 and batch_id_sp % save_interval == 0:
            #         model_save_path = '{}_epoch_{}_batch_{}.pth.tar'.format(save_folder, epoch, batch_id)
            #         model_save_dir = os.path.dirname(model_save_path)
            #         if not os.path.exists(model_save_dir):
            #             os.makedirs(model_save_dir)
                    
            #         log.info('Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)) 
            #         torch.save({
            #                     'ecpoch': epoch,
            #                     'batch_id': batch_id,
            #                     'state_dict': model_cpu.state_dict(),
            #                     'optimizer': optimizer.state_dict()},
            #                     model_save_path)

def main():
    # Load and log experiment configuration
    config, sets = load_config()
    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    model1 = get_model(config)
    # state_dict = torch.load(sets.pretrain_path)['state_dict']
    # model1.load_state_dict(state_dict)
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

    img_list = "../data/brain_henan/algo_mask/config/neuro_vascular_seg_mix_0_1.txt"
    data_root = "../data/brain_henan/algo_mask/brain_seg_by_threshold_preprocessed"

    crop_size = [sets.input_D, sets.input_H, sets.input_W]

    pin_memory = False

    training_dataset = NeuroVascularSegmentDS(sets.data_root, sets.img_list, sets.phase, crop_size, crop_size)
    data_loader = DataLoader(training_dataset, batch_size=sets.batch_size, shuffle=True, num_workers=sets.num_workers, pin_memory=pin_memory)

    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    train(model1, loss_criterion, data_loader, model, optimizer, scheduler, total_epochs=sets.n_epochs, save_interval=sets.save_intervals, save_folder=sets.save_folder, sets=sets) 

    print('pause')

if __name__ == '__main__':
    main()