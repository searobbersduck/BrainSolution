import os
import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np

from datasets.ncct_gan_dataset import NCCT_GAN_MASK_DS, NCCT_GAN_PREDICT_UTILS
from torch.utils.data import DataLoader, Dataset

from models.pixel2pixel_3d_model import Pix2PixModel, ResnetGenerator

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse

'''
CUDA_VISIBLE_DEVICES=1 python 1.convert_pytorch_model_to_onnx.py --weights ../data/gan/hospital_4_2/experiment_registration2/9.2.model_out/model_train_ncct_to_dwi_bxxx_hospital4_2_nonmask_20200508/pixel2pixel_netG_epoch_400_loss_16.9547.pth --outname model.onnx
CUDA_VISIBLE_DEVICES=1 python 1.convert_pytorch_model_to_onnx.py --weights ../data/gan/hospital_6/experiment_registration2/9.2.model_out/model_train_cta_to_dwi_bxxx_hospital6_nonmask_20200514/pixel2pixel_netG_epoch_0_loss_85.3071.pth --outname model.onnx
'''

def parse_args():
    parser = argparse.ArgumentParser(description='convert pytorch model to onnx')
    parser.add_argument('--weights', required=True)
    parser.add_argument('--outname', default='dr.onnx')
    return parser.parse_args()

opt = parse_args()
print('\n====>opt:\t')
print(opt)
print('\n')
# model = resnet34(num_classes=6, shortcut_type=True, sample_size=128, sample_duration=128)
model = ResnetGenerator(1,1, 32, n_blocks=6)

# weights = '../train/model/dr_cls_yyy/ct_pos_recognition_0020_best.pth'
weights = opt.weights
model.load_state_dict(torch.load(weights))

model = model.cuda()
model.eval()

dummy_input = torch.randn(1,1,32,512,512).cuda()

outname = opt.outname
torch.onnx.export(model, dummy_input, outname, verbose=True, input_names=['input'], output_names=['output'])

print('====> export to onnx model!')

