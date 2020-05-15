import os
import numpy as np
import pickle
from tensorrtserver.api import *
import cv2
import time
import torch

import sys
sys.path.append('../')
from datasets.ncct_gan_dataset import NCCT_GAN_MASK_DS, NCCT_GAN_PREDICT_UTILS

import SimpleITK as sitk


class AverageMeter(object):
    """Computes and stores the average and current value"""
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

def infer_with_trt(infile, service_ip, service_port, service_name, outdir):
    protocol = ProtocolType.from_str('http')
    # ctx = InferContext('10.100.37.20:19992', protocol, 'dr_cls', -1, True)
    # ctx = InferContext('10.100.37.20:19992', protocol, 'feibuzhang_cls', -1, True)
    ctx = InferContext('{}:{}'.format(service_ip, service_port), protocol, service_name, -1, True)

    # test_file = '/data/zhangwd/data/examples/dr_deformable/test_label.txt'

    predict_utils = NCCT_GAN_PREDICT_UTILS()
    crop_size = [32, 512, 512]
    image_tensors, d_cnt, h_cnt, w_cnt = predict_utils.get_image_tensors(infile, crop_size)
    

    out_arr = []
    output_byte_size = 1*1*32*512*512
    # shm_op0_handle = shm.create_shared_memory_region("output0_data", "/output0_simple", output_byte_size)
    for image_tensor in image_tensors:
        result = ctx.run({'input': (image_tensor.numpy(),)}, {'output': InferContext.ResultFormat.RAW,}, 1)
        sub_arr = result['output'][0][0][0]
        out_arr.append(sub_arr)
        print('hello world!')
    dst_arr = predict_utils.compose_arrays_to_image(out_arr, [d_cnt, h_cnt, w_cnt], crop_size)

    os.makedirs(outdir, exist_ok=True)
    outname = os.path.join(outdir, os.path.basename(infile).replace('.nii.gz', '_fake.nii.gz'))
    sitk_img = sitk.GetImageFromArray(dst_arr)
    raw_img = sitk.ReadImage(infile)
    sitk_img.SetOrigin(raw_img.GetOrigin())
    sitk_img.SetDirection(raw_img.GetDirection())
    sitk_img.SetSpacing(raw_img.GetSpacing())
    sitk.WriteImage(sitk_img, outname)
    print('hello world')

if __name__ == '__main__':
    # infer_with_trt('/data/zhangwd/data/examples/dr_deformable/test_label.txt', '10.100.37.20', '19000', 'dr_cls', None)
    # infer_with_trt('../data/gan/hospital_4/experiment_registration2/8.2.out/NCCT/439856_first_BS_NCCT.nii.gz', '10.100.37.20', '19000', 'gan_ncct', '')
    infer_with_trt('../data/gan/hospital_6/experiment_registration2/8.2.out/NCCT/4692992_first_BS_NCCT.nii.gz', '10.100.37.20', '19000', 'gan_ncct', '../data/gan/hospital_6/experiment_registration2/10.predict_trt')
