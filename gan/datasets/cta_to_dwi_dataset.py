'''
@Description: 
@Version: 1.0
@Autor: searobbersanduck
@Date: 2020-03-27 17:01:15
@LastEditors: searobbersanduck
@LastEditTime: 2020-06-01 15:08:34
@License : (C)Copyright 2020-2021, MIT
'''

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
import SimpleITK as sitk

import fire
from glob import glob
from tqdm import tqdm

'''
@description: 提取ct数据中对应的非空气部分
@param {type} 
@return: 
@author: searobbersanduck
'''
def extract_valid_volume(ct_file, dwi_file):
    ct_data = sitk.ReadImage(ct_file)
    ct_arr = sitk.GetArrayFromImage(ct_data)
    dwi_data = sitk.ReadImage(dwi_file)
    dwi_arr = sitk.GetArrayFromImage(dwi_data)

    # calculate ct ranges
    low_thres = -100
    ranges = np.where(ct_arr > low_thres)
    [max_z, max_h, max_w] = np.max(np.array(ranges), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(ranges), axis=1)
    # print('cta threshold ranges, z:[{}-{}], y:[{}-{}], x:[{}-{}]'.format(
    #     min_z, max_z, min_h, max_h, min_w, max_w
    # ))
    cta_ranges = [min_z, max_z, min_h, max_h, min_w, max_w]
    # calculate dwi ranges
    low_thres = 100
    ranges = np.where(dwi_arr > low_thres)
    [max_z, max_h, max_w] = np.max(np.array(ranges), axis=1)
    [min_z, min_h, min_w] = np.min(np.array(ranges), axis=1)
    # print('dwi threshold ranges, z:[{}-{}], y:[{}-{}], x:[{}-{}]'.format(
    #     min_z, max_z, min_h, max_h, min_w, max_w
    # ))
    dwi_ranges = [min_z, max_z, min_h, max_h, min_w, max_w]

    return cta_ranges, dwi_ranges

def test_extract_valid_volume():
    ct_file = '../data/gan/cta2dwi/case_178_forHuang/CTA/3901698_first_BS_CTA_rigid_affine_aligned.nii.gz'
    dwi_file = '../data/gan/cta2dwi/case_178_forHuang/DWI/3901698_first_FU_DWI_rigid_affine_aligned.nii.gz'

    extract_valid_volume(ct_file, dwi_file)


# 分析cta生成dwi任务中，dwi原始数据
def analyze_cta2dwi_ori_data(indir, outfile):
    
    ''' note:
    tree -L 5

    ├── 1291454
    │   └── first
    │       ├── baseline
    │       │   └── CTA
    │       │       └── 1.2.840.113704.1.111.8472.1487222869.25
    │       └── follow-up
    │           └── DWI
    │               ├── 1.3.46.670589.11.42500.5.0.4456.2017022317020963159
    │               └── qu_adc
    ├── 1305155
    │   └── first
    │       ├── baseline
    │       │   └── CTA
    │       │       └── 1.2.840.113704.1.111.13540.1538823328.20
    │       └── follow-up
    │           └── DWI
    │               ├── 1.3.12.2.1107.5.2.19.145446.201810101749125709523838.0.0.0
    │               └── qu_adc
    '''
    '''
    indir: /home/zhangwd/code/work/BrainSolution/gan/data/gan/cta2dwi/Atlas-crec-CTA-ASPECT/2 Patient_dcm_sorted
    relative indir: ../data/gan/cta2dwi/Atlas-crec-CTA-ASPECT/2 Patient_dcm_sorted
    '''

    def analyze_dwi_data(series_id):
        info = []
        info.append('\t====> processing {}'.format(series_id))
        reader = sitk.ImageSeriesReader()
        dicomfilenames = reader.GetGDCMSeriesFileNames(series_id)
        reader.SetFileNames(dicomfilenames)
        image = reader.Execute()
        info.append('\t\tDirection:\t{}'.format(image.GetDirection()))
        info.append('\t\tSize:\t{}'.format(image.GetSize()))
        info.append('\t\tSpacing:\t{}'.format(image.GetSpacing()))
        return info
            
    patients = os.listdir(indir)
    print('====> process patients number:\t{}'.format(len(patients)))
    info = []
    for patient_id in patients:
        patient = os.path.join(indir, patient_id)
        # DWI analysis
        DWI_path = os.path.join(patient, 'first', 'follow-up', 'DWI')
        sub_modalitys = os.listdir(DWI_path)
        dwi_path = None
        if 'qu_adc' in sub_modalitys:
            dwi_path = 'qu_adc'
        else:
            dwi_path = sub_modalitys[0]
        dwi_path = os.path.join(DWI_path, dwi_path)
        if not os.path.isdir(dwi_path):
            continue
        info1 = analyze_dwi_data(dwi_path)
        # CTA analysis
        CTA_path = os.path.join(patient, 'first', 'baseline', 'CTA')
        sub_modalitys = os.listdir(CTA_path)
        cta_path = sub_modalitys[0]
        cta_path = os.path.join(CTA_path, cta_path)
        if not os.path.isdir(cta_path):
            continue
        info2 = analyze_dwi_data(cta_path)
        info += ['DWI:'] + info1 + ['CTA:'] + info2 + ['\n']
    outdir = os.path.dirname(outfile)
    os.makedirs(outdir, exist_ok=True)
    with open(outfile, 'w') as f:
        f.write('\n'.join(info))
    for line in info:
        print(line)

# 提取cta to dwi所有数据的边界
def extract_valid_volume_all(cta_dir, dwi_dir):
    cta_files = glob(os.path.join(cta_dir, '*affine_aligned.nii*'))
    dwi_files = []
    for cta_file in cta_files:
        if not os.path.isfile(cta_file):
            continue
        index = os.path.basename(cta_file).split('_')[0]
        dwi_file = os.path.join(dwi_dir, '{}_first_FU_DWI_rigid_affine_aligned.nii.gz'.format(index))
        if not os.path.isfile(dwi_file):
            continue
        dwi_files.append(dwi_file)
    assert len(cta_files) == len(dwi_files)

    max_d = 0
    max_h = 0
    max_w = 0
    for i in tqdm(range(len(cta_files))):
        cta_range, dwi_range = extract_valid_volume(cta_files[i], dwi_files[i])
        d = cta_range[1] - cta_range[0]
        h = cta_range[3] - cta_range[2]
        w = cta_range[5] - cta_range[4]
        if max_d < d:
            max_d = d
        if max_h < h:
            max_h = h
        if max_w < w:
            max_w = w
    print('\n ====> max range:\t[{}, {}, {}]\n'.format(max_d, max_h, max_w)) 


# 提取ncct to dwi所有数据的边界
def extract_valid_volume_all(cta_dir, dwi_dir):
    cta_files = glob(os.path.join(cta_dir, '*_NCCT.nii*'))
    dwi_files = []
    for cta_file in cta_files:
        if not os.path.isfile(cta_file):
            continue
        index = os.path.basename(cta_file).split('_')[0]
        dwi_file = os.path.join(dwi_dir, '{}_first_FU_DWI.nii.gz'.format(index))
        if not os.path.isfile(dwi_file):
            continue
        dwi_files.append(dwi_file)
    assert len(cta_files) == len(dwi_files)

    max_d = 0
    max_h = 0
    max_w = 0
    for i in tqdm(range(len(cta_files))):
        cta_range, dwi_range = extract_valid_volume(cta_files[i], dwi_files[i])
        d = cta_range[1] - cta_range[0]
        h = cta_range[3] - cta_range[2]
        w = cta_range[5] - cta_range[4]
        if max_d < d:
            max_d = d
        if max_h < h:
            max_h = h
        if max_w < w:
            max_w = w
    print('\n ====> max range:\t[{}, {}, {}]\n'.format(max_d, max_h, max_w)) 

    
# 提取所有数据的边界并保存成配置文件，以便后续处理
def genereate_cta2dwi_range_config_file(cta_dir, dwi_dir, out_config_file, cta_pattern='*affine_aligned.nii*', dwi_pattern='_first_FU_DWI_rigid_affine_aligned.nii.gz'):
    cta_files = glob(os.path.join(cta_dir, cta_pattern))
    dwi_files = []
    for cta_file in cta_files:
        if not os.path.isfile(cta_file):
            continue
        index = os.path.basename(cta_file).split('_')[0]
        dwi_file = os.path.join(dwi_dir, '{}{}'.format(index, dwi_pattern))
        if not os.path.isfile(dwi_file):
            continue
        dwi_files.append(dwi_file)
    assert len(cta_files) == len(dwi_files)

    max_d = 0
    max_h = 0
    max_w = 0
    infos = []
    for i in tqdm(range(len(cta_files))):
        cta_range, dwi_range = extract_valid_volume(cta_files[i], dwi_files[i])
        d = cta_range[1] - cta_range[0]
        h = cta_range[3] - cta_range[2]
        w = cta_range[5] - cta_range[4]
        if max_d < d:
            max_d = d
        if max_h < h:
            max_h = h
        if max_w < w:
            max_w = w
        info = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(os.path.basename(cta_files[i]), 
        os.path.basename(dwi_files[i]), 
        cta_range[0], cta_range[1], cta_range[2], cta_range[3], cta_range[4], cta_range[5])
        infos.append(info)
    print('\n ====> max range:\t[{}, {}, {}]\n'.format(max_d, max_h, max_w))
    dirname = os.path.dirname(out_config_file)
    os.makedirs(dirname, exist_ok=True)
    with open(out_config_file, 'w') as f:
        f.write('\n'.join(infos))
    print('\n save configuration to {}'.format(out_config_file))

def calculate_crop_range(in_config_file):
    min_d = 1000
    max_d = 0
    min_h = 1000
    max_h = 0
    min_w = 1000
    max_w = 0
    with open(in_config_file, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            ss = line.split('\t')
            if len(ss) < 8:
                continue
            min_z = int(ss[2])
            max_z = int(ss[3])
            min_y = int(ss[4])
            max_y = int(ss[5])
            min_x = int(ss[6])
            max_x = int(ss[7])
            d = max_z - min_z
            h = max_y - min_y
            w = max_x - min_x
            if max_d < d:
                max_d = d
            if min_d > d:
                min_d = d
            if max_h < h:
                max_h = h
            if min_h > h:
                min_h = h
            if max_w < w:
                max_w = w
            if min_w > w:
                min_w = w
    print('min depth: {}'.format(min_d))
    print('max depth: {}'.format(max_d))
    print('min height: {}'.format(min_h))
    print('max height: {}'.format(max_h))
    print('min width: {}'.format(min_w))
    print('max width: {}'.format(max_w))

def sitk_new_blank_image(size, spacing, direction, origin, default_value=0.):
    image = sitk.GetImageFromArray(np.ones(size, dtype=np.float).T * default_value)
    image.SetSpacing(spacing)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image

def sitk_resample_to_image(image, reference_image, interpolator, default_value=0., transform=None,
                           output_pixel_type=None):
    if transform is None:
        direction = image.GetDirection()
        if direction == [1,0,0,0,1,0,0,0,1]:
            transform = sitk.Transform()
            transform.SetIdentity()
        else:
            mat_direction = np.array(direction).reshape(3,3)
            inv_mat_direction = np.linalg.inv(mat_direction)
            inv_direction = inv_mat_direction.reshape(-1)
            transform = sitk.AffineTransform(3)
            transform.SetMatrix(mat_direction.ravel())
    if output_pixel_type is None:
        output_pixel_type = image.GetPixelID()
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(interpolator)
    resample_filter.SetTransform(transform)
    resample_filter.SetOutputDirection([1,0,0,0,1,0,0,0,1])
    resample_filter.SetOutputPixelType(output_pixel_type)
    resample_filter.SetDefaultPixelValue(default_value)
    resample_filter.SetReferenceImage(reference_image)
    return resample_filter.Execute(image)

def generate_data_propotional_cube(series_uid, scale, interpolator, output_pixel_type, is_dcm=True):
    if is_dcm:
        reader = sitk.ImageSeriesReader()
        filenames = reader.GetGDCMSeriesFileNames(series_uid)
        reader.SetFileNames(filenames)
        im = reader.Execute()
    else:
        im = sitk.ReadImage(series_uid)
    ori_spacing = im.GetSpacing()
    ori_size = im.GetSize()
    new_size = scale
    
    new_spacing = np.empty(3)
    for i in range(3):
        new_spacing[i] = ori_size[i] * ori_spacing[i] / new_size[i]

    black_im = sitk_new_blank_image(new_size, new_spacing, im.GetDirection(), im.GetOrigin())
    new_im = sitk_resample_to_image(im, black_im, interpolator, default_value=0, output_pixel_type=output_pixel_type)

    return new_im

def generate_data_propotional(series_uid, scale, interpolator, output_pixel_type, is_dcm=True):
    if is_dcm:
        reader = sitk.ImageSeriesReader()
        filenames = reader.GetGDCMSeriesFileNames(series_uid)
        reader.SetFileNames(filenames)
        im = reader.Execute()
    else:
        im = sitk.ReadImage(series_uid)
    ori_spacing = im.GetSpacing()
    ori_size = im.GetSize()
    new_size = [0,0,0]
    for i in range(3):
        new_size[i] = int(ori_spacing[i]*ori_size[i]/(ori_spacing[0]))
    new_spacing = [ori_spacing[0]]*3

    # interpolator = sitk.sitkNearestNeighbor

    black_im = sitk_new_blank_image(new_size, new_spacing, im.GetDirection(), im.GetOrigin())
    new_im = sitk_resample_to_image(im, black_im, interpolator, default_value=0, output_pixel_type=output_pixel_type)

    return new_im

def rescale_data_for_cta2dwi(indir, outdir, size=None):
    if size:
        size = [int(size), int(size), int(size)]
    if not os.path.isdir(indir):
        print('input dir not exist:{}'.format(indir))
    in_cta_dir = os.path.join(indir, 'CTA')
    in_dwi_dir = os.path.join(indir, 'DWI')
    out_cta_dir = os.path.join(outdir, 'CTA')
    out_dwi_dir = os.path.join(outdir, 'DWI')
    os.makedirs(out_cta_dir, exist_ok=True)
    os.makedirs(out_dwi_dir, exist_ok=True)
    
    cta_files = glob(os.path.join(in_cta_dir, '*CTA*.nii*'))
    dwi_files = []
    dwi_pattern = glob(os.path.join(in_dwi_dir, '*_DWI_*.nii*'))[0]
    dwi_pattern = os.path.basename(dwi_pattern)
    dwi_pattern = dwi_pattern.replace(dwi_pattern.split('_')[0], '')
    for cta_file in cta_files:
        if not os.path.isfile(cta_file):
            continue
        index = os.path.basename(cta_file).split('_')[0]
        dwi_file = os.path.join(in_dwi_dir, '{}{}'.format(index, dwi_pattern))
        if not os.path.isfile(dwi_file):
            continue
        dwi_files.append(dwi_file)
    assert len(cta_files) == len(dwi_files)

    for in_cta_file in tqdm(cta_files):
        # break
        if not size:
            out_cta_img = generate_data_propotional(in_cta_file, None, sitk.sitkLinear, sitk.sitkInt16, False)
        else:
            out_cta_img = generate_data_propotional_cube(in_cta_file, size, sitk.sitkLinear, sitk.sitkInt16, False)
        out_cta_file = os.path.join(out_cta_dir, os.path.basename(in_cta_file))
        sitk.WriteImage(out_cta_img, out_cta_file)


    for in_dwi_file in tqdm(dwi_files):
        # break
        if not size:
            out_dwi_img = generate_data_propotional(in_dwi_file, None, sitk.sitkLinear, sitk.sitkInt16, False)
        else:
            out_dwi_img = generate_data_propotional_cube(in_dwi_file, size, sitk.sitkLinear, sitk.sitkInt16, False)
        out_dwi_file = os.path.join(out_dwi_dir, os.path.basename(in_dwi_file))
        sitk.WriteImage(out_dwi_img, out_dwi_file)


    in_brain_dir = os.path.join(indir, 'cerebral_parenchyma')
    if not os.path.isdir(in_brain_dir):
        return
    out_brain_dir = os.path.join(outdir, 'cerebral_parenchyma')
    os.makedirs(out_brain_dir, exist_ok=True)
    brain_pattern = glob(os.path.join(in_brain_dir, '*_brain_*.nii*'))[0]
    brain_pattern = os.path.basename(brain_pattern)
    brain_pattern = brain_pattern.replace(brain_pattern.split('_')[0], '')
    brain_files = []
    for cta_file in cta_files:
        if not os.path.isfile(cta_file):
            continue
        index = os.path.basename(cta_file).split('_')[0]
        brain_file = os.path.join(in_brain_dir, '{}{}'.format(index, brain_pattern))
        if not os.path.isfile(brain_file):
            continue
        brain_files.append(brain_file)

    for in_brain_file in tqdm(brain_files):
        if not size:
            out_brain_img = generate_data_propotional(in_brain_file, None, sitk.sitkNearestNeighbor, sitk.sitkInt16, False)
        else:
            out_brain_img = generate_data_propotional_cube(in_brain_file, size, sitk.sitkNearestNeighbor, sitk.sitkInt16, False)
        out_brain_file = os.path.join(out_brain_dir, os.path.basename(in_brain_file))
        sitk.WriteImage(out_brain_img, out_brain_file)

    

def rescale_data_for_ncct2dwi(indir, outdir, size):
    if size:
        size = [int(size), int(size), int(size)]
    if not os.path.isdir(indir):
        print('input dir not exist:{}'.format(indir))
    in_cta_dir = os.path.join(indir, 'NCCT')
    in_dwi_dir = os.path.join(indir, 'DWI')
    out_cta_dir = os.path.join(outdir, 'NCCT')
    out_dwi_dir = os.path.join(outdir, 'DWI')
    os.makedirs(out_cta_dir, exist_ok=True)
    os.makedirs(out_dwi_dir, exist_ok=True)
    
    cta_files = glob(os.path.join(in_cta_dir, '*_NCCT.nii*'))
    dwi_files = []
    for cta_file in cta_files:
        if not os.path.isfile(cta_file):
            continue
        index = os.path.basename(cta_file).split('_')[0]
        dwi_file = os.path.join(in_dwi_dir, '{}_first_FU_DWI.nii.gz'.format(index))
        if not os.path.isfile(dwi_file):
            continue
        dwi_files.append(dwi_file)
    assert len(cta_files) == len(dwi_files)

    for in_cta_file in tqdm(cta_files):
        if not size:
            out_cta_img = generate_data_propotional(in_cta_file, None, sitk.sitkLinear, sitk.sitkInt16, False)
        else:
            out_cta_img = generate_data_propotional_cube(in_cta_file, size, sitk.sitkLinear, sitk.sitkInt16, False)
        out_cta_file = os.path.join(out_cta_dir, os.path.basename(in_cta_file))
        sitk.WriteImage(out_cta_img, out_cta_file)


    for in_dwi_file in tqdm(dwi_files):
        if not size:
            out_dwi_img = generate_data_propotional(in_dwi_file, None, sitk.sitkLinear, sitk.sitkInt16, False)
        else:
            out_dwi_img = generate_data_propotional_cube(in_dwi_file, size, sitk.sitkLinear, sitk.sitkInt16, False)
        out_dwi_file = os.path.join(out_dwi_dir, os.path.basename(in_dwi_file))
        sitk.WriteImage(out_dwi_img, out_dwi_file)


def utils_get_folder_pattern(indir, initpattern):
    files = glob(os.path.join(indir, initpattern))
    pattern = os.path.basename(files[0])
    pattern = pattern.replace(pattern.split('_')[0], '')
    return pattern

# 提取脑实质部分的mask, 所有层面都按照最大层进行mask运算
def generate_cerebral_parenchyma(indir, outdir, inpattern):
    os.makedirs(outdir, exist_ok=True)
    infiles = glob(os.path.join(indir, inpattern))
    # out_arr = np.array()
    for infile in tqdm(infiles):
        in_img = sitk.ReadImage(infile)
        in_arr = sitk.GetArrayFromImage(in_img)
        # print('z size:\t{}'.format(in_arr.shape[0]))
        out_arr = np.zeros(in_arr.shape, dtype=in_arr.dtype)
        for z in range(in_arr.shape[0]):
            for y in range(in_arr.shape[1]):
                x_arr = in_arr[z,y,:]
                low_thres = 0
                ranges = np.where(x_arr != low_thres)
                if len(ranges[0]) > 0:
                    [x_min] = np.min(ranges, axis=1)
                    [x_max] = np.max(ranges, axis=1)
                    out_arr[z,y,x_min:x_max+1] = 1
        # for z in range(in_arr.shape[0]):
        #     for x in range(in_arr.shape[2]):
        #         y_arr = in_arr[z,:,x]
        #         low_thres = 0
        #         ranges = np.where(y_arr != low_thres)
        #         if len(ranges[0]) > 0:
        #             [y_min] = np.min(ranges, axis=1)
        #             [y_max] = np.max(ranges, axis=1)
        #             out_arr[z,y_min:y_max+1, x] = 1
        
        # 在保留的断层中，mask区域扩大到和最大层面面积相等
        max_region = np.max(out_arr, axis=0)
        # 根据脑实质的区域大小，选择是否保留
        layers = np.sum(out_arr, axis=(1,2))
        for z in range(in_arr.shape[0]):
            if layers[z]/max(layers) < 0.3:
                out_arr[z,:,:] = 0
            else:
                out_arr[z,:,:] = max_region

        out_img = sitk.GetImageFromArray(out_arr)
        out_img.CopyInformation(in_img)
        out_file = os.path.join(outdir, os.path.basename(infile))
        writer = sitk.ImageFileWriter()
        writer.SetFileName(out_file)
        writer.Execute(out_img)
        # break

# 提取脑实质部分的mask, 进行腐蚀膨胀,已保证血管包含在内；对于脑的下半部分，没有脑实质，mask直接补0
def generate_cerebral_parenchyma_dilation(indir, outdir, inpattern):
    os.makedirs(outdir, exist_ok=True)
    infiles = glob(os.path.join(indir, inpattern))
    # out_arr = np.array()
    for infile in tqdm(infiles):
        in_img = sitk.ReadImage(infile)
        in_arr = sitk.GetArrayFromImage(in_img)
        # print('z size:\t{}'.format(in_arr.shape[0]))
        out_arr = np.zeros(in_arr.shape, dtype=in_arr.dtype)
        for z in range(in_arr.shape[0]):
            for y in range(in_arr.shape[1]):
                x_arr = in_arr[z,y,:]
                low_thres = 0
                ranges = np.where(x_arr != low_thres)
                if len(ranges[0]) > 0:
                    [x_min] = np.min(ranges, axis=1)
                    [x_max] = np.max(ranges, axis=1)
                    out_arr[z,y,x_min:x_max] = 1
        
        # 在保留的断层中，mask区域扩大到和最大层面面积相等
        max_region = np.max(out_arr, axis=0)
        # 根据脑实质的区域大小，选择是否保留
        layers = np.sum(out_arr, axis=(1,2))
        
        

        out_img = sitk.GetImageFromArray(out_arr)
        out_img.CopyInformation(in_img)

        # erode
        erode_filter = sitk.BinaryErodeImageFilter()
        erode_filter.SetForegroundValue(1)
        erode_filter.SetBackgroundValue(0)
        erode_filter.SetKernelRadius(2)
        out_img = erode_filter.Execute(out_img)

        # # dilation
        # dilation_filter = sitk.BinaryDilateImageFilter()
        # dilation_filter.SetForegroundValue(1)
        # dilation_filter.SetBackgroundValue(0)
        # dilation_filter.SetKernelRadius(3)
        # out_img = dilation_filter.Execute(out_img)

        out_arr = sitk.GetArrayFromImage(out_img)
        for z in range(in_arr.shape[0]):
            if layers[z] == 0:
                out_arr[z,:,:] = 0
        out_img = sitk.GetImageFromArray(out_arr)
        out_img.CopyInformation(in_img)

        out_file = os.path.join(outdir, os.path.basename(infile))
        writer = sitk.ImageFileWriter()
        writer.SetFileName(out_file)
        writer.Execute(out_img)
        # break


# 根据mask提取相应的区域并保存
def extract_region_by_mask(maskdir, srcdir, outdir, maskpattern, srcpattern):
    
    mask_files = glob(os.path.join(maskdir, maskpattern))
    src_pattern = glob(os.path.join(srcdir, srcpattern))[0]
    src_pattern = os.path.basename(src_pattern)
    src_pattern = src_pattern.replace(src_pattern.split('_')[0], '')
    src_files = []
    for mask_file in mask_files:
        if not os.path.isfile(mask_file):
            continue
        index = os.path.basename(mask_file).split('_')[0]
        src_file = os.path.join(srcdir, '{}{}'.format(index, src_pattern))
        if not os.path.isfile(src_file):
            continue
        src_files.append(src_file)
    assert(len(mask_files) == len(src_files))
    
    os.makedirs(outdir, exist_ok=True)
    for i in tqdm(range(len(mask_files))):
        assert(os.path.basename(mask_files[i]).split('_')[0] == os.path.basename(src_files[i]).split('_')[0])
        mask_file = mask_files[i]
        src_file = src_files[i]
        mask_img = sitk.ReadImage(mask_file)
        src_img = sitk.ReadImage(src_file)
        maskfilter = sitk.MaskImageFilter()
        maskfilter.SetOutsideValue(-1024)
        src_img = sitk.Cast(src_img, sitk.sitkInt16)
        mask_img = sitk.Cast(mask_img, sitk.sitkInt16)
        
        # print()
        # print(mask_file)
        # print(src_img.GetSize())
        # print(mask_img.GetSize())
        out_img = maskfilter.Execute(src_img, mask_img)
        # print(out_img.GetSize())
        # print()

        outfile = os.path.join(outdir, os.path.basename(src_files[i]))
        writerfilter = sitk.ImageFileWriter()
        writerfilter.SetFileName(outfile)
        writerfilter.Execute(out_img)


def extract_region_by_mask_cut_onecase(mask_file, src_file, outfile):
    '''
    debug cmd: extract_region_by_mask_cut_onecase('../data/gan/hospital_6/experiment_registration2/8.1.out/cerebral_parenchyma/1014186_first_BS_brain.nii.gz', '../data/gan/hospital_6/experiment_registration2/5 dwi_rigid_align_ncct/1014186_first_BS_NCCT.nii.gz', None)
    '''
    mask_img = sitk.ReadImage(mask_file)
    mask_arr = sitk.GetArrayFromImage(mask_img)

    mask_z_sum = np.sum(np.sum(mask_arr, axis=-1), axis=-1)

    ranges = np.where(mask_z_sum > 0)
    [z_min] = np.min(np.array(ranges), axis=1)
    [z_max] = np.max(np.array(ranges), axis=1)


    src_img = sitk.ReadImage(src_file)
    src_img = sitk.Cast(src_img, sitk.sitkInt16)
    src_arr = sitk.GetArrayFromImage(src_img)
    out_arr = src_arr[z_min:z_max+1, :, :]
    
    out_img = sitk.GetImageFromArray(out_arr)
    
    print(outfile)
    sitk.WriteImage(out_img, outfile)



def extract_region_by_mask_cut_singletask(mask_files, srcdir, outdir, maskpattern, srcpattern):
    src_pattern = glob(os.path.join(srcdir, srcpattern))[0]
    src_pattern = os.path.basename(src_pattern)
    src_pattern = src_pattern.replace(src_pattern.split('_')[0], '')
    
    for mask_file in tqdm(mask_files):
        if not os.path.isfile(mask_file):
            continue
        index = os.path.basename(mask_file).split('_')[0]
        src_file = os.path.join(srcdir, '{}{}'.format(index, src_pattern))
        if not os.path.isfile(src_file):
            continue
        outfile = os.path.join(outdir, os.path.basename(src_file))
        extract_region_by_mask_cut_onecase(mask_file, src_file, outfile)


def extract_region_by_mask_cut_multiprocess(maskdir, srcdir, outdir, maskpattern, srcpattern,  process_num=8):
    '''
    python cta_to_dwi_dataset.py extract_region_by_mask_cut_multiprocess '../data/gan/hospital_6/experiment_registration2/8.1.out/cerebral_parenchyma' '../data/gan/hospital_6/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_6/experiment_registration2/8.2.out/NCCT' *brain*.nii.gz *NCCT.nii.gz
    '''
    os.makedirs(outdir, exist_ok=True)

    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []

    mask_files = glob(os.path.join(maskdir, maskpattern))

    num_per_process = (len(mask_files) + process_num - 1)//process_num

    for i in range(process_num):
        sub_infiles = mask_files[num_per_process*i:min(num_per_process*(i+1), len(mask_files))]
        print(sub_infiles)
        result = pool.apply_async(extract_region_by_mask_cut_singletask, args=(sub_infiles,srcdir, outdir, maskpattern, srcpattern))
        results.append(result)

    pool.close()
    pool.join()



def extract_region_by_mask_cut_only_onecase(mask_file, src_file, outfile):
    '''
    debug cmd: extract_region_by_mask_cut_onecase('../data/gan/hospital_6/experiment_registration2/8.1.out/cerebral_parenchyma/1014186_first_BS_brain.nii.gz', '../data/gan/hospital_6/experiment_registration2/5 dwi_rigid_align_ncct/1014186_first_BS_NCCT.nii.gz', None)
    '''
    mask_img = sitk.ReadImage(mask_file)
    mask_arr = sitk.GetArrayFromImage(mask_img)

    mask_z_sum = np.sum(np.sum(mask_arr, axis=-1), axis=-1)

    ranges = np.where(mask_z_sum > 0)
    [z_min] = np.min(np.array(ranges), axis=1)
    [z_max] = np.max(np.array(ranges), axis=1)


    src_img = sitk.ReadImage(src_file)
    maskfilter = sitk.MaskImageFilter()
    maskfilter.SetOutsideValue(-1024)
    src_img = sitk.Cast(src_img, sitk.sitkInt16)
    mask_img = sitk.Cast(mask_img, sitk.sitkInt16)
    src_img = maskfilter.Execute(src_img, mask_img)
    src_arr = sitk.GetArrayFromImage(src_img)
    out_arr = src_arr[z_min:z_max+1, :, :]
    
    out_img = sitk.GetImageFromArray(out_arr)
    
    print(outfile)
    sitk.WriteImage(out_img, outfile)



def extract_region_by_mask_cut_only_singletask(mask_files, srcdir, outdir, maskpattern, srcpattern):
    src_pattern = glob(os.path.join(srcdir, srcpattern))[0]
    src_pattern = os.path.basename(src_pattern)
    src_pattern = src_pattern.replace(src_pattern.split('_')[0], '')
    
    for mask_file in tqdm(mask_files):
        if not os.path.isfile(mask_file):
            continue
        index = os.path.basename(mask_file).split('_')[0]
        src_file = os.path.join(srcdir, '{}{}'.format(index, src_pattern))
        if not os.path.isfile(src_file):
            continue
        outfile = os.path.join(outdir, os.path.basename(src_file))
        try:
            extract_region_by_mask_cut_only_onecase(mask_file, src_file, outfile)
        except Exception as e:
            print('{}:\t{}'.format(index, e))


def extract_region_by_mask_cut_only_multiprocess(maskdir, srcdir, outdir, maskpattern, srcpattern,  process_num=8):
    '''
    python cta_to_dwi_dataset.py extract_region_by_mask_cut_only_multiprocess '../data/gan/hospital_6/experiment_registration2/8.1.out/cerebral_parenchyma' '../data/gan/hospital_6/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_6/experiment_registration2/8.2.out/NCCT' *brain*.nii.gz *NCCT.nii.gz
    '''
    os.makedirs(outdir, exist_ok=True)

    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []

    mask_files = glob(os.path.join(maskdir, maskpattern))

    num_per_process = (len(mask_files) + process_num - 1)//process_num

    for i in range(process_num):
        sub_infiles = mask_files[num_per_process*i:min(num_per_process*(i+1), len(mask_files))]
        print(sub_infiles)
        result = pool.apply_async(extract_region_by_mask_cut_only_singletask, args=(sub_infiles,srcdir, outdir, maskpattern, srcpattern))
        results.append(result)

    pool.close()
    pool.join()


# 根据脑实质mask， 生成训练和测试的配置文件
def genereate_cta2dwi_config_file_with_cerebral_parenchyma(indir, configdir, train_ratio=0.8):
    '''
    目录结构， cerebral_parenchyma为脑实质mask文件
    .
    ├── cerebral_parenchyma
    ├── CTA
    └── DWI
    CTA 文件名格式: 4765366_first_BS_CTA_rigid_aligned.nii.gz 
    cerebral_parenchyma 文件名格式: 4765366_first_BS_brain_rigid_aligned.nii.gz
    DWI 文件名格式: 4765366_first_FU_DWI_rigid_aligned.nii.gz
    '''
    brain_files = glob(os.path.join(indir, 'cerebral_parenchyma', '*.nii.gz'))
    cta_files = []
    dwi_files = []
    cta_pattern = utils_get_folder_pattern(os.path.join(indir, 'CTA'), '*.nii.gz')
    dwi_pattern = utils_get_folder_pattern(os.path.join(indir, 'DWI'), '*.nii.gz')
    for i in range(len(brain_files)):
        if not os.path.isfile(brain_files[i]):
            continue
        index = os.path.basename(brain_files[i]).split('_')[0]
        cta_file = os.path.join(indir, 'CTA', '{}{}'.format(index, cta_pattern))
        dwi_file = os.path.join(indir, 'DWI', '{}{}'.format(index, dwi_pattern))
        if not os.path.isfile(cta_file):
            continue
        if not os.path.isfile(dwi_file):
            continue
        cta_files.append(cta_file)
        dwi_files.append(dwi_file)
    assert len(cta_files) == len(dwi_files) == len(brain_files)
    config_infos = []
    for i in tqdm(range(len(brain_files))):
        brain_img = sitk.ReadImage(brain_files[i])
        brain_arr = sitk.GetArrayFromImage(brain_img)
        ranges = np.where(brain_arr == 0)
        [z_min, y_min, x_min] = np.min(np.array(ranges), axis=1)
        [z_max, y_max, x_max] = np.max(np.array(ranges), axis=1)
        info = '{}\t{}\{}\t{}\t{}\t{}\{}\t{}'.format(
            os.path.basename(cta_files[i]), os.path.basename(dwi_files[i]), 
            z_min, z_max, y_min, y_max, x_min, x_max
        )
        
    np.random.shuffle(config_infos)
    pos = int(len(config_infos)*float(train_ratio))
    train_config_infos = config_infos[:pos]
    test_config_infos = config_infos[pos:]
    os.makedirs(configdir, exist_ok=True)
    train_config_file = os.path.join(configdir, 'train_config_file.txt')
    test_config_file = os.path.join(configdir, 'test_config_file.txt')
    with open(train_config_file, 'w') as f:
        f.write('\n'.join(train_config_infos))
    with open(test_config_file, 'w') as f:
        f.write('\n'.join(test_config_infos))
    

def test():
    infile = '../data/gan/ncct2dwi/experiment_registration2/1.nii_file/470933_first_BS_NCCT.nii.gz' 
    out_img = generate_data_propotional(infile, None, sitk.sitkNearestNeighbor, sitk.sitkInt16, False)   
    writer = sitk.ImageFileWriter()
    writer.SetFileName('../data/gan/ncct2dwi/experiment_registration2/tmp/xxx6.nii.gz') 
    writer.Execute(out_img)

class CTA2DWI_GAN_DS(Dataset):
    def __init__(self, root_dir, config_file, phase, crop_size, scale_size, debug=False):
        super().__init__()
        self.ct_list = []
        self.dwi_list = []
        self.root_dir = root_dir
        self.phase = phase
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.debug = debug
        with open(config_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                if len(ss) < 8:
                    continue
                ct_file = os.path.join(root_dir, 'CTA', ss[0])
                dwi_file = os.path.join(root_dir, 'DWI', ss[1])
                if not os.path.isfile(ct_file):
                    continue
                if not os.path.isfile(dwi_file):
                    continue
                d = int(ss[3])-int(ss[2])
                h = int(ss[5])-int(ss[4])
                w = int(ss[7])-int(ss[6])
                if d < self.crop_size[0] or h < self.crop_size[1] or w < self.crop_size[2]:
                    continue
                self.ct_list.append(ct_file)
                self.dwi_list.append(dwi_file)
    
    def __random_crop_data(self, ct_data, dwi_data, size):
        [img_d, img_h, img_w] = dwi_data.shape
        [input_d, input_h, input_w] = size
        # assert np.all(np.less_equal(size, dwi_data.shape))
        z_min_upper = img_d - input_d
        y_min_upper = img_h - input_h
        x_min_upper = img_w - input_w

        Z_min = np.random.randint(0, z_min_upper)
        Y_min = np.random.randint(0, y_min_upper)
        X_min = np.random.randint(0, x_min_upper)

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

        return ct_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], dwi_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __center_crop_data(self, ct_data, dwi_data, size):
        [img_d, img_h, img_w] = dwi_data.shape
        [input_d, input_h, input_w] = size
        # assert np.all(np.less_equal(size, dwi_data.shape))
        Z_min = img_d//2-input_d//2
        Y_min = img_h//2-input_h//2
        X_min = img_w//2-input_w//2
        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

        return ct_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], dwi_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __len__(self):
        return len(self.ct_list)

    def __getitem__(self, idx):
        if self.phase == 'train':
            ct_file = self.ct_list[idx]
            dwi_file = self.dwi_list[idx]
            ct_img = sitk.ReadImage(ct_file)
            ct_data = sitk.GetArrayFromImage(ct_img)
            dwi_img = sitk.ReadImage(dwi_file)
            dwi_data = sitk.GetArrayFromImage(dwi_img)
            if np.random.rand() < 0.8:
                cropped_ct, cropped_dwi = self.__random_crop_data(ct_data, dwi_data, self.crop_size)
            else:
                cropped_ct, cropped_dwi = self.__center_crop_data(ct_data, dwi_data, self.crop_size)

            if self.debug:
                mid_dir = os.path.join(self.root_dir, 'tmp')
                os.makedirs(mid_dir, exist_ok=True)
                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(mid_dir, 'ct_index_{}.nii.gz'.format(idx)))
                writer.Execute(sitk.GetImageFromArray(cropped_ct))

                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(mid_dir, 'dwi_index_{}.nii.gz'.format(idx)))
                writer.Execute(sitk.GetImageFromArray(cropped_dwi))

            cropped_ct = torch.from_numpy(cropped_ct).float()
            cropped_ct = torch.unsqueeze(cropped_ct, axis=0)
            cropped_dwi = torch.from_numpy(cropped_dwi).float()
            cropped_dwi = torch.unsqueeze(cropped_dwi, axis=0)
            return cropped_ct, cropped_dwi, os.path.basename(ct_file), os.path.basename(dwi_file)

def check_CTA2DWI_GAN_DS_middle_result():
    ds = CTA2DWI_GAN_DS('../data/gan/cta2dwi/case_178_forHuang_rescale', 
    '../data/gan/cta2dwi/case_178_forHuang_rescale/config/config_file_1.txt', 
    'train', [160, 256, 224], [160, 256, 224], debug=True)
    dataloader = DataLoader(ds, num_workers=2, batch_size=1, pin_memory=True)
    for index, _ in enumerate(dataloader):
        print(index)

class NCCT2DWI_GAN_DS(Dataset):
    def __init__(self, root_dir, config_file, phase, crop_size, scale_size, debug=False):
        super().__init__()
        self.ct_list = []
        self.dwi_list = []
        self.root_dir = root_dir
        self.phase = phase
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.debug = debug
        with open(config_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                if len(ss) < 8:
                    continue
                ct_file = os.path.join(root_dir, 'NCCT', ss[0])
                dwi_file = os.path.join(root_dir, 'DWI', ss[1])
                if not os.path.isfile(ct_file):
                    continue
                if not os.path.isfile(dwi_file):
                    continue
                d = int(ss[3])-int(ss[2])
                h = int(ss[5])-int(ss[4])
                w = int(ss[7])-int(ss[6])
                if d < self.crop_size[0] or h < self.crop_size[1] or w < self.crop_size[2]:
                    continue
                self.ct_list.append(ct_file)
                self.dwi_list.append(dwi_file)
    
    def __random_crop_data(self, ct_data, dwi_data, size):
        [img_d, img_h, img_w] = dwi_data.shape
        [input_d, input_h, input_w] = size
        # assert np.all(np.less_equal(size, dwi_data.shape))
        z_min_upper = img_d - input_d
        y_min_upper = img_h - input_h
        x_min_upper = img_w - input_w

        Z_min = np.random.randint(0, z_min_upper)
        Y_min = np.random.randint(0, y_min_upper)
        X_min = np.random.randint(0, x_min_upper)

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

        return ct_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], dwi_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __center_crop_data(self, ct_data, dwi_data, size):
        [img_d, img_h, img_w] = dwi_data.shape
        [input_d, input_h, input_w] = size
        # assert np.all(np.less_equal(size, dwi_data.shape))
        Z_min = img_d//2-input_d//2
        Y_min = img_h//2-input_h//2
        X_min = img_w//2-input_w//2
        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

        return ct_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], dwi_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __len__(self):
        return len(self.ct_list)

    def __getitem__(self, idx):
        if self.phase == 'train':
            ct_file = self.ct_list[idx]
            dwi_file = self.dwi_list[idx]
            ct_img = sitk.ReadImage(ct_file)
            ct_data = sitk.GetArrayFromImage(ct_img)
            dwi_img = sitk.ReadImage(dwi_file)
            dwi_data = sitk.GetArrayFromImage(dwi_img)
            if np.random.rand() < 0.8:
                cropped_ct, cropped_dwi = self.__random_crop_data(ct_data, dwi_data, self.crop_size)
            else:
                cropped_ct, cropped_dwi = self.__center_crop_data(ct_data, dwi_data, self.crop_size)

            if self.debug:
                mid_dir = os.path.join(self.root_dir, 'tmp')
                os.makedirs(mid_dir, exist_ok=True)
                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(mid_dir, 'ct_index_{}.nii.gz'.format(idx)))
                writer.Execute(sitk.GetImageFromArray(cropped_ct))

                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(mid_dir, 'dwi_index_{}.nii.gz'.format(idx)))
                writer.Execute(sitk.GetImageFromArray(cropped_dwi))

            cropped_ct = torch.from_numpy(cropped_ct).float()
            cropped_ct = torch.unsqueeze(cropped_ct, axis=0)
            cropped_dwi = torch.from_numpy(cropped_dwi).float()
            cropped_dwi = torch.unsqueeze(cropped_dwi, axis=0)
            return cropped_ct, cropped_dwi, os.path.basename(ct_file), os.path.basename(dwi_file)



if __name__ == '__main__':
    fire.Fire()
    # test_extract_valid_volume()
    # rescale_data_for_cta2dwi('../data/gan/cta2dwi/case_178_forHuang', '../data/gan/cta2dwi/case_178_forHuang_rescale')
    # generate_cerebral_parenchyma('../data/gan/cta2dwi/experiment_data1/rigid_registration/cerebral_parenchyma', '../data/gan/cta2dwi/experiment_data1/rigid_registration_mask/cerebral_parenchyma', '*_brain_rigid_aligned.nii.gz')
    # rescale_data_for_cta2dwi('../data/gan/cta2dwi/experiment_data1/rigid_registration', '../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_256', [256, 256, 256])
    # generate_cerebral_parenchyma('../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512/cerebral_parenchyma', '../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512_mask/cerebral_parenchyma', '*_brain_rigid_aligned.nii.gz')
    # extract_region_by_mask('../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_mask/cerebral_parenchyma', '../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale/CTA', '../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_mask/CTA', '*brain*.nii.gz', '*CTA*.nii.gz')
    # extract_region_by_mask('../data/gan/ncct2dwi/experiment_registration2/8.out/cerebral_parenchyma', '../data/gan/ncct2dwi/experiment_registration2/5 dwi_rigid_align_ncct', '../data/gan/ncct2dwi/experiment_registration2/8.out/NCCT', '*brain*.nii.gz', '*NCCT.nii.gz')
    # extract_region_by_mask_cut_onecase('../data/gan/hospital_6/experiment_registration2/8.1.out/cerebral_parenchyma/1014186_first_BS_brain.nii.gz', '../data/gan/hospital_6/experiment_registration2/5 dwi_rigid_align_ncct/1014186_first_BS_NCCT.nii.gz', None)