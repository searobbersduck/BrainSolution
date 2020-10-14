import os
import sys

import skimage
import skimage.measure

import SimpleITK as sitk

from glob import glob

import numpy as np
import pandas as pd

from tqdm import tqdm


def mae(img1, img2):
    mae = np.mean( abs(img1 - img2)  )
    return mae 

real_dwi_file = '../data/gan/hospital_6/experiment_registration2/10.predict_retain/REAL_DWI/1014186_first_FU_DWI_BXXX.nii.gz'
fake_dwi_file = '../data/gan/hospital_6/experiment_registration2/10.predict_retain/FAKE_DWI/1014186_first_BS_NCCT_fake.nii.gz'

def calc_metrics_one_case(real_dwi_file, fake_dwi_file):
    print('====> processing {}'.format(real_dwi_file))
    real_dwi_img = sitk.ReadImage(real_dwi_file)
    fake_dwi_img = sitk.ReadImage(fake_dwi_file)

    real_dwi_arr = sitk.GetArrayFromImage(real_dwi_img)
    fake_dwi_arr = sitk.GetArrayFromImage(fake_dwi_img)

    fake_dwi_arr = fake_dwi_arr[:real_dwi_arr.shape[0], :, :]

    wl = 200
    ww = 400
    min_v = wl-ww//2
    max_v = wl+ww//2
    real_dwi_arr = np.clip(real_dwi_arr, min_v, max_v)
    real_dwi_arr = (real_dwi_arr-min_v)/ww*255

    fake_dwi_arr = np.clip(fake_dwi_arr, min_v, max_v)
    fake_dwi_arr = (fake_dwi_arr-min_v)/ww*255

    real_dwi_arr = np.array(real_dwi_arr, dtype=np.int16)
    fake_dwi_arr = np.array(fake_dwi_arr, dtype=np.int16)

    psnr = skimage.measure.compare_psnr(real_dwi_arr, fake_dwi_arr)
    ssim = skimage.measure.compare_ssim(real_dwi_arr, fake_dwi_arr)
    mse = skimage.measure.compare_mse(real_dwi_arr, fake_dwi_arr)
    nrmse = skimage.measure.compare_nrmse(real_dwi_arr, fake_dwi_arr)
    mae_metrics = mae(real_dwi_arr, fake_dwi_arr)

    print('psnr:{:.3f}, ssim:{:.3f}, mse:{:.3f}, nrmse:{:.3f}, mae_metrics:{:.3f}'.format(psnr, ssim, mse, nrmse, mae_metrics))

    return psnr, ssim, mse, nrmse, mae_metrics, real_dwi_arr.min(), real_dwi_arr.max()


def calc_metrics_all(indir, outdir):
    real_dir = os.path.join(indir, 'REAL_DWI')
    fake_dir = os.path.join(indir, 'FAKE_DWI')
    
    real_list = glob(os.path.join(real_dir, '*DWI_BXXX.nii.gz'))
    patient_ids = [os.path.basename(i).split('_')[0] for i in real_list]
    
    row_elems = []
    for pid in tqdm(patient_ids):
        real_dwi_file = os.path.join(real_dir, '{}_first_FU_DWI_BXXX.nii.gz'.format(pid))
        fake_dwi_file = os.path.join(fake_dir, '{}_first_BS_NCCT_fake.nii.gz'.format(pid))
        psnr, ssim, mse, nrmse, mae, min_v, max_v = calc_metrics_one_case(real_dwi_file, fake_dwi_file)
        row_elems.append(np.array([pid, psnr, ssim, mse, nrmse, mae, min_v, max_v]))
    df = pd.DataFrame(np.array(row_elems), columns=['pid', 'psnr', 'ssim', 'mse', 'nrmse', 'mae', 'min_v', 'max_v'])
    df.to_csv('./cta2dwi_metrics_norm.csv')
    
def calc_metrics_all_2d(indir, outdir):
    real_dir = os.path.join(indir, 'REAL_DWI')
    fake_dir = os.path.join(indir, 'FAKE_DWI_2D')
    
    real_list = glob(os.path.join(real_dir, '*DWI_BXXX.nii.gz'))
    patient_ids = [os.path.basename(i).split('_')[0] for i in real_list]
    
    row_elems = []
    for pid in tqdm(patient_ids):
        real_dwi_file = os.path.join(real_dir, '{}_first_FU_DWI_BXXX.nii.gz'.format(pid))
        fake_dwi_file = os.path.join(fake_dir, '{}_first_BS_NCCT_fake_2d.nii.gz'.format(pid))
        psnr, ssim, mse, nrmse, mae, min_v, max_v = calc_metrics_one_case(real_dwi_file, fake_dwi_file)
        row_elems.append(np.array([pid, psnr, ssim, mse, nrmse, mae, min_v, max_v]))
    df = pd.DataFrame(np.array(row_elems), columns=['pid', 'psnr', 'ssim', 'mse', 'nrmse', 'mae', 'min_v', 'max_v'])
    df.to_csv('./cta2dwi_metrics_norm_2d.csv')

if __name__ == '__main__':
    # calc_metrics_all('../data/gan/hospital_6/experiment_registration2/10.predict_retain', None)
    calc_metrics_all_2d('../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/tmp', None)
