import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))

print(os.path.dirname(__file__))

import SimpleITK as sitk
import numpy as np
from glob import glob
import shutil
from tqdm import tqdm

import fire

from gan.datasets.ncct_gan_dataset import global_ncct_error_list


def split_3d_to_2d_slice_onecase(file_a, file_b, pid, out_dir):
    '''
    note:
        file_a:     (format: nii.gz/nii), source file
        file_b:     (format: nii.gz/nii), target file
        id:         task id
        out_dir:    output dir
    '''
    img_a = sitk.ReadImage(file_a)
    img_b = sitk.ReadImage(file_b)
    
    arr_a = sitk.GetArrayFromImage(img_a)
    arr_b = sitk.GetArrayFromImage(img_b)

    out_sub_dir = os.path.join(out_dir, pid)
    os.makedirs(out_sub_dir, exist_ok=True)
    print(out_sub_dir)
    for iz in range(arr_a.shape[0]):
        out_file_a = os.path.join(out_sub_dir, '{}_{}_a.npy'.format(pid, iz))
        out_file_b = os.path.join(out_sub_dir, '{}_{}_b.npy'.format(pid, iz))

        np.save(out_file_a, arr_a[iz])
        np.save(out_file_b, arr_b[iz])


def cta_gan_split_3d_to_2d_slice_singletask(cta_files, dwi_files, out_dir):
    for i in tqdm(range(len(cta_files))):
        pid = os.path.basename(cta_files[i]).split('_')[0]
        split_3d_to_2d_slice_onecase(cta_files[i], dwi_files[i], pid, out_dir)


def cta_gan_split_3d_to_2d_slice_multiprocess(config_file, data_root, out_dir, process_num=12):
    '''
    config_file:        ../../data/gan/hospital_6/experiment_registration2/8.2.out/config/anno_mask_ncct_to_dwi_bxxx_train_config_file.txt
    data_root:          ../../data/gan/hospital_6/experiment_registration2/8.2.out

    debug cmd:          cta_gan_split_3d_to_2d_slice_multiprocess('../../data/gan/hospital_6/experiment_registration2/8.2.out/config/anno_mask_ncct_to_dwi_bxxx_train_config_file.txt', '../../data/gan/hospital_6/experiment_registration2/8.2.out', '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/train')
    invoke cmd:         python data_processing.py cta_gan_split_3d_to_2d_slice_multiprocess '../../data/gan/hospital_6/experiment_registration2/8.2.out/config/anno_mask_ncct_to_dwi_bxxx_train_config_file.txt' '../../data/gan/hospital_6/experiment_registration2/8.2.out' '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/train'
    '''
    cta_dir = os.path.join(data_root, 'NCCT')
    dwi_dir = os.path.join(data_root, 'DWI_BXXX')
    
    cta_files = []
    dwi_files = []
    with open(config_file) as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            ss = line.split('\t')
            cta_file = os.path.join(data_root, ss[0])
            dwi_file = os.path.join(data_root, ss[1])
            pid = os.path.basename(cta_file).split('_')[0]
            if pid in global_ncct_error_list:
                continue
            cta_files.append(cta_file)
            dwi_files.append(dwi_file)
    
    print(len(cta_files))


    # cta_gan_split_3d_to_2d_slice_singletask(cta_files, dwi_files, out_dir)

    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []

    num_per_process = (len(cta_files) + process_num - 1)//process_num

    os.makedirs(out_dir, exist_ok=True)
    for i in range(process_num):
        sub_cta_files = cta_files[num_per_process*i:min(num_per_process*(i+1), len(cta_files))]
        sub_dwi_files = dwi_files[num_per_process*i:min(num_per_process*(i+1), len(dwi_files))]
        print(sub_cta_files)
        print(sub_dwi_files)
        result = pool.apply_async(cta_gan_split_3d_to_2d_slice_singletask, args=(sub_cta_files, sub_dwi_files, out_dir))
        results.append(result)

    pool.close()
    pool.join() 

def generate_config_file(in_2d_dir, out_config_dir, postfix=''):
    '''
    debug cmd:      generate_config_file('../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/train', '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/config', 'train')
    invoke cmd:     python data_processing.py generate_config_file '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/train' '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/config' 'train'
    '''
    os.makedirs(out_config_dir, exist_ok=True)
    pairs = []
    for sub_folder in os.listdir(in_2d_dir):
        sub_in_2d_dir = os.path.join(in_2d_dir, sub_folder)
        image_list = glob(os.path.join(sub_in_2d_dir, '*_a.npy'))
        for file_a in image_list:
            file_b = file_a.replace('_a', '_b')
            if not os.path.isfile(file_a):
                continue
            if not os.path.isfile(file_b):
                continue
            tmp_str = os.path.basename(sub_in_2d_dir)
            pairs.append('{}\t{}'.format(os.path.join(tmp_str, os.path.basename(file_a)), os.path.join(tmp_str, os.path.basename(file_b))))

    os.makedirs(out_config_dir, exist_ok=True)
    out_config_file = os.path.join(out_config_dir, 'cta_to_dwi_2d_{}.txt'.format(postfix))
    with open(out_config_file, 'w') as f:
        f.write('\n'.join(pairs))



if __name__ == '__main__':
    fire.Fire()
    # cta_gan_split_3d_to_2d_slice_multiprocess('../../data/gan/hospital_6/experiment_registration2/8.2.out/config/anno_mask_ncct_to_dwi_bxxx_train_config_file.txt', '../../data/gan/hospital_6/experiment_registration2/8.2.out', '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/train')
    # generate_config_file('../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/train', '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/config', 'train')
    # print('hello world!')