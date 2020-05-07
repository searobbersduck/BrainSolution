'''
@Description: 
@Version: 1.0
@Autor: searobbersanduck
@Date: 2020-04-09 09:52:50
@LastEditors: searobbersanduck
@LastEditTime: 2020-05-06 11:59:39
@License : (C)Copyright 2020-2021, MIT
'''

import os
from glob import glob
import numpy as np
import fire
import SimpleITK as sitk
from tqdm import tqdm
import time
import pydicom
import shutil
import pandas as pd
import shutil
import csv
import pandas as pd

import sys
# print(os.path.join(os.path.dirname(__file__), os.path.pardir))
# sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
abs_dir = os.getcwd()
work_dir = os.path.abspath(os.path.join(abs_dir,os.path.pardir))
sys.path.append(work_dir)
work_dir = os.path.abspath(os.path.join(abs_dir,os.path.pardir,os.path.pardir))
sys.path.append(work_dir)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from cerebral_parenchyma.train.train import inference, extract_region_by_mask, extract_region_by_mask1

# 1. extract_exam_info
# todo:


# 2. extract ncct series by exam info file & convert *.dcm files to *.nii.gz files
import xlrd
import datetime
import SimpleITK as sitk

def ncct_extract_from_hospital_folder(in_folder, out_folder):
    '''
    ncct_extract_from_hospital_folder('../data/gan/hospital_4/CT+灌注/2018住院1/GENERAL', '../data/gan/hospital_4/0.ori')
    '''
    '''
    in_folder文件夹形式如下: 
    .
    ├── 1.2.528.1.1001.200.10.4357.2081.1.20200419060625507
    │   ├── DICOMDIR
    │   └── SDY00000
    ├── 1.2.528.1.1001.200.10.4357.2081.1.20200419060643899
    │   ├── DICOMDIR
    │   └── SDY00000
    ├── 1.2.528.1.1001.200.10.4357.2081.1.20200419060738842
    │   ├── DICOMDIR
    │   └── SDY00000
    └── 1.2.528.1.1001.200.10.4401.4133.1.20200419060247414
        ├── DICOMDIR
        └── SDY00000
            ├── SRS00000
            ├── SRS00001
            ├── SRS00002
            ├── SRS00003
            ├── SRS00004
            ├── SRS00005
            ├── SRS00006
            ├── SRS00007
            ├── SRS00008
            ├── SRS00009
            ├── SRS00010
            ├── SRS00011
            ├── SRS00012
            ├── SRS00013
            ├── SRS00014
            ├── SRS00015
            ├── SRS00016
            ├── SRS00017
            ├── SRS00018
            ├── SRS00019
            ├── SRS00020
            ├── SRS00021
            ├── SRS00022
            ├── SRS00023
            └── SRS00024
    '''

    for patient_folder1 in tqdm(os.listdir(in_folder)):
        patient_folder1 = os.path.join(in_folder, patient_folder1)
        if not os.path.isdir(patient_folder1):
            continue
        patient_folder2s = glob(os.path.join(patient_folder1, 'SDY*'))
        for patient_folder2 in patient_folder2s:
            if not os.path.isdir(patient_folder2):
                continue
            series_folders = glob(os.path.join(patient_folder2, 'SRS*'))
            for series_folder in series_folders:
                if not os.path.isdir(series_folder):
                    continue
                dcm_files = glob(os.path.join(series_folder, '*.DCM'))
                for dcm_file in dcm_files:
                    metadata = pydicom.dcmread(dcm_file)
                    patient_id = metadata.PatientID
                    series_uid = metadata.SeriesInstanceUID
                    
                    outdir = os.path.join(out_folder, patient_id, series_uid)
                    os.makedirs(outdir, exist_ok=True)
                    outfile = os.path.join(out_folder, patient_id, series_uid, os.path.basename(dcm_file))
                    shutil.copyfile(dcm_file, outfile)

def ncct_extract_from_hospital_folder_all():
    '''
    .
    ├── 2017-住院12月
    ├── 2017-门诊
    ├── 2018住院1
    ├── 2018住院2
    ├── 2018住院3
    └── 2019-门诊

    '''
    out_dir = '../data/gan/hospital_4/0.ori'
    os.makedirs(out_dir, exist_ok=True)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4/CT+灌注/2017-住院12月/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4/CT+灌注/2018住院1/GENERAL', out_dir)
    # ncct_extract_from_hospital_folder('../data/gan/hospital_4/CT+灌注/2018住院2/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4/CT+灌注/2018住院2', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4/CT+灌注/2018住院3/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4/CT+灌注/2019-门诊/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4/CT+灌注/2017-门诊', out_dir)


def ncct_extract_from_hospital4_2_folder_all():
    '''
    查看2层目录
    .
    ├── CT+灌注
    │   ├── 2017-住院10月
    │   ├── 2017-住院4月
    │   ├── 2017-住院5月
    │   ├── 2017-住院6月
    │   ├── 2017-住院7月
    │   ├── 2017-住院8月
    │   ├── 2017-住院9月
    │   ├── 2019-住院aaa
    │   ├── 2019-住院bbb
    │   ├── 2019-住院ccc
    │   └── 2019-住院ddd
    └── CT+灌注1
        ├── 2017-住院11月
        ├── 2017-住院1月
        ├── 2017-住院2月
        ├── 2017-住院3月
        ├── 2018-门诊1
        ├── 2018-门诊2
        ├── 2019-住院1
        ├── 2019-住院2
        ├── 2019-门诊2
        ├── 2019-门诊3
        └── 2019-门诊4


    查看3层目录：
    .
    ├── CT+灌注
    │   ├── 2017-住院10月
    │   │   ├── GENERAL
    │   │   ├── local.ldb
    │   │   └── local.mdb
    │   ├── 2017-住院4月
    │   │   ├── GENERAL
    │   │   ├── local.ldb
    │   │   └── local.mdb
    │   ├── 2017-住院5月
    │   │   ├── GENERAL
    │   │   ├── local.ldb
    │   │   └── local.mdb
    │   ├── 2017-住院6月
    │   │   ├── GENERAL
    │   │   ├── local.ldb
    │   │   └── local.mdb
    │   ├── 2017-住院7月
    │   │   ├── GENERAL
    │   │   ├── local.ldb
    │   │   └── local.mdb
    │   ├── 2017-住院8月
    │   │   ├── GENERAL
    │   │   ├── local.ldb
    │   │   └── local.mdb
    │   ├── 2017-住院9月
    │   │   ├── GENERAL
    │   │   ├── local.ldb
    │   │   └── local.mdb
    │   ├── 2019-住院aaa
    │   │   └── LOCAL
    │   ├── 2019-住院bbb
    │   │   └── LOCAL
    │   ├── 2019-住院ccc
    │   │   └── LOCAL
    │   └── 2019-住院ddd
    │       └── LOCAL
    └── CT+灌注1
        ├── 2017-住院11月
        │   ├── GENERAL
        │   ├── local.ldb
        │   └── local.mdb
        ├── 2017-住院1月
        │   ├── GENERAL
        │   ├── local.ldb
        │   └── local.mdb
        ├── 2017-住院2月
        │   ├── GENERAL
        │   ├── local.ldb
        │   └── local.mdb
        ├── 2017-住院3月
        │   ├── GENERAL
        │   ├── local.ldb
        │   └── local.mdb
        ├── 2018-门诊1
        │   ├── GENERAL
        │   ├── local.ldb
        │   └── local.mdb
        ├── 2018-门诊2
        │   └── LOCAL
        ├── 2019-住院1
        │   └── LOCAL
        ├── 2019-住院2
        │   └── LOCAL
        ├── 2019-门诊2
        │   └── LOCAL
        ├── 2019-门诊3
        │   └── LOCAL
        └── 2019-门诊4
            └── LOCAL
   

    '''
    out_dir = '../data/gan/hospital_4_2/0.ori'
    os.makedirs(out_dir, exist_ok=True)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注/2017-住院10月/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注/2017-住院4月/GENERAL', out_dir)
    # ncct_extract_from_hospital_folder('../data/gan/hospital_4/CT+灌注/2018住院2/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注/2017-住院5月/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注/2017-住院6月/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注/2017-住院7月/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注/2017-住院8月/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注/2017-住院9月/GENERAL', out_dir)

    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注/2019-住院aaa/LOCAL/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注/2019-住院bbb/LOCAL/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注/2019-住院ccc/LOCAL/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注/2019-住院ddd/LOCAL/GENERAL', out_dir)


    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注1/2017-住院11月/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注1/2017-住院1月/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注1/2017-住院2月/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注1/2017-住院3月/GENERAL', out_dir)

    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注1/2018-门诊1/GENERAL', out_dir)

    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注1/2018-门诊2/LOCAL/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注1/2019-住院1/LOCAL/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注1/2019-住院2/LOCAL/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注1/2019-门诊2/LOCAL/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注1/2019-门诊3/LOCAL/GENERAL', out_dir)
    ncct_extract_from_hospital_folder('../data/gan/hospital_4_2/CT+灌注1/2019-门诊4/LOCAL/GENERAL', out_dir)

def ncct_generate_table_all_series_one_patient(patient_path):
    infos = []
    for series_uid in os.listdir(patient_path):
        series_path = os.path.join(patient_path, series_uid)
        if not os.path.isdir(series_path):
            continue
        dcm_files1 = glob(os.path.join(series_path, '*.DCM'))
        dcm_files2 = glob(os.path.join(series_path, '*.dcm'))
        dcm_files = dcm_files1 + dcm_files2
        dcm_files.sort()
        dcm_file = dcm_files[0]
        metadata = pydicom.dcmread(dcm_file)
        infos.append(metadata)
    return infos

def ncct_generate_table_all_series(in_dir, out_dir):
    '''
    ncct_generate_table_all_series('../data/gan/hospital_4/0.ori', '../data/gan/hospital_4/0.table')
    ncct_generate_table_all_series('../data/gan/hospital_4_2/0.ori', '../data/gan/hospital_4_2/0.table')
    '''
    os.makedirs(out_dir, exist_ok=True)
    row_elems = []
    for patient_id in os.listdir(in_dir):
        patient_path = os.path.join(in_dir, patient_id)
        infos = ncct_generate_table_all_series_one_patient(patient_path)
        for info in infos:
            series_uid = info.SeriesInstanceUID if 'SeriesInstanceUID' in info else ''
            desc = info.SeriesDescription if 'SeriesDescription' in info else ''
            modality = info.Modality if 'Modality' in info else ''
            age = info.PatientAge if 'PatientAge' in info else ''
            sex = info.PatientSex if 'PatientSex' in info else ''
            pos = info.PatientPosition if 'PatientPosition' in info else ''
            pid = info.PatientID if 'PatientID' in info else ''
            study_desc = info.StudyDescription if 'StudyDescription' in info else ''
            study_uid = info.StudyInstanceUID if 'StudyInstanceUID' in info else ''
            if 'AcquisitionDate' in info and 'AcquisitionTime' in info:
                acq_time = info.AcquisitionDate + info.AcquisitionTime
            elif 'AcquisitionDateTime' in info:
                acq_time = info.AcquisitionDateTime
            else:
                acq_time = ''
            acq_time = acq_time.split('.')[0]
            
            row_elems.append(np.array([pid, study_uid, series_uid, study_desc, desc, modality, age, sex, pos, acq_time]))
            # pd.DataFrame(np.array(row_elems), columns=['pid', 'study_uid', 'series_uid', 'study_desc', 'desc', 'modality', 'age', 'sex', 'pos', 'acq_time'])
    df = pd.DataFrame(np.array(row_elems), columns=['pid', 'study_uid', 'series_uid', 'study_desc', 'desc', 'modality', 'age', 'sex', 'pos', 'acq_time'])
    df.to_csv(os.path.join(out_dir, 'hospital_total.csv'))


def ncct_extract_info_from_patient(patient_path, interval=10800):
    '''
    ncct_extract_info_from_patient('../data/gan/hospital_4/0.ori/332594')
    '''
    infos = {}
    infos['DWI'] = []
    infos['ADC'] = []
    infos['NCCT'] = []
    infos['RAPID'] = []
    infos['PID'] = os.path.basename(patient_path)
    selected_infos = None
    for series_uid in os.listdir(patient_path):
        series_path = os.path.join(patient_path, series_uid)
        if not os.path.isdir(series_path):
            continue
        dcm_files1 = glob(os.path.join(series_path, '*.DCM'))
        dcm_files2 = glob(os.path.join(series_path, '*.dcm'))
        dcm_files = dcm_files1 + dcm_files2
        if len(dcm_files) < 7:
            continue
        dcm_files.sort()
        dcm_file = dcm_files[0]
        metadata = pydicom.dcmread(dcm_file)
        study_uid = metadata.StudyInstanceUID
        # print(metadata.Modality)
        image_type = metadata.ImageType
        if ('DIFFUSION' in image_type and 'TRACEW' in image_type):
            dwi_series_uid = series_uid
            dwi_acq_time = metadata.AcquisitionDate + metadata.AcquisitionTime
            info = {}
            info['series_uid'] = series_uid
            info['acq_time'] = dwi_acq_time
            info['study_uid'] = study_uid
            infos['DWI'].append(info)
            continue
        if ('DIFFUSION' in image_type and 'ADC' in image_type):
            adc_series_uid = series_uid
            adc_acq_time = metadata.AcquisitionDate + metadata.AcquisitionTime
            info = {}
            info['series_uid'] = series_uid
            info['acq_time'] = adc_acq_time
            info['study_uid'] = study_uid
            infos['ADC'].append(info)
            continue
        if ('RAPID Summary Outputs' in metadata.SeriesDescription):
            info = {}
            info['series_uid'] = series_uid
            infos['RAPID'] = info
            info['study_uid'] = study_uid
            continue
        
        # series_uid = series_uid
        # acq_time = metadata.AcquisitionDate + metadata.AcquisitionTime
        # info = {}
        # info['series_uid'] = series_uid
        # info['acq_time'] = acq_time

        if ('CT' in metadata.Modality):
            if 'ConvolutionKernel' in metadata and 's' == metadata.ConvolutionKernel[-1]:
                series_uid = series_uid
                acq_time = metadata.AcquisitionDate + metadata.AcquisitionTime
                info = {}
                info['series_uid'] = series_uid
                info['acq_time'] = acq_time
                info['study_uid'] = study_uid
                info['slice_thickness'] = metadata.SliceThickness
                infos['NCCT'].append(info)
    
    # 合并NCCT同次扫描的不同层厚的NCCT
    if len(infos['NCCT']) > 1:
        ncct_infos = []
        # 1. 合并同次检查，保留层厚较小的序列, todo: 这里可以作为参数设置
        tmp_study_uids = {}
        for ncct_info in infos['NCCT']:
            if ncct_info['study_uid'] in tmp_study_uids:
                tmp_study_uids[ncct_info['study_uid']] += 1
            else:
                tmp_study_uids[ncct_info['study_uid']] = 1
        if len(tmp_study_uids) != len(infos['NCCT']):
            tmp_study_uid = None
            for key, val in tmp_study_uids.items():
                if val > 1:
                    tmp_study_uid = key
            tmp_ncct_info = None
            tmp_ncct_thickness = None
            for ncct_info in infos['NCCT']:
                if ncct_info['study_uid'] == tmp_study_uid:
                    if tmp_ncct_info is None:
                        tmp_ncct_thickness = ncct_info['slice_thickness']
                        tmp_ncct_info = ncct_info.copy()
                    else:
                        if float(tmp_ncct_thickness) > float(ncct_info['slice_thickness']):
                            tmp_ncct_thickness = ncct_info['slice_thickness']
                            tmp_ncct_info = ncct_info.copy()
                else:
                    ncct_infos.append(ncct_info)
            ncct_infos.append(tmp_ncct_info)
            infos['NCCT'] = ncct_infos
    
    # 如果CT数据只有一例，那么选取时间间隔在两小时以内，dwi在ct采集之后，如果有多例选择最后一例
    if len(infos['NCCT']) == 1:
        if len(infos['DWI']) == 1:
            ncct_acq_time = infos['NCCT'][0]['acq_time']
            dwi_acq_time = infos['DWI'][0]['acq_time']
            ncct_acq_time = datetime.datetime.strptime(ncct_acq_time.split('.')[0], '%Y%m%d%H%M%S')
            dwi_acq_time = datetime.datetime.strptime(dwi_acq_time.split('.')[0], '%Y%m%d%H%M%S')
            scan_interval_threshold = datetime.timedelta(seconds=interval)
            # scan_interval_threshold = datetime.timedelta(seconds=86400)
            zero_interval_threshold = datetime.timedelta(seconds=0)
            ncct_study_uid = infos['NCCT'][0]['study_uid']
            dwi_study_uid = infos['DWI'][0]['study_uid']
            
            if ((dwi_acq_time - ncct_acq_time) < scan_interval_threshold and (dwi_acq_time - ncct_acq_time) > zero_interval_threshold):
                selected_infos = infos
            else:
                selected_infos = None
        elif len(infos['DWI']) >= 2:
            dwi_infos = infos['DWI']
            dwi_infos_bk = []
            # 找到所有dwi与ncct时间间隔小于3小时并且大于0的
            for dwi_info in dwi_infos:
                ncct_acq_time = infos['NCCT'][0]['acq_time']
                dwi_acq_time = dwi_info['acq_time']
                ncct_acq_time = datetime.datetime.strptime(ncct_acq_time.split('.')[0], '%Y%m%d%H%M%S')
                dwi_acq_time = datetime.datetime.strptime(dwi_acq_time.split('.')[0], '%Y%m%d%H%M%S')
                scan_interval_threshold = datetime.timedelta(seconds=interval)
                # scan_interval_threshold = datetime.timedelta(seconds=86400)
                zero_interval_threshold = datetime.timedelta(seconds=0)
                ncct_study_uid = infos['NCCT'][0]['study_uid']
                dwi_study_uid = dwi_info['study_uid']
                if ((dwi_acq_time - ncct_acq_time) < scan_interval_threshold and  (dwi_acq_time - ncct_acq_time) > zero_interval_threshold):
                    dwi_infos_bk.append(dwi_info)
            # 在所有三个小时以内的数据中，挑选时间更靠后的，因为两次时间贴的特别近的dwi有可能其中一次是废片
            infos_bk = {}
            infos_bk = infos.copy()
            infos_bk['DWI'] = []
            cur_dwi_time = None
            cur_dwi_info = None
            for dwi_info in dwi_infos_bk:
                dwi_acq_time = dwi_info['acq_time']
                dwi_acq_time = datetime.datetime.strptime(dwi_acq_time.split('.')[0], '%Y%m%d%H%M%S')
                if cur_dwi_time is None:
                    cur_dwi_time = dwi_acq_time
                    cur_dwi_info = dwi_info
                else:
                    if cur_dwi_time < dwi_acq_time:
                        cur_dwi_time = dwi_acq_time
                        cur_dwi_info = dwi_info
            if cur_dwi_info is not None:
                infos_bk['DWI'].append(cur_dwi_info)
            infos_bk['ADC'] = []
            if len(infos_bk['DWI']) > 0:
                for adc_info in infos['ADC']:
                    adc_acq_time = adc_info['acq_time']
                    adc_acq_time = datetime.datetime.strptime(adc_acq_time.split('.')[0], '%Y%m%d%H%M%S')
                    if adc_acq_time > cur_dwi_time:
                        infos_bk['ADC'].append(adc_info)
            selected_infos = infos_bk

    # 在所有NCCT数据中筛选层厚更厚的那个
    
    # 在所有NCCT数据中筛选时间差
    if len(infos['NCCT']) > 2:
        for ncct_id in infos['NCCT']:
            ncct_acq_time = ncct_id['acq_time']
            ncct_acq_time_t = datetime.datetime.strptime(ncct_acq_time.split('.')[0], '%Y%m%d%H%M%S')
            for dwi_id in infos['DWI']:
                dwi_acq_time = dwi_id['acq_time']
                dwi_acq_time_t = datetime.datetime.strptime(dwi_acq_time.split('.')[0], '%Y%m%d%H%M%S')
                delta_t = dwi_acq_time_t - ncct_acq_time_t
                print(delta_t)


    return infos, selected_infos
    # print('hello world!')       

def ncct_extract_infos_from_patients_all(in_dir, out_dir):
    '''
    ncct_extract_infos_from_patients_all('../data/gan/hospital_4/0.ori', '../data/gan/hospital_4/0.raw_dcm')
    '''
    ncct_cnt = 0
    all_select_infos = []
    for patient_id in os.listdir(in_dir):
        patient_path = os.path.join(in_dir, patient_id)
        infos, select_infos = ncct_extract_info_from_patient(patient_path)
        # if len(infos['DWI']) > 1 and len(infos['ADC']) > 0 and len(infos['NCCT']) > 0:
        #     print(infos)
        # if len(infos['NCCT']) == 2:
        #     ncct_cnt += 1
        if select_infos is not None and len(select_infos['DWI']) > 0:
            all_select_infos.append(select_infos)
            # print(select_infos)
    # print(ncct_cnt)
    row_elems = []
    for select_info in tqdm(all_select_infos):
        patient_id = select_info['PID']
        ncct_uid = select_info['NCCT'][0]['series_uid']
        ncct_time = select_info['NCCT'][0]['acq_time']
        ncct_time_t = datetime.datetime.strptime(ncct_time.split('.')[0], '%Y%m%d%H%M%S')
        dwi_uid = select_info['DWI'][0]['series_uid']
        dwi_time = select_info['DWI'][0]['acq_time']
        dwi_time_t = datetime.datetime.strptime(dwi_time.split('.')[0], '%Y%m%d%H%M%S')
        adc_uid = select_info['ADC'][0]['series_uid']
        adc_time = select_info['ADC'][0]['acq_time']
        adc_time_t = datetime.datetime.strptime(adc_time.split('.')[0], '%Y%m%d%H%M%S')
        if len(select_info['RAPID']) == 0:
            rapid_uid = ''
        else:
            rapid_uid = select_info['RAPID']['series_uid']
        study_uid = select_info['DWI'][0]['study_uid']
        delta_dwi_ncct = dwi_time_t - ncct_time_t
        delta_adc_ncct = adc_time_t - ncct_time_t
        # print(str(ncct_time_t))
        row_elems.append(np.array([patient_id, study_uid, ncct_uid, str(ncct_time_t), dwi_uid, str(dwi_time_t), adc_uid, str(adc_time_t), rapid_uid, str(delta_dwi_ncct), str(delta_adc_ncct)]))
    df = pd.DataFrame(np.array(row_elems), columns=['patient_id', 'study_uid', 'ncct_uid', 'ncct_time_t', 'dwi_uid', 'dwi_time_t', 'adc_uid', 'adc_time_t', 'rapid_uid', 'delta_dwi_ncct', 'delta_adc_ncct'])
    df.to_csv('test_3h.csv')

        
    # for select_info in tqdm(all_select_infos):
    #     patient_uid = select_info['PID']

    #     if len(select_info['DWI']) > 0:
    #         src_series = os.path.join(in_dir, patient_uid, select_info['DWI'][0]['series_uid'])
    #         dst_series = os.path.join(out_dir, patient_uid, 'DWI', select_info['DWI'][0]['series_uid'])
    #         os.makedirs(os.path.dirname(dst_series), exist_ok=True)
    #         ncct_extract_dwi_from_raw_dwi_single(src_series, dst_series)
    #         # shutil.copytree(src_series, dst_series)

    #     if len(select_info['ADC']) > 0:
    #         src_series = os.path.join(in_dir, patient_uid, select_info['ADC'][0]['series_uid'])
    #         dst_series = os.path.join(out_dir, patient_uid, 'ADC', select_info['ADC'][0]['series_uid'])
    #         os.makedirs(os.path.dirname(dst_series), exist_ok=True)
    #         shutil.copytree(src_series, dst_series)

    #     if len(select_info['NCCT']) > 0:
    #         src_series = os.path.join(in_dir, patient_uid, select_info['NCCT'][0]['series_uid'])
    #         dst_series = os.path.join(out_dir, patient_uid, 'NCCT', select_info['NCCT'][0]['series_uid'])
    #         os.makedirs(os.path.dirname(dst_series), exist_ok=True)
    #         shutil.copytree(src_series, dst_series)

    #     if len(select_info['RAPID']) > 0:
    #         src_series = os.path.join(in_dir, patient_uid, select_info['RAPID']['series_uid'])
    #         dst_series = os.path.join(out_dir, patient_uid, 'RAPID', select_info['RAPID']['series_uid'])
    #         os.makedirs(os.path.dirname(dst_series), exist_ok=True)
    #         shutil.copytree(src_series, dst_series)
    print('====> finish ncct_extract_infos_from_patients_all!\n\n')

def ncct_generate_table_all_ncct_dwi_adc_pairs(in_dir, out_dir, interval_time):
    '''
    ncct_generate_table_all_ncct_dwi_adc_pairs('../data/gan/hospital_4/0.ori', '../data/gan/hospital_4/0.table')
    ncct_generate_table_all_ncct_dwi_adc_pairs('../data/gan/hospital_4_2/0.ori', '../data/gan/hospital_4_2/0.table')
    '''
    interval_time = int(interval_time)
    os.makedirs(out_dir, exist_ok=True)
    ncct_cnt = 0
    all_select_infos = []
    for patient_id in os.listdir(in_dir):
        patient_path = os.path.join(in_dir, patient_id)
        infos, select_infos = ncct_extract_info_from_patient(patient_path, interval_time)
        # if len(infos['DWI']) > 1 and len(infos['ADC']) > 0 and len(infos['NCCT']) > 0:
        #     print(infos)
        # if len(infos['NCCT']) == 2:
        #     ncct_cnt += 1
        if select_infos is not None and len(select_infos['DWI']) > 0:
            all_select_infos.append(select_infos)
            # print(select_infos)
    # print(ncct_cnt)
    row_elems = []
    for select_info in tqdm(all_select_infos):
        patient_id = select_info['PID']
        ncct_uid = select_info['NCCT'][0]['series_uid']
        ncct_time = select_info['NCCT'][0]['acq_time']
        ncct_time_t = datetime.datetime.strptime(ncct_time.split('.')[0], '%Y%m%d%H%M%S')
        dwi_uid = select_info['DWI'][0]['series_uid']
        dwi_time = select_info['DWI'][0]['acq_time']
        dwi_time_t = datetime.datetime.strptime(dwi_time.split('.')[0], '%Y%m%d%H%M%S')
        adc_uid = select_info['ADC'][0]['series_uid']
        adc_time = select_info['ADC'][0]['acq_time']
        adc_time_t = datetime.datetime.strptime(adc_time.split('.')[0], '%Y%m%d%H%M%S')
        if len(select_info['RAPID']) == 0:
            rapid_uid = ''
        else:
            rapid_uid = select_info['RAPID']['series_uid']
        study_uid = select_info['DWI'][0]['study_uid']
        delta_dwi_ncct = dwi_time_t - ncct_time_t
        delta_adc_ncct = adc_time_t - ncct_time_t
        # print(str(ncct_time_t))
        row_elems.append(np.array([patient_id, study_uid, ncct_uid, str(ncct_time_t), dwi_uid, str(dwi_time_t), adc_uid, str(adc_time_t), rapid_uid, str(delta_dwi_ncct), str(delta_adc_ncct)]))
    df = pd.DataFrame(np.array(row_elems), columns=['patient_id', 'study_uid', 'ncct_uid', 'ncct_time_t', 'dwi_uid', 'dwi_time_t', 'adc_uid', 'adc_time_t', 'rapid_uid', 'delta_dwi_ncct', 'delta_adc_ncct'])
    df.to_csv(os.path.join(out_dir, 'ncct_dwi_adc_pairs_{}s.csv'.format(interval_time)))

def ncct_extract_infos_from_xlsx(info_file):
    '''
    info_file = '../data/gan/ncct2dwi/experiment_registration1/config/V1 四院NCCT-DWI-ADC-RAPID.xlsx'
    invoke cmd: python utils.py extract_infos_from_xlsx '../data/gan/ncct2dwi/experiment_registration1/config/V1 四院NCCT-DWI-ADC-RAPID.xlsx'
    debug cmd: extract_infos_from_xlsx('../data/gan/ncct2dwi/experiment_registration1/config/V1 四院NCCT-DWI-ADC-RAPID.xlsx')
    
    file head: [text:'来源医院', text:'\taccesionNumber (院方ID)', 
    text:'PID (杏脉ID)', text:' Anonymized_PID', 
    text:'NCCT  Series Instance UID ', 
    text:'DWI Series Instance UID ', 
    text:'ADC Series Instance UID ', 
    text:'RAPID DWI-PWI Summary  Series Instance UID ', 
    text:'NCCT Acquisition DateTime', 
    text:'DWI Acquisition DateTime', 
    text:'ADC Acquisition DateTime', 
    text:'RAPID Study DateTime', 
    text:'时间间隔 DWI - NCCT time', 
    text:'时间间隔 ADC - NCCT time', 
    text:'时间间隔 RAPID - NCCT time'] 
    '''
    
    wb = xlrd.open_workbook(info_file)
    sheet_names = wb.sheet_names()

    ws = wb.sheet_by_index(0)

    # print('{}\t{}\t{}'.format(ws.name, ws.nrows, ws.ncols))

    pid_index = ws.row_values(0).index('PID (杏脉ID)')
    ncct_index = ws.row_values(0).index('NCCT  Series Instance UID ')
    dwi_index = ws.row_values(0).index('DWI Series Instance UID ')
    adc_index = ws.row_values(0).index('ADC Series Instance UID ')
    dwi_ncct_interval_index = ws.row_values(0).index('时间间隔 DWI - NCCT time')
    adc_ncct_interval_index = ws.row_values(0).index('时间间隔 ADC - NCCT time')
    ncct_time_index =  ws.row_values(0).index('NCCT Acquisition DateTime')
    dwi_time_index =  ws.row_values(0).index('DWI Acquisition DateTime')
    adc_time_index =  ws.row_values(0).index('ADC Acquisition DateTime')

    # 统计病人数量， 如果一个病人有多个扫描序列，只保存扫描时间间隔最短的病例
    pid_dict = {}
    patient_infos = {}
    for i_r in range(1,ws.nrows):
        pid = ws.row_values(i_r)[pid_index]
        if len(pid) == 0 or pid is None:
            continue

        patient_info = {}
        if pid in patient_infos:
            patient_info = patient_infos[pid]
        
        ncct_series = ws.row_values(i_r)[ncct_index]
        dwi_series = ws.row_values(i_r)[dwi_index]
        adc_series = ws.row_values(i_r)[adc_index]
        ncct_time = ws.row_values(i_r)[ncct_time_index]
        dwi_time = ws.row_values(i_r)[dwi_time_index]
        adc_time = ws.row_values(i_r)[adc_time_index]
        
        ncct_time = datetime.datetime.strptime(ncct_time, '%Y%m%d%H%M%S')
        dwi_time = datetime.datetime.strptime(dwi_time, '%Y%m%d%H%M%S')
        adc_time = datetime.datetime.strptime(adc_time, '%Y%m%d%H%M%S')

        delta_ncct2dwi_time = dwi_time - ncct_time
        delta_ncct2adc_time = adc_time - ncct_time

        if 'ncct_series' not in patient_info:
            patient_info['ncct_series'] = ncct_series
        
        if 'delta_ncct2dwi_time' in patient_info:
            if delta_ncct2dwi_time < patient_info['delta_ncct2dwi_time']:
                patient_info['delta_ncct2dwi_time'] = delta_ncct2dwi_time
                patient_info['dwi_series'] = dwi_series
        else:
            patient_info['delta_ncct2dwi_time'] = delta_ncct2dwi_time
            patient_info['dwi_series'] = dwi_series

        if 'delta_ncct2adc_time' in patient_info:
            if delta_ncct2adc_time < patient_info['delta_ncct2adc_time']:
                patient_info['delta_ncct2adc_time'] = delta_ncct2adc_time
                patient_info['adc_series'] = adc_series
        else:
            patient_info['delta_ncct2adc_time'] = delta_ncct2adc_time
            patient_info['adc_series'] = adc_series

        patient_infos[pid] = patient_info

    print('patient count:\t{}'.format(len(patient_infos.keys())))
    
    # 去除扫描时间超过三个小时的病例
    scan_interval_threshold = datetime.timedelta(seconds=10800)
    to_delete_patient_keys = []
    for key in patient_infos.keys():
        patient_info = patient_infos[key]
        if patient_info['delta_ncct2dwi_time'] > scan_interval_threshold or patient_info['delta_ncct2adc_time'] > scan_interval_threshold:
            to_delete_patient_keys.append(key)
    for key in to_delete_patient_keys:
        patient_infos.pop(key)
    print('patient which scan time less 2 hous count :\t{}'.format(len(patient_infos.keys())))
    return patient_infos

def ncct_convert_dcm_to_niigz_single(dcm_path, out_file, is_oblique=False):
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    reader = sitk.ImageSeriesReader()
    dicomfilenames = reader.GetGDCMSeriesFileNames(dcm_path)
    reader.SetFileNames(dicomfilenames)
    image = reader.Execute()

    if is_oblique:
        beg_image = dicomfilenames[0]
        end_image = dicomfilenames[-1]
        beg_metadata = pydicom.dcmread(beg_image)
        end_metadata = pydicom.dcmread(end_image)
        z_direction = np.array(beg_metadata.ImagePositionPatient)-np.array(end_metadata.ImagePositionPatient)
        z_direction = z_direction/np.linalg.norm(z_direction)
        direction = image.GetDirection()
        new_direction = np.array([0]*9).astype(np.float64)
        new_direction[:6] = np.array(direction)[:6]
        new_direction[6:] = np.abs(z_direction)

        old_size = image.GetSize() # x,y,z
        new_arr = np.full(old_size, -1024)

        ori_direction = new_direction
        ori_direction = np.reshape(ori_direction, (3,3))
        spacing = image.GetSpacing()
        ori_direction = ori_direction*np.array(spacing)
        
        ori_arr = sitk.GetArrayFromImage(image)
        new_arr = np.copy(ori_arr)
        shape = new_arr.shape

        for i in tqdm(range(shape[2])):
            for j in range(shape[1]):
                for k in range(shape[0]):
                    zuobiao = np.array([i,j,k])
                    new_zuobiao = np.round(np.dot(ori_direction, zuobiao))/np.array(spacing)
                    new_zuobiao = new_zuobiao.astype(np.int32)
                    #print(zuobiao, new_zuobiao)
                    if new_zuobiao[0]<0 or new_zuobiao[1]<0 or new_zuobiao[2]<0 or new_zuobiao[0]>shape[2]-1 or new_zuobiao[1]>shape[1]-1 or new_zuobiao[2]>shape[0]-1:
                        new_arr[k,j,i] = -1024
                    else:
                        new_arr[k,j,i] = ori_arr[new_zuobiao[2], new_zuobiao[1], new_zuobiao[0]]

        new_image = sitk.GetImageFromArray(new_arr)
        new_image.CopyInformation(image)
        new_image.SetOrigin([0,0,0])
        new_image.SetSpacing(image.GetSpacing())
        new_image.SetDirection(new_direction)

        image = new_image
        
    # image.SetDirection(tuple(new_direction))

    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_file)
    writer.Execute(image)

def ncct_convert_dcm_to_niigz(indir, outdir):
    '''
    indir = '../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm'
    outdir = '../data/gan/ncct2dwi/experiment_registration2/1.nii_file'

    invoke cmd: python utils.py ncct_convert_dcm_to_niigz '../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm' '../data/gan/ncct2dwi/experiment_registration2/1.nii_file'
    debug cmd: ncct_convert_dcm_to_niigz('../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm', '../data/gan/ncct2dwi/experiment_registration2/1.nii_file')
    
    '''
    os.makedirs(outdir, exist_ok=True)

    patient_ids = os.listdir(indir)
    ncct_convert_dcm_to_niigz_onecase(indir, patient_ids, outdir)
    # for patient_id in tqdm(patient_ids):
    #     print(patient_id)
    #     index = patient_id
    #     patient_id = os.path.join(indir, patient_id)
    #     ncct_path = os.path.join(patient_id, 'NCCT')
    #     adc_path = os.path.join(patient_id, 'ADC')
    #     dwi_path = os.path.join(patient_id, 'DWI')
    #     if not os.path.isdir(ncct_path):
    #         continue
    #     if not os.path.isdir(adc_path):
    #         continue
    #     if not os.path.isdir(dwi_path):
    #         continue
    #     ncct_path = os.path.join(ncct_path, os.listdir(ncct_path)[0])
    #     adc_path = os.path.join(adc_path, os.listdir(adc_path)[0])
    #     dwi_path = os.path.join(dwi_path, os.listdir(dwi_path)[0])
    #     if not os.path.isdir(ncct_path):
    #         continue
    #     if not os.path.isdir(adc_path):
    #         continue
    #     if not os.path.isdir(dwi_path):
    #         continue
    #     out_ncct_file = os.path.join(outdir, '{}_first_BS_NCCT.nii.gz'.format(index))
    #     out_adc_file = os.path.join(outdir, '{}_first_FU_ADC.nii.gz'.format(index))

    #     ncct_convert_dcm_to_niigz_single(ncct_path, out_ncct_file, True)
    #     ncct_convert_dcm_to_niigz_single(adc_path, out_adc_file)

    #     b0_path = os.path.join(dwi_path, 'b0')
    #     bxxx_path = os.path.join(dwi_path, 'bxxx')
    #     if not os.path.isdir(b0_path):
    #         continue
    #     if not os.path.isdir(bxxx_path):
    #         continue
    #     out_b0_file = os.path.join(outdir, '{}_first_FU_DWI_B0.nii.gz'.format(index))
    #     out_bxxx_file = os.path.join(outdir, '{}_first_FU_DWI_BXXX.nii.gz'.format(index))
    #     ncct_convert_dcm_to_niigz_single(b0_path, out_b0_file)
    #     ncct_convert_dcm_to_niigz_single(bxxx_path, out_bxxx_file)

def ncct_convert_dcm_to_niigz_onecase(indir, patient_ids, outdir):
    for patient_id in tqdm(patient_ids):
        # print(patient_id)
        index = patient_id
        patient_id = os.path.join(indir, patient_id)
        ncct_path = os.path.join(patient_id, 'NCCT')
        adc_path = os.path.join(patient_id, 'ADC')
        dwi_path = os.path.join(patient_id, 'DWI')
        if not os.path.isdir(ncct_path):
            print('ncct_path:\t{}'.format(ncct_path))
            continue
        if not os.path.isdir(adc_path):
            print('adc_path:\t{}'.format(adc_path))
            continue
        if not os.path.isdir(dwi_path):
            print('dwi_path:\t{}'.format(dwi_path))
            continue
        ncct_path = os.path.join(ncct_path, os.listdir(ncct_path)[0])
        adc_path = os.path.join(adc_path, os.listdir(adc_path)[0])
        dwi_path = os.path.join(dwi_path, os.listdir(dwi_path)[0])
        if not os.path.isdir(ncct_path):
            continue
        if not os.path.isdir(adc_path):
            continue
        if not os.path.isdir(dwi_path):
            continue
        out_ncct_file = os.path.join(outdir, '{}_first_BS_NCCT.nii.gz'.format(index))
        out_adc_file = os.path.join(outdir, '{}_first_FU_ADC.nii.gz'.format(index))

        ncct_convert_dcm_to_niigz_single(ncct_path, out_ncct_file, True)
        ncct_convert_dcm_to_niigz_single(adc_path, out_adc_file)

        b0_path = os.path.join(dwi_path, 'b0')
        bxxx_path = os.path.join(dwi_path, 'bxxx')
        if not os.path.isdir(b0_path):
            print('b0_path:\t{}'.format(b0_path))
            continue
        if not os.path.isdir(bxxx_path):
            print('bxxx_path:\t{}'.format(bxxx_path))
            continue
        out_b0_file = os.path.join(outdir, '{}_first_FU_DWI_B0.nii.gz'.format(index))
        out_bxxx_file = os.path.join(outdir, '{}_first_FU_DWI_BXXX.nii.gz'.format(index))
        ncct_convert_dcm_to_niigz_single(b0_path, out_b0_file)
        ncct_convert_dcm_to_niigz_single(bxxx_path, out_bxxx_file)

def ncct_convert_dcm_to_niigz_multiprocess(indir, outdir, process_num=24):
    '''
    indir = '../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm'
    outdir = '../data/gan/ncct2dwi/experiment_registration2/1.nii_file'

    invoke cmd: python utils.py ncct_convert_dcm_to_niigz_multiprocessing '../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm' '../data/gan/ncct2dwi/experiment_registration2/1.nii_file'
    debug cmd: ncct_convert_dcm_to_niigz_multiprocessing('../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm', '../data/gan/ncct2dwi/experiment_registration2/1.nii_file')
    
    '''
    os.makedirs(outdir, exist_ok=True)

    patient_ids = os.listdir(indir)
    # ncct_convert_dcm_to_niigz_onecase(indir, patient_ids, outdir)

    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []

    num_per_process = (len(patient_ids) + process_num - 1)//process_num

    for i in range(process_num):
        sub_infiles = patient_ids[num_per_process*i:min(num_per_process*(i+1), len(patient_ids)-1)]
        print(sub_infiles)
        result = pool.apply_async(ncct_convert_dcm_to_niigz_onecase, args=(indir, sub_infiles, outdir))
        results.append(result)

    pool.close()
    pool.join()

    

def ncct_extract_dwi_dcm_from_raw_series_single(in_dcm_path, out_dcm_path):
    pass

def reset_dcm_info(in_dcm_path, out_dcm_path, is_dcm=False):
    '''
    debug cmd: reset_dcm_info('../data/gan/ncct2dwi/siyuan_dcm_with_pid/137611/1.3.12.2.1107.5.1.4.95874.30000016121100222306600009841', '/ssd2/zhangwd/data/brain/gan/ncct2dwi/experiment_registration1/tmp/test3', True)
    '''
    if is_dcm:
        series_reader = sitk.ImageSeriesReader()
        dicomfilenames = series_reader.GetGDCMSeriesFileNames(in_dcm_path)
        series_reader.SetFileNames(dicomfilenames)

        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()

        image = series_reader.Execute()
    else:
        image = sitk.ReadImage(in_dcm_path)


    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    tags_to_copy = ["0010|0010", # Patient Name
                    "0010|0020", # Patient ID
                    "0010|0030", # Patient Birth Date
                    "0020|000D", # Study Instance UID, for machine consumption
                    "0020|0010", # Study ID, for human consumption
                    "0008|0020", # Study Date
                    "0008|0030", # Study Time
                    "0008|0050", # Accession Number
                    "0008|0060"  # Modality
    ]

    
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    filtered_image = image
    uni_direction = tuple([1,0,0,0,1,0,0,0,1])
    uni_origin = tuple([0,0,0])
    filtered_image.SetOrigin(uni_origin)
    # filtered_image.SetDirection(uni_direction)   

    direction = filtered_image.GetDirection()

    series_tag_values = [(k, series_reader.GetMetaData(0,k)) for k in tags_to_copy if series_reader.HasMetaDataKey(0,k)] + \
                    [("0008|0031",modification_time), # Series Time
                    ("0008|0021",modification_date), # Series Date
                    ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                    ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                    ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                        direction[1],direction[4],direction[7])))),
                    ("0008|103e", series_reader.GetMetaData(0,"0008|103e") + " Processed-SimpleITK")] # Series Description


    os.makedirs(out_dcm_path, exist_ok=True)
    for i in range(filtered_image.GetDepth()):
        image_slice = filtered_image[:,:,i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str,filtered_image.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i)) # Instance Number
        image_slice.SetMetaData("0020|000D", str('asdfasdfasdfasdf')) # Instance Number

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(os.path.join(out_dcm_path,str(i)+'.dcm'))
        writer.Execute(image_slice)

def convert_nii_to_dcm(in_dcm_path, out_dcm_path, is_dcm=False):
    '''
    debug cmd: convert_nii_to_dcm('../data/gan/ncct2dwi/experiment_registration2/test/470933_first_BS_NCCT.nii.gz', '/ssd2/zhangwd/data/brain/gan/ncct2dwi/experiment_registration2/tmp/test3', False)
    '''
    if is_dcm:
        series_reader = sitk.ImageSeriesReader()
        dicomfilenames = series_reader.GetGDCMSeriesFileNames(in_dcm_path)
        series_reader.SetFileNames(dicomfilenames)

        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()

        image = series_reader.Execute()
    else:
        image = sitk.ReadImage(in_dcm_path)


    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    tags_to_copy = ["0010|0010", # Patient Name
                    "0010|0020", # Patient ID
                    "0010|0030", # Patient Birth Date
                    "0020|000D", # Study Instance UID, for machine consumption
                    "0020|0010", # Study ID, for human consumption
                    "0008|0020", # Study Date
                    "0008|0030", # Study Time
                    "0008|0050", # Accession Number
                    "0008|0060"  # Modality
    ]

    
    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")

    filtered_image = image
    uni_direction = tuple([1,0,0,0,1,0,0,0,1])
    uni_origin = tuple([0,0,0])
    filtered_image.SetOrigin(uni_origin)
    # filtered_image.SetDirection(uni_direction)   

    direction = filtered_image.GetDirection()

    series_tag_values = [(k, image.GetMetaData(k)) for k in tags_to_copy if image.HasMetaDataKey(k)] + \
                    [("0008|0031",modification_time), # Series Time
                    ("0008|0021",modification_date), # Series Date
                    ("0008|0008","DERIVED\\SECONDARY"), # Image Type
                    ("0020|000e", "1.2.826.0.1.3680043.2.1125."+modification_date+".1"+modification_time), # Series Instance UID
                    ("0020|0037", '\\'.join(map(str, (direction[0], direction[3], direction[6],# Image Orientation (Patient)
                                                        direction[1],direction[4],direction[7])))),
                    ("0008|103e", " Processed-SimpleITK")] # Series Description


    os.makedirs(out_dcm_path, exist_ok=True)
    for i in range(filtered_image.GetDepth()):
        image_slice = filtered_image[:,:,i]
        # Tags shared by the series.
        for tag, value in series_tag_values:
            image_slice.SetMetaData(tag, value)
        # Slice specific tags.
        image_slice.SetMetaData("0008|0012", time.strftime("%Y%m%d")) # Instance Creation Date
        image_slice.SetMetaData("0008|0013", time.strftime("%H%M%S")) # Instance Creation Time
        image_slice.SetMetaData("0020|0032", '\\'.join(map(str,filtered_image.TransformIndexToPhysicalPoint((0,0,i))))) # Image Position (Patient)
        image_slice.SetMetaData("0020|0013", str(i)) # Instance Number
        image_slice.SetMetaData("0020|000D", str('asdfasdfasdfasdf')) # Instance Number

        # Write to the output directory and add the extension dcm, to force writing in DICOM format.
        writer.SetFileName(os.path.join(out_dcm_path,str(i)+'.dcm'))
        writer.Execute(image_slice)


def ncct_extract_dwi_from_raw_dwi_single(in_dwi_path, out_dwi_path):
    #  os.makedirs(out_dwi_path, exist_ok=True)
    in_files1 = glob(os.path.join(in_dwi_path, '*.dcm'))
    in_files2 = glob(os.path.join(in_dwi_path, '*.DCM'))
    in_files = in_files1 + in_files2
    not_dwi_files = []
    dwi_b0_files = []
    dwi_bxxx_files = []
    for in_file in in_files:
        metadata = pydicom.dcmread(in_file)
        # ['DERIVED', 'PRIMARY', 'DIFFUSION', 'NONE', 'TRACEW', 'ND']
        image_type = metadata.ImageType
        if not('DIFFUSION' in image_type and 'TRACEW' in image_type):
            not_dwi_files.append(in_file)
            continue
        try:
            seq_name = metadata.SequenceName
            if 'b0' in seq_name:
                dwi_b0_files.append(in_file)
            elif 'b1000' in seq_name:
                dwi_bxxx_files.append(in_file)
        except:
            pass
    assert len(in_files) == (len(not_dwi_files) + len(dwi_b0_files) + len(dwi_bxxx_files))
    not_dwi_dir = os.path.join(out_dwi_path, 'not_dwi')
    os.makedirs(not_dwi_dir, exist_ok=True)
    dwi_b0_dir = os.path.join(out_dwi_path, 'b0')
    os.makedirs(dwi_b0_dir, exist_ok=True)
    dwi_bxxx_dir = os.path.join(out_dwi_path, 'bxxx')
    os.makedirs(dwi_bxxx_dir, exist_ok=True)
    for src_file in not_dwi_files:
        dst_file = os.path.join(not_dwi_dir, os.path.basename(src_file))
        shutil.copyfile(src_file, dst_file)
    for src_file in dwi_b0_files:
        dst_file = os.path.join(dwi_b0_dir, os.path.basename(src_file))
        shutil.copyfile(src_file, dst_file)
    for src_file in dwi_bxxx_files:
        dst_file = os.path.join(dwi_bxxx_dir, os.path.basename(src_file))
        shutil.copyfile(src_file, dst_file)
    return len(in_files) == (len(not_dwi_files) + len(dwi_b0_files) + len(dwi_bxxx_files)) and len(not_dwi_files) == 0

def ncct_extract_series_from_raw_series(in_dir, out_dir, info_file):
    '''
    indir = '../data/gan/ncct2dwi/siyuan_dcm_with_pid'
    outdir = '../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm'
    info_file = '../data/gan/ncct2dwi/experiment_registration2/config/V1 四院NCCT-DWI-ADC-RAPID.xlsx'

    invoke cmd: python utils.py ncct_extract_series_from_raw_series '../data/gan/ncct2dwi/siyuan_dcm_with_pid' '../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm' '../data/gan/ncct2dwi/experiment_registration2/config/V1 四院NCCT-DWI-ADC-RAPID.xlsx'
    debug cmd: ncct_extract_series_from_raw_series('../data/gan/ncct2dwi/siyuan_dcm_with_pid', '../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm', '../data/gan/ncct2dwi/experiment_registration2/config/V1 四院NCCT-DWI-ADC-RAPID.xlsx')
    
    out_dir:
    tree -L 4
    ├── 137611
    ├── ADC
    │   └── 1.3.12.2.1107.5.2.30.26961.2016121209575876909543666.0.0.0
    ├── DWI
    │   └── 1.3.12.2.1107.5.2.30.26961.2016121209575876909443665.0.0.0
    │       ├── b0
    │       ├── bxxx
    │       └── not_dwi
    └── NCCT
        └── 1.3.12.2.1107.5.1.4.95874.30000016121100222306600009841


    '''
    os.makedirs(out_dir, exist_ok=True)
    patient_infos = ncct_extract_infos_from_xlsx(info_file)
    for key in tqdm(patient_infos.keys()):
        patient_info = patient_infos[key]
        patient_in_root = os.path.join(in_dir, key)
        if not os.path.isdir(patient_in_root):
            continue
        ncct_series_path = os.path.join(patient_in_root, patient_info['ncct_series'])
        dwi_series_path = os.path.join(patient_in_root, patient_info['dwi_series'])
        adc_series_path = os.path.join(patient_in_root, patient_info['adc_series'])
        if not os.path.isdir(ncct_series_path):
            continue
        if not os.path.isdir(dwi_series_path):
            continue
        if not os.path.isdir(adc_series_path):
            continue
        out_ncct_series = os.path.join(out_dir, key, 'NCCT', patient_info['ncct_series'])
        shutil.copytree(ncct_series_path, out_ncct_series)
        out_adc_series = os.path.join(out_dir, key, 'ADC', patient_info['adc_series'])
        shutil.copytree(adc_series_path, out_adc_series)
        
        out_dwi_series = os.path.join(out_dir, key, 'DWI', patient_info['dwi_series'])
        ncct_extract_dwi_from_raw_dwi_single(dwi_series_path, out_dwi_series)

def ncct_set_origal_point_single(infile, outfile, origal=[0,0,0]):
    image = sitk.ReadImage(infile)
    image.SetOrigin(origal)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outfile)
    writer.Execute(image)

# 将nii.gz数据的起点设置到统一的位置，以便在看图软件中查看
def ncct_set_original_point(indir, outdir, original=[0,0,0]):
    '''
    indir = '../data/gan/ncct2dwi/experiment_registration2/1.nii_file'
    outdir = '../data/gan/ncct2dwi/experiment_registration2/2.nii_file_ori'

    invoke cmd: python utils.py ncct_set_original_point '../data/gan/ncct2dwi/experiment_registration2/1.nii_file' '../data/gan/ncct2dwi/experiment_registration2/2.nii_file_ori'
    debug cmd: ncct_set_original_point('../data/gan/ncct2dwi/experiment_registration2/1.nii_file', '../data/gan/ncct2dwi/experiment_registration2/2.nii_file_ori')
    
    '''    
    os.makedirs(outdir, exist_ok=True)
    infiles = glob(os.path.join(indir, '*.nii.gz'))
    for infile in tqdm(infiles):
        outfile = os.path.join(outdir, os.path.basename(infile))
        ncct_set_origal_point_single(infile, outfile, original)


# 提取脑实质
def extract_cerebral_parenchyma_onecase(infile, outdir, inpattern='_NCCT.nii.gz', outpattern='_brain.nii.gz'):
    if not os.path.isfile(infile):
        return
    sitk_mask = inference(infile, '../../cerebral_parenchyma/train/model/extract_cerebral_parenchyma/extract_cerebral_parenchyma_0056_best_loss_0.011.pth', None, is_dcm=False)
    writer = sitk.ImageFileWriter()
    
    outfile = os.path.join(outdir, os.path.basename(infile).replace(inpattern, outpattern))
    writer.SetFileName(outfile)
    writer.Execute(sitk_mask)




def extract_cerebral_parenchyma_singletask(infiles, outdir, inpattern='_NCCT.nii.gz', outpattern='_brain.nii.gz'):
    for infile in tqdm(infiles):
        extract_cerebral_parenchyma_onecase(infile, outdir, inpattern, outpattern)

def extract_cerebral_parenchyma_multiprocess(indir, outdir, inpattern='_NCCT.nii.gz', outpattern='_brain.nii.gz'):
    infiles = glob(os.path.join(indir, '*{}'.format(inpattern)))
    extract_cerebral_parenchyma_singletask(infiles, outdir, inpattern, outpattern)




# 提取脑实质部分的mask, 所有层面都按照最大层进行mask运算
def ncct_generate_cerebral_parenchyma(indir, outdir, inpattern):
    os.makedirs(outdir, exist_ok=True)
    infiles = glob(os.path.join(indir, inpattern))
    # out_arr = np.array()
    for infile in tqdm(infiles):
        in_img = sitk.ReadImage(infile)
        in_arr = sitk.GetArrayFromImage(in_img)
        # print('z size:\t{}'.format(in_arr.shape[0]))
        out_arr = np.zeros(in_arr.shape, dtype=in_arr.dtype)
        # 范围限定在(5, in_arr.shape[0]-5)，因为配准时脑实质图像的上下边缘生成有问题
        for z in range(5, in_arr.shape[0]-5):
            for y in range(in_arr.shape[1]):
                x_arr = in_arr[z,y,:]
                low_thres = 0
                ranges = np.where(x_arr != low_thres)
                if len(ranges[0]) > 0:
                    [x_min] = np.min(ranges, axis=1)
                    [x_max] = np.max(ranges, axis=1)
                    out_arr[z,y,x_min:x_max+1] = 1
        
        # 在保留的断层中，mask区域扩大到和最大层面面积相等
        max_region = np.max(out_arr, axis=0)
        # 根据脑实质的区域大小，选择是否保留
        layers = np.sum(out_arr, axis=(1,2))
        for z in range(in_arr.shape[0]):
            if layers[z]/max(layers) < 0.5:
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

def ncct_generate_cerebral_parenchyma_single(infiles, outdir):
    for infile in infiles:
        in_img = sitk.ReadImage(infile)
        in_arr = sitk.GetArrayFromImage(in_img)
        # print('z size:\t{}'.format(in_arr.shape[0]))
        out_arr = np.zeros(in_arr.shape, dtype=in_arr.dtype)
        # 范围限定在(5, in_arr.shape[0]-5)，因为配准时脑实质图像的上下边缘生成有问题
        for z in range(5, in_arr.shape[0]-5):
            for y in range(in_arr.shape[1]):
                x_arr = in_arr[z,y,:]
                low_thres = 0
                ranges = np.where(x_arr != low_thres)
                if len(ranges[0]) > 0:
                    [x_min] = np.min(ranges, axis=1)
                    [x_max] = np.max(ranges, axis=1)
                    out_arr[z,y,x_min:x_max+1] = 1
        
        # 在保留的断层中，mask区域扩大到和最大层面面积相等
        max_region = np.max(out_arr, axis=0)
        # 根据脑实质的区域大小，选择是否保留
        layers = np.sum(out_arr, axis=(1,2))
        for z in range(in_arr.shape[0]):
            if layers[z]/max(layers) < 0.5:
                out_arr[z,:,:] = 0
            else:
                out_arr[z,:,:] = max_region

        out_img = sitk.GetImageFromArray(out_arr)
        out_img.CopyInformation(in_img)
        out_file = os.path.join(outdir, os.path.basename(infile))
        writer = sitk.ImageFileWriter()
        writer.SetFileName(out_file)
        writer.Execute(out_img)

def ncct_generate_cerebral_parenchyma_multiprocess(indir, outdir, inpattern, process_num=12):
    
    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []
    
    os.makedirs(outdir, exist_ok=True)
    infiles = glob(os.path.join(indir, inpattern))
    
    num_per_process = (len(infiles) + process_num - 1)//process_num

    for i in range(process_num):
        sub_infiles = infiles[num_per_process*i:min(num_per_process*(i+1), len(infiles)-1)]
        result = pool.apply_async(ncct_generate_cerebral_parenchyma_single, args=(sub_infiles, outdir))
        results.append(result)

    pool.close()
    pool.join()


def utils_get_folder_pattern(indir, initpattern):
    files = glob(os.path.join(indir, initpattern))
    pattern = os.path.basename(files[0])
    pattern = pattern.replace(pattern.split('_')[0], '')
    return pattern

# 根据脑实质mask， 生成训练和测试的配置文件
def ncct_genereate_cta2dwi_config_file_with_cerebral_parenchyma(indir, configdir, train_ratio=0.8):
    '''
    目录结构， cerebral_parenchyma为脑实质mask文件
    .
    ├── ADC
    ├── cerebral_parenchyma
    ├── DWI_B0
    ├── DWI_BXXX
    └── NCCT

    ADC 文件名格式: 137611_first_FU_ADC*.nii.gz 
    cerebral_parenchyma 文件名格式: 137611_first_BS_brain*.nii.gz
    DWI_B0 文件名格式: 137611_first_FU_DWI_B0*.nii.gz
    DWI_BXXX 文件名格式: 137611_first_FU_DWI_BXXX*.nii.gz
    NCCT 文件名格式: 137611_first_BS_NCCT*.nii.gz

    debug command: ncct_genereate_cta2dwi_config_file_with_cerebral_parenchyma('../data/gan/ncct2dwi/experiment_registration2/8.out', '../data/gan/ncct2dwi/experiment_registration2/8.out/config')
    '''
    brain_files = glob(os.path.join(indir, 'cerebral_parenchyma', '*.nii.gz'))
    ncct_files = []
    dwi_b0_files = []
    dwi_bxxx_files = []
    adc_files = []
    mask_files = []
    ncct_pattern = utils_get_folder_pattern(os.path.join(indir, 'NCCT'), '*.nii.gz')
    dwi_b0_pattern = utils_get_folder_pattern(os.path.join(indir, 'DWI_B0'), '*.nii.gz')
    dwi_bxxx_pattern = utils_get_folder_pattern(os.path.join(indir, 'DWI_BXXX'), '*.nii.gz')
    adc_pattern = utils_get_folder_pattern(os.path.join(indir, 'ADC'), '*.nii.gz')
    mask_pattern = utils_get_folder_pattern(os.path.join(indir, 'cerebral_parenchyma'), '*.nii.gz')
    for i in range(len(brain_files)):
        if not os.path.isfile(brain_files[i]):
            continue
        index = os.path.basename(brain_files[i]).split('_')[0]
        ncct_file = os.path.join(indir, 'NCCT', '{}{}'.format(index, ncct_pattern))
        dwi_b0_file = os.path.join(indir, 'DWI_B0', '{}{}'.format(index, dwi_b0_pattern))
        dwi_bxxx_file = os.path.join(indir, 'DWI_BXXX', '{}{}'.format(index, dwi_bxxx_pattern))
        adc_file = os.path.join(indir, 'ADC', '{}{}'.format(index, adc_pattern))
        mask_file = os.path.join(indir, 'cerebral_parenchyma', '{}{}'.format(index, mask_pattern))
        if not os.path.isfile(ncct_file):
            continue
        if not os.path.isfile(dwi_b0_file):
            continue
        if not os.path.isfile(dwi_bxxx_file):
            continue
        if not os.path.isfile(adc_file):
            continue
        if not os.path.isfile(mask_file):
            continue
        ncct_files.append(ncct_file)
        dwi_b0_files.append(dwi_b0_file)
        dwi_bxxx_files.append(dwi_bxxx_file)
        adc_files.append(adc_file)
        mask_files.append(mask_file)
    assert len(ncct_files) == len(dwi_b0_files) == len(brain_files) == len(dwi_bxxx_files) == len(adc_files) == len(mask_files)
    config_infos = []
    config_ncct_to_dwi_b0_infos = []
    config_ncct_to_dwi_bxxx_infos = []
    config_ncct_to_adc_infos = []

    config_mask_infos = []
    config_mask_ncct_to_dwi_b0_infos = []
    config_mask_ncct_to_dwi_bxxx_infos = []
    config_mask_ncct_to_adc_infos = []

    def _helper_get_file_name(fullname):
        # 取文件当前name和上一级目录name
        # 如输入'../data/gan/ncct2dwi/experiment_registration2/8.out/NCCT/137611_first_BS_NCCT.nii.gz'
        # 返回 'NCCT/137611_first_BS_NCCT.nii.gz'
        return os.path.join(os.path.basename(os.path.dirname(fullname)), os.path.basename(fullname))
    
    for i in tqdm(range(len(brain_files))):
        brain_img = sitk.ReadImage(brain_files[i])
        brain_arr = sitk.GetArrayFromImage(brain_img)
        ranges = np.where(brain_arr > 0)
        [z_min, y_min, x_min] = np.min(np.array(ranges), axis=1)
        [z_max, y_max, x_max] = np.max(np.array(ranges), axis=1)
        # ncct / dwi_b0 / dwi_bxxx / adc
        info = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            _helper_get_file_name(ncct_files[i]),_helper_get_file_name(dwi_b0_files[i]),
            _helper_get_file_name(dwi_bxxx_files[i]), _helper_get_file_name(adc_files[i]),
            z_min, z_max, y_min, y_max, x_min, x_max
        )
        config_infos.append(info)
        # ncct / dwi_b0
        info = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            _helper_get_file_name(ncct_files[i]), _helper_get_file_name(dwi_b0_files[i]), 
            z_min, z_max, y_min, y_max, x_min, x_max
        )
        config_ncct_to_dwi_b0_infos.append(info)
        # ncct / dwi_bxxx
        info = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            _helper_get_file_name(ncct_files[i]), _helper_get_file_name(dwi_bxxx_files[i]), 
            z_min, z_max, y_min, y_max, x_min, x_max
        )
        config_ncct_to_dwi_bxxx_infos.append(info)
        # ncct / adc
        info = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            _helper_get_file_name(ncct_files[i]),_helper_get_file_name(adc_files[i]), 
            z_min, z_max, y_min, y_max, x_min, x_max
        )
        config_ncct_to_adc_infos.append(info)


        # ncct / dwi_b0 / dwi_bxxx / adc / mask
        info = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            _helper_get_file_name(ncct_files[i]),_helper_get_file_name(dwi_b0_files[i]),
            _helper_get_file_name(dwi_bxxx_files[i]), _helper_get_file_name(adc_files[i]),
            z_min, z_max, y_min, y_max, x_min, x_max, _helper_get_file_name(mask_files[i])
        )
        config_mask_infos.append(info)
        # ncct / dwi_b0 / mask
        info = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            _helper_get_file_name(ncct_files[i]), _helper_get_file_name(dwi_b0_files[i]), 
            z_min, z_max, y_min, y_max, x_min, x_max, _helper_get_file_name(mask_files[i])
        )
        config_mask_ncct_to_dwi_b0_infos.append(info)
        # ncct / dwi_bxxx / mask
        info = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            _helper_get_file_name(ncct_files[i]), _helper_get_file_name(dwi_bxxx_files[i]), 
            z_min, z_max, y_min, y_max, x_min, x_max, _helper_get_file_name(mask_files[i])
        )
        config_mask_ncct_to_dwi_bxxx_infos.append(info)
        # ncct / adc / mask
        info = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(
            _helper_get_file_name(ncct_files[i]),_helper_get_file_name(adc_files[i]), 
            z_min, z_max, y_min, y_max, x_min, x_max, _helper_get_file_name(mask_files[i])
        )
        config_mask_ncct_to_adc_infos.append(info)

        
    def write_config_file(config_infos, train_ratio, prefix=''):
        np.random.shuffle(config_infos)
        pos = int(len(config_infos)*float(train_ratio))
        train_config_infos = config_infos[:pos]
        test_config_infos = config_infos[pos:]
        os.makedirs(configdir, exist_ok=True)
        train_config_file = os.path.join(configdir, '{}train_config_file.txt'.format(prefix))
        test_config_file = os.path.join(configdir, '{}test_config_file.txt'.format(prefix))
        with open(train_config_file, 'w') as f:
            f.write('\n'.join(train_config_infos))
        with open(test_config_file, 'w') as f:
            f.write('\n'.join(test_config_infos))

    write_config_file(config_infos, train_ratio, '')
    write_config_file(config_ncct_to_dwi_b0_infos, train_ratio, 'ncct_to_dwi_b0_')
    write_config_file(config_ncct_to_dwi_bxxx_infos, train_ratio, 'ncct_to_dwi_bxxx_')
    write_config_file(config_ncct_to_adc_infos, train_ratio, 'ncct_to_adc_')

    write_config_file(config_mask_infos, train_ratio, '')
    write_config_file(config_mask_ncct_to_dwi_b0_infos, train_ratio, 'mask_ncct_to_dwi_b0_')
    write_config_file(config_mask_ncct_to_dwi_bxxx_infos, train_ratio, 'mask_ncct_to_dwi_bxxx_')
    write_config_file(config_mask_ncct_to_adc_infos, train_ratio, 'mask_ncct_to_adc_')
    print('finish ncct_genereate_cta2dwi_config_file_with_cerebral_parenchyma!')

def split_dicoms_by_series_uid(indir, outdir, inpattern='*.dcm'):
    '''
    将一个文件夹下的dcm文件, 根据series uid进行划分，并copy到指定位置
    '''
    dcm_files = glob(os.path.join(indir, inpattern))
    for dcm_file in dcm_files:
        metadata = pydicom.dcmread(dcm_file)
        # metadata.SeriesInstanceUID
        # metadata.AcquisitionDateTime
        out_sub_dir = os.path.join(outdir, metadata.SeriesInstanceUID)
        os.makedirs(out_sub_dir, exist_ok=True)
        dst_file = os.path.join(out_sub_dir, os.path.basename(dcm_file))
        shutil.copyfile(dcm_file, dst_file)

def split_dicoms_by_series_uid_batch_processing(indir, outdir):
    '''
    python utils.py split_dicoms_by_series_uid_batch_processing ../data/gan/cta2dwi/cta_dwi_pairs1/新增LVO病例 ../data/gan/cta2dwi/cta_dwi_pairs1/processed

    indir: 

    └── 新增LVO病例
        ├── 1237062
        │   ├── CTA
        │   │   ├── ImageFileName000.dcm
        │   │   ├── ImageFileName001.dcm
        │   │   ├── ImageFileName002.dcm
        │   │   ├── ImageFileName003.dcm
        │   │   ├── ImageFileName004.dcm
        │   │   ├── ImageFileName005.dcm
        │   │   ├── ImageFileName006.dcm
        │   │   ├── ImageFileName007.dcm
        │   │   ├── ImageFileName008.dcm
        │   │   ├── ImageFileName009.dcm
        │   │   ├── ImageFileName010.dcm
        │   │   ├── ImageFileName011.dcm
        │   │   ├── ImageFileName012.dcm
        │   │   ├── ImageFileName013.dcm
        │   │   ├── ImageFileName014.dcm
        │   │   ├── ImageFileName015.dcm
        │   │   ├── ImageFileName016.dcm
        │   │   ├── ImageFileName017.dcm
        │   │   ├── ImageFileName018.dcm
        │   │   ├── ImageFileName019.dcm
        │   │   ├── ImageFileName020.dcm

    outdir:
    .
    ├── processed
    │   ├── 1237062
    │   │   ├── CTA
    │   │   │   ├── 1.2.156.112605.189250946103856.200130014638.3.6240.105874
    │   │   │   ├── 1.2.156.112605.189250946103856.200130014638.3.6240.115874
    │   │   │   ├── 1.2.156.112605.189250946103856.200130014638.3.6240.28416
    │   │   │   ├── 1.2.156.112605.189250946103856.200130014638.3.6240.48416
    │   │   │   ├── 1.2.156.112605.189250946103856.200130014638.3.6240.62644
    │   │   │   ├── 1.2.156.112605.189250946103856.200130014638.3.6240.826441
    │   │   │   ├── 1.2.156.112605.189250946103856.200130015029.3.6240.25613
    │   │   │   ├── 1.3.46.670589.50.2.1842472151086579274.2768611560886914130
    │   │   │   ├── 1.3.46.670589.50.2.192950329412217928.23945447611569155034
    │   │   │   ├── 1.3.46.670589.50.2.2574957372138396236.2269125123460991912
    │   │   │   ├── 1.3.46.670589.50.2.26793237261059333197.27159137933919647822
    │   │   │   ├── 1.3.46.670589.50.2.32250269733773645127.25925457891242442603
    │   │   │   ├── 1.3.46.670589.50.2.4106609966789961798.31052890471153572692
    │   │   │   ├── 1.3.46.670589.50.2.4479652623846357324.29839196292206168438
    │   │   │   └── 1.3.6.1.4.1.19439.1.000001.066077080.0
    │   │   └── MR
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.2020020109210919911225629.0.0.0
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.2020020109223437160926177.0.0.0
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.2020020109224055015226294.0.0.0
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.2020020109242467654626695.0.0.0
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.2020020109252616277526818.0.0.0
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.2020020109252616277726819.0.0.0
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.202002010926276562627435.0.0.0
    │   │       └── 1.3.6.1.4.1.19439.1.000001.066077080.0



    '''
    for dirname in tqdm(os.listdir(indir)):
        in_ct_dir = os.path.join(indir, dirname, 'CTA')
        in_mr_dir = os.path.join(indir, dirname, 'MR')
        if not os.path.isdir(in_ct_dir):
            continue
        if not os.path.isdir(in_mr_dir):
            continue
        out_ct_dir = os.path.join(outdir, dirname, 'CTA')
        out_mr_dir = os.path.join(outdir, dirname, 'MR')
        split_dicoms_by_series_uid(in_ct_dir, out_ct_dir)
        split_dicoms_by_series_uid(in_mr_dir, out_mr_dir)

def cta_find_arterial_phase(in_cta_dir):
    '''
    '''
    cur_time = None
    arterial_phase_series_uid = None
    cur_time_str = None
    for series_uid in os.listdir(in_cta_dir):
        series_path = os.path.join(in_cta_dir, series_uid)
        if not os.path.isdir(series_path):
            continue
        dcm_files = glob(os.path.join(series_path, '*.dcm'))
        if len(dcm_files) < 50:
            continue
        dcm_file = dcm_files[0]
        metadata = pydicom.dcmread(dcm_file)
        if 'CT' not in metadata.Modality:
            continue
        if 'SliceThickness' in metadata:
            if float(metadata.SliceThickness) > 2:
                continue
        if 'AcquisitionDate' not in metadata:
            continue
        meta_time = metadata.AcquisitionDate + metadata.AcquisitionTime
        acq_time = datetime.datetime.strptime(meta_time.split('.')[0], '%Y%m%d%H%M%S')
        if cur_time is None:
            cur_time = acq_time
            arterial_phase_series_uid = series_uid
            cur_time_str = meta_time
        else:
            if cur_time > acq_time:
                cur_time = acq_time
                arterial_phase_series_uid = series_uid
                cur_time_str = meta_time
    return arterial_phase_series_uid, cur_time_str

def cta_find_dwi(in_mr_dir):
    dwi_series_uid = None
    adc_series_uid = None
    dwi_acq_time = ''
    adc_acq_time = ''
    for series_uid in os.listdir(in_mr_dir):
        series_path = os.path.join(in_mr_dir, series_uid)
        if not os.path.isdir(series_path):
            continue
        dcm_files = glob(os.path.join(series_path, '*.dcm'))
        if len(dcm_files) < 10:
            continue
        dcm_file = dcm_files[0]
        metadata = pydicom.dcmread(dcm_file)

        image_type = metadata.ImageType
        if ('DIFFUSION' in image_type and 'TRACEW' in image_type):
            dwi_series_uid = series_uid
            dwi_acq_time = metadata.AcquisitionDate + metadata.AcquisitionTime
            continue
        if ('DIFFUSION' in image_type and 'ADC' in image_type):
            adc_series_uid = series_uid
            adc_acq_time = metadata.AcquisitionDate + metadata.AcquisitionTime
            continue
    return dwi_series_uid, dwi_acq_time, adc_series_uid, adc_acq_time

def cta_generate(indir):
    '''
    indir
    .
    ├── processed
    │   ├── 1237062
    │   │   ├── CTA
    │   │   │   ├── 1.2.156.112605.189250946103856.200130014638.3.6240.105874
    │   │   │   ├── 1.2.156.112605.189250946103856.200130014638.3.6240.115874
    │   │   │   ├── 1.2.156.112605.189250946103856.200130014638.3.6240.28416
    │   │   │   ├── 1.2.156.112605.189250946103856.200130014638.3.6240.48416
    │   │   │   ├── 1.2.156.112605.189250946103856.200130014638.3.6240.62644
    │   │   │   ├── 1.2.156.112605.189250946103856.200130014638.3.6240.826441
    │   │   │   ├── 1.2.156.112605.189250946103856.200130015029.3.6240.25613
    │   │   │   ├── 1.3.46.670589.50.2.1842472151086579274.2768611560886914130
    │   │   │   ├── 1.3.46.670589.50.2.192950329412217928.23945447611569155034
    │   │   │   ├── 1.3.46.670589.50.2.2574957372138396236.2269125123460991912
    │   │   │   ├── 1.3.46.670589.50.2.26793237261059333197.27159137933919647822
    │   │   │   ├── 1.3.46.670589.50.2.32250269733773645127.25925457891242442603
    │   │   │   ├── 1.3.46.670589.50.2.4106609966789961798.31052890471153572692
    │   │   │   ├── 1.3.46.670589.50.2.4479652623846357324.29839196292206168438
    │   │   │   └── 1.3.6.1.4.1.19439.1.000001.066077080.0
    │   │   └── MR
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.2020020109210919911225629.0.0.0
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.2020020109223437160926177.0.0.0
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.2020020109224055015226294.0.0.0
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.2020020109242467654626695.0.0.0
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.2020020109252616277526818.0.0.0
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.2020020109252616277726819.0.0.0
    │   │       ├── 1.3.12.2.1107.5.2.36.40534.202002010926276562627435.0.0.0
    │   │       └── 1.3.6.1.4.1.19439.1.000001.066077080.0

    '''
    row_elems = []
    for patient_id in tqdm(os.listdir(indir)):
        cta_dir = os.path.join(indir, patient_id, 'CTA')
        mr_dir = os.path.join(indir, patient_id, 'MR')
        if not os.path.isdir(cta_dir):
            continue
        if not os.path.isdir(mr_dir):
            continue
        cta_id, cta_acq_time = cta_find_arterial_phase(cta_dir)
        dwi_id, dwi_acq_time, adc_id, adc_acq_time = cta_find_dwi(mr_dir)
        if dwi_id is None:
            continue

        cta_acq_time_t = datetime.datetime.strptime(cta_acq_time.split('.')[0], '%Y%m%d%H%M%S')
        cta_acq_time_str = cta_acq_time_t.strftime('%Y/%m/%d %H:%M')

        dwi_acq_time_t = datetime.datetime.strptime(dwi_acq_time.split('.')[0], '%Y%m%d%H%M%S')
        dwi_acq_time_str = dwi_acq_time_t.strftime('%Y/%m/%d %H:%M')

        adc_acq_time_t = datetime.datetime.strptime(adc_acq_time.split('.')[0], '%Y%m%d%H%M%S')
        adc_acq_time_str = adc_acq_time_t.strftime('%Y/%m/%d %H:%M')

        dwi_cta_time_str = str(dwi_acq_time_t-cta_acq_time_t)
        adc_cta_time_str = str(adc_acq_time_t-cta_acq_time_t)

        # row_elems.append(np.array([patient_id, cta_id, cta_acq_time, dwi_id, dwi_acq_time, adc_id, adc_acq_time]))
        row_elems.append(np.array([patient_id, cta_id, cta_acq_time_str, dwi_id, dwi_acq_time_str, adc_id, adc_acq_time_str, dwi_cta_time_str, adc_cta_time_str]))
    df = pd.DataFrame(np.array(row_elems), columns=['patient_id', 'cta_id', 'cta_acq_time', 'dwi_id', 'dwi_acq_time', 'adc_id', 'adc_acq_time', 'dwi_cta_time', 'adc_cta_time_str'])
    df.to_csv('cta_dwi_adc_test.csv')


if __name__ =='__main__':
    fire.Fire()
    # ncct_extract_infos_from_xlsx('../data/gan/ncct2dwi/experiment_registration1/config/V1 四院NCCT-DWI-ADC-RAPID.xlsx')
    # ncct_convert_dcm_to_niigz('../data/gan/ncct2dwi/siyuan_dcm_with_pid', '../data/gan/ncct2dwi/experiment_registration1/raw', '../data/gan/ncct2dwi/experiment_registration1/config/V1 四院NCCT-DWI-ADC-RAPID.xlsx')
    # reset_dcm_info('../data/gan/ncct2dwi/siyuan_dcm_with_pid/137611/1.3.12.2.1107.5.1.4.95874.30000016121100222306600009841', '/ssd2/zhangwd/data/brain/gan/ncct2dwi/experiment_registration1/tmp/test3', True)
    # extract_dwi_from_raw_dwi(in_dwi_path='../data/gan/ncct2dwi/siyuan_dcm_with_pid/475170/1.3.12.2.1107.5.2.30.26961.201908091523268765824442.0.0.0', out_dwi_path=None)
    # ncct_extract_dwi_from_raw_dwi_single(in_dwi_path='../data/gan/ncct2dwi/siyuan_dcm_with_pid/475170/1.3.12.2.1107.5.2.30.26961.201908091523268765724441.0.0.0', out_dwi_path=None)
    # ncct_extract_series_from_raw_series('../data/gan/ncct2dwi/siyuan_dcm_with_pid', '../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm', '../data/gan/ncct2dwi/experiment_registration2/config/V1 四院NCCT-DWI-ADC-RAPID.xlsx')
    # ncct_convert_dcm_to_niigz('../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm', '../data/gan/ncct2dwi/experiment_registration2/1.nii_file')
    # convert_nii_to_dcm('../data/gan/ncct2dwi/experiment_registration2/test/470933_first_BS_NCCT.nii.gz', '/ssd2/zhangwd/data/brain/gan/ncct2dwi/experiment_registration2/tmp/test3', False)
    # ncct_genereate_cta2dwi_config_file_with_cerebral_parenchyma('../data/gan/ncct2dwi/experiment_registration2/8.out', '../data/gan/ncct2dwi/experiment_registration2/8.out/config')
    # split_dicoms_by_series_uid('../data/gan/cta2dwi/cta_dwi_pairs1/新增LVO病例/4999020/CTA', None)
    # split_dicoms_by_series_uid_batch_processing('../data/gan/cta2dwi/cta_dwi_pairs1/新增LVO病例', '../data/gan/cta2dwi/cta_dwi_pairs1/processed')
    # cta_find_arterial_phase('../data/gan/cta2dwi/cta_dwi_pairs1/processed/1792160/CTA')
    # cta_find_arterial_phase('../data/gan/cta2dwi/cta_dwi_pairs1/processed/1792160/CTA')
    # cta_find_dwi('../data/gan/cta2dwi/cta_dwi_pairs1/processed/1792160/MR')
    # cta_generate('../data/gan/cta2dwi/cta_dwi_pairs1/processed')
    # ncct_extract_from_hospital_folder('../data/gan/hospital_4/CT+灌注/2018住院1/GENERAL', '../data/gan/hospital_4/0.ori')
    # ncct_extract_from_hospital_folder_all()
    # ncct_extract_info_from_patient('../data/gan/hospital_4/0.ori/332594')
    # ncct_extract_infos_from_patients_all('../data/gan/hospital_4/0.ori', '../data/gan/hospital_4/0.raw_dcm')
    # extract_cerebral_parenchyma_multiprocess('../data/gan/ncct2dwi/experiment_registration2/4 Patient_nii_unity', '../data/gan/ncct2dwi/experiment_registration2/4 Patient_nii_unity', '_NCCT.nii.gz', '_brain.nii.gz')
    # extract_cerebral_parenchyma_multiprocess('../data/gan/hospital_4/experiment_registration2/4 Patient_nii_unity', '../data/gan/hospital_4/experiment_registration2/4 Patient_nii_unity', '_NCCT.nii.gz', '_brain.nii.gz')
    # ncct_generate_table_all_series('../data/gan/hospital_4/0.ori', '../data/gan/hospital_4/0.table')