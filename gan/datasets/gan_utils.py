'''
@Description: 
@Version: 1.0
@Autor: searobbersanduck
@Date: 2020-04-09 09:52:50
@LastEditors: searobbersanduck
@LastEditTime: 2020-07-08 11:00:45
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
import cv2

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


def read_dcm_file(in_dcm_path):
    series_reader = sitk.ImageSeriesReader()
    dicomfilenames = series_reader.GetGDCMSeriesFileNames(in_dcm_path)
    series_reader.SetFileNames(dicomfilenames)

    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()

    image = series_reader.Execute()
    return image

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
    infos['MRP'] = []
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
        if len(dcm_files) > 500 and 'ProtocolName' in metadata and 'perf' in metadata.ProtocolName:
            info = {}
            info['series_uid'] = series_uid
            infos['MRP'] = info
            info['study_uid'] = study_uid
        
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
        infos, select_infos = ncct_extract_info_from_patient(patient_path, 259200)
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

        
    for select_info in tqdm(all_select_infos):
        patient_uid = select_info['PID']

        if len(select_info['DWI']) > 0:
            src_series = os.path.join(in_dir, patient_uid, select_info['DWI'][0]['series_uid'])
            dst_series = os.path.join(out_dir, patient_uid, 'DWI', select_info['DWI'][0]['series_uid'])
            os.makedirs(os.path.dirname(dst_series), exist_ok=True)
            ncct_extract_dwi_from_raw_dwi_single(src_series, dst_series)
            # shutil.copytree(src_series, dst_series)

        if len(select_info['ADC']) > 0:
            src_series = os.path.join(in_dir, patient_uid, select_info['ADC'][0]['series_uid'])
            dst_series = os.path.join(out_dir, patient_uid, 'ADC', select_info['ADC'][0]['series_uid'])
            os.makedirs(os.path.dirname(dst_series), exist_ok=True)
            shutil.copytree(src_series, dst_series)

        if len(select_info['NCCT']) > 0:
            src_series = os.path.join(in_dir, patient_uid, select_info['NCCT'][0]['series_uid'])
            dst_series = os.path.join(out_dir, patient_uid, 'NCCT', select_info['NCCT'][0]['series_uid'])
            os.makedirs(os.path.dirname(dst_series), exist_ok=True)
            shutil.copytree(src_series, dst_series)

        if len(select_info['RAPID']) > 0:
            src_series = os.path.join(in_dir, patient_uid, select_info['RAPID']['series_uid'])
            dst_series = os.path.join(out_dir, patient_uid, 'RAPID', select_info['RAPID']['series_uid'])
            os.makedirs(os.path.dirname(dst_series), exist_ok=True)
            shutil.copytree(src_series, dst_series)

        if len(select_info['MRP']) > 0:
            src_series = os.path.join(in_dir, patient_uid, select_info['MRP']['series_uid'])
            dst_series = os.path.join(out_dir, patient_uid, 'MRP', select_info['MRP']['series_uid'])
            os.makedirs(os.path.dirname(dst_series), exist_ok=True)
            ncct_extract_mrp_subseries_from_raw_mrp_single(src_series, dst_series)
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
    print('processing\t{}'.format(dcm_path))
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

def ncct_convert_dcm_to_niigz_onecase(indir, patient_ids, outdir, include_adc=True):
    for patient_id in tqdm(patient_ids):
        # print(patient_id)
        index = patient_id
        patient_id = os.path.join(indir, patient_id)
        ncct_path = os.path.join(patient_id, 'NCCT')
        # adc_path = os.path.join(patient_id, 'ADC')
        dwi_path = os.path.join(patient_id, 'DWI')
        if not os.path.isdir(ncct_path):
            print('ncct_path:\t{}'.format(ncct_path))
            continue
        # if not os.path.isdir(adc_path):
        #     print('adc_path:\t{}'.format(adc_path))
        #     continue
        if not os.path.isdir(dwi_path):
            print('dwi_path:\t{}'.format(dwi_path))
            continue
        ncct_path = os.path.join(ncct_path, os.listdir(ncct_path)[0])
        # adc_path = os.path.join(adc_path, os.listdir(adc_path)[0])
        dwi_path = os.path.join(dwi_path, os.listdir(dwi_path)[0])
        if not os.path.isdir(ncct_path):
            print('ncct_path:\t{}'.format(ncct_path))
            continue
        # if not os.path.isdir(adc_path):
        #     continue
        if not os.path.isdir(dwi_path):
            print('dwi_path:\t{}'.format(dwi_path))
            continue
        out_ncct_file = os.path.join(outdir, '{}_first_BS_NCCT.nii.gz'.format(index))
        out_adc_file = os.path.join(outdir, '{}_first_FU_ADC.nii.gz'.format(index))

        # ncct_convert_dcm_to_niigz_single(adc_path, out_adc_file)

        b0_path = os.path.join(dwi_path, 'b0')
        bxxx_path = os.path.join(dwi_path, 'bxxx')
        if not os.path.isdir(b0_path) or len(os.listdir(b0_path)) == 0:
            print('b0_path:\t{}'.format(b0_path))
            continue
        if not os.path.isdir(bxxx_path) or len(os.listdir(bxxx_path)) == 0:
            print('bxxx_path:\t{}'.format(bxxx_path))
            continue
        out_b0_file = os.path.join(outdir, '{}_first_FU_DWI_B0.nii.gz'.format(index))
        out_bxxx_file = os.path.join(outdir, '{}_first_FU_DWI_BXXX.nii.gz'.format(index))
        # 将ncct的转换放在此处，是为了保证ncct和dwi同时存在或不存在
        ncct_convert_dcm_to_niigz_single(ncct_path, out_ncct_file, True)
        ncct_convert_dcm_to_niigz_single(b0_path, out_b0_file)
        ncct_convert_dcm_to_niigz_single(bxxx_path, out_bxxx_file)

        # 有些数据不包括ADC数据，这里单独进行处理
        if include_adc:
            adc_path = os.path.join(patient_id, 'ADC')
            if not os.path.isdir(adc_path):
                print('adc_path:\t{}'.format(adc_path))
                continue
            adc_path = os.path.join(adc_path, os.listdir(adc_path)[0])
            if not os.path.isdir(adc_path):
                continue
            ncct_convert_dcm_to_niigz_single(adc_path, out_adc_file)


        

def ncct_convert_dcm_to_niigz_multiprocess(indir, outdir, process_num=24, include_adc=True):
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
        sub_infiles = patient_ids[num_per_process*i:min(num_per_process*(i+1), len(patient_ids))]
        print(sub_infiles)
        result = pool.apply_async(ncct_convert_dcm_to_niigz_onecase, args=(indir, sub_infiles, outdir, include_adc))
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



def ncct_extract_mrp_subseries_from_raw_mrp_single(in_mrp_path, out_mrp_path):
    '''
    make sure input is mrp series
    '''
    in_files1 = glob(os.path.join(in_mrp_path, '*.dcm'))
    in_files2 = glob(os.path.join(in_mrp_path, '*.DCM'))
    in_files = in_files1 + in_files2
    for in_file in in_files:
        metadata = pydicom.dcmread(in_file)
        acq_num = metadata.AcquisitionNumber
        sub_out_dir = os.path.join(out_mrp_path, str(acq_num))
        os.makedirs(sub_out_dir, exist_ok=True)
        dst_file = os.path.join(sub_out_dir, os.path.basename(in_file))
        shutil.copyfile(in_file, dst_file)


def cta_extract_exotic_flower_dwi(series_path):
    dcm_files = os.listdir(series_path)
    assert(len(dcm_files)%2 == 0)
    info_np = np.zeros((len(dcm_files),6))
    for j in range(len(dcm_files)):
        info_np[j,0] = j
        dcm_file = os.path.join(series_path, dcm_files[j])
        metadata = pydicom.dcmread(dcm_file)
        info_np[j,3] = float(metadata.WindowWidth)
        info_np[j,4] = float(metadata.WindowCenter)
        info_np[j,5] = float(metadata.SliceLocation)
        img = sitk.ReadImage(dcm_file)
        arr = sitk.GetArrayFromImage(img)
        info_np[j,2] = np.max(arr)
        info_np[j,1] = np.min(arr)
    sort_info = info_np[np.argsort(info_np[:,-1])]
    dwi_bxxx_files = []
    dwi_b0_files = []
    for j in range(sort_info.shape[0]//2):
        minus = sort_info[2*j,2:5]-sort_info[2*j+1, 2:5]
        p = (minus>0)*1
        xp = np.sum(p)
        assert(xp == 3 or xp == 0)
        if xp == 3:
            out_indx = sort_info[2*j+1, 0]
        else:
            out_indx = sort_info[2*j,0]
        dwi_bxxx_files.append(dcm_files[int(out_indx)])
    for f in dcm_files:
        if f not in dwi_bxxx_files:
            dwi_b0_files.append(f)
    dwi_b0_files = [os.path.join(series_path, i) for i in dwi_b0_files]
    dwi_bxxx_files = [os.path.join(series_path, i) for i in dwi_bxxx_files]
    return dwi_b0_files, dwi_bxxx_files


def cta_extract_dwi_from_raw_dwi_single(in_dwi_path, out_dwi_path):
    '''
    使用此函数的前提是所有的数据数据都为DWI数据，仅利用此数据来区分b0和bxxx
    '''
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
        # if not('DIFFUSION' in image_type and 'TRACEW' in image_type):
        #     not_dwi_files.append(in_file)
        #     continue
        try:            
            if 'SequenceName' in metadata and metadata.SequenceName != '':
                seq_name = metadata.SequenceName
                if 'b0' in seq_name:
                    dwi_b0_files.append(in_file)
                elif 'b1000' in seq_name:
                    dwi_bxxx_files.append(in_file)
            elif 'DiffusionBValue' in metadata:
                b_value = float(metadata.DiffusionBValue)
                if b_value > 100:
                    dwi_bxxx_files.append(in_file)
                else:
                    dwi_b0_files.append(in_file)
            else:
                not_dwi_files.append(in_file)
        except:
            pass
    
    if len(not_dwi_files) > 0 and len(dwi_bxxx_files) == 0:
        dwi_b0_files, dwi_bxxx_files = cta_extract_exotic_flower_dwi(in_dwi_path)
        not_dwi_files = []
    
    # assert len(in_files) == (len(not_dwi_files) + len(dwi_b0_files) + len(dwi_bxxx_files))
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
    '''
    ncct_set_origal_point_single('../data/gan/hospital_6/experiment_registration2/1.nii_file/3833955_first_BS_NCCT.nii.gz', '../data/gan/hospital_6/experiment_registration2/2.nii_file_ori/3833955_first_BS_NCCT.nii.gz')
    '''
    image = sitk.ReadImage(infile)
    image.SetOrigin(origal)

    writer = sitk.ImageFileWriter()
    writer.SetFileName(outfile)
    writer.Execute(image)

# 将nii.gz数据的起点设置到统一的位置，以便在看图软件中查看

def ncct_set_origal_point_singletask(infiles, outdir, original=[0,0,0]):
    for infile in tqdm(infiles):
        outfile = os.path.join(outdir, os.path.basename(infile))
        ncct_set_origal_point_single(infile, outfile, original)

def ncct_set_original_point(indir, outdir, original=[0,0,0], process_num=24):
    '''
    indir = '../data/gan/ncct2dwi/experiment_registration2/1.nii_file'
    outdir = '../data/gan/ncct2dwi/experiment_registration2/2.nii_file_ori'

    invoke cmd: python utils.py ncct_set_original_point '../data/gan/hospital_6/experiment_registration2/1.nii_file' '../data/gan/hospital_6/experiment_registration2/2.nii_file_ori'
    debug cmd: ncct_set_original_point('../data/gan/hospital_6/experiment_registration2/1.nii_file', '../data/gan/hospital_6/experiment_registration2/2.nii_file_ori')
    
    '''    
    os.makedirs(outdir, exist_ok=True)
    infiles = glob(os.path.join(indir, '*.nii.gz'))


    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []

    num_per_process = (len(infiles) + process_num - 1)//process_num

    print(len(infiles))
    for i in range(process_num):
        sub_infiles = infiles[num_per_process*i:min(num_per_process*(i+1), len(infiles))]
        print(len(sub_infiles))
        result = pool.apply_async(ncct_set_origal_point_singletask, args=(sub_infiles, outdir, original))
        results.append(result)

    pool.close()
    pool.join()



# 提取脑实质
def extract_cerebral_parenchyma_onecase(infile, outdir, inpattern='_NCCT.nii.gz', outpattern='_brain.nii.gz'):
    '''
    extract_cerebral_parenchyma_onecase('../data/gan/hospital_6/experiment_registration2/4 Patient_nii_unity/4495700_first_BS_NCCT.nii.gz', '../data/gan/hospital_6/experiment_registration2/tmp')
    '''
    if not os.path.isfile(infile):
        return
    os.makedirs(outdir, exist_ok=True)
    sitk_mask = inference(infile, '../../cerebral_parenchyma/train/model/extract_cerebral_parenchyma/extract_cerebral_parenchyma_0056_best_loss_0.011.pth', None, is_dcm=False)
    # writer = sitk.ImageFileWriter()
    # outfile = os.path.join(outdir, os.path.basename(infile).replace(inpattern, outpattern))
    # writer.SetFileName(outfile)
    # writer.Execute(sitk_mask)

    ## 截取mask有脑实质的部分
    ## 通过mask提取脑实质部分存成文件，pattern：*_brain.nii.gz
    ## 1. 根据mask算出脑实质的有效范围，并算出z方向上的区间
    ## 2. 根据2中计算出的区间范围，分别截取CT、脑实质mask、脑实质，存储后缀分别为_NCCT_cut.nii.gz, _brain_mask.nii.gz, _brain.nii.gz

    ## step 1.
    mask_arr = sitk.GetArrayFromImage(sitk_mask)
    mask_z_sum = np.sum(np.sum(mask_arr, axis=-1), axis=-1)
    ranges = np.where(mask_z_sum > 0)
    [z_min] = np.min(np.array(ranges), axis=1)
    [z_max] = np.max(np.array(ranges), axis=1)

    ## step 2.1 截取CT
    src_img_ct = sitk.ReadImage(infile)
    origin = src_img_ct.GetOrigin()
    spc = src_img_ct.GetSpacing()
    direction = src_img_ct.GetDirection()
    src_arr = sitk.GetArrayFromImage(src_img_ct)
    out_arr = src_arr[z_min:z_max+1, :, :]
    out_img = sitk.GetImageFromArray(out_arr)
    out_img.SetOrigin(origin)
    out_img.SetDirection(direction)
    out_img.SetSpacing(spc)
    # out_file = os.path.join(outdir, os.path.basename(infile).replace(inpattern, inpattern))
    out_file = os.path.join(outdir, os.path.basename(infile).replace(inpattern, '_NCCT_crop.nii.gz'))
    sitk.WriteImage(out_img, out_file)
    ## step 2.2 截取mask
    src_img = sitk_mask
    src_arr = sitk.GetArrayFromImage(src_img)
    out_arr = src_arr[z_min:z_max+1, :, :]
    out_img = sitk.GetImageFromArray(out_arr)
    out_img.SetOrigin(origin)
    out_img.SetDirection(direction)
    out_img.SetSpacing(spc)
    out_file = os.path.join(outdir, os.path.basename(infile).replace(inpattern, '_brain_mask.nii.gz'))
    sitk.WriteImage(out_img, out_file)
    ## step 2.3 截取脑实质
    maskfilter = sitk.MaskImageFilter()
    maskfilter.SetOutsideValue(-1024)
    mask_img = sitk.Cast(sitk_mask, sitk.sitkInt16)
    src_img_brain = sitk.Cast(src_img_ct, sitk.sitkInt16)
    out_img = maskfilter.Execute(src_img_brain, mask_img)
    src_img = out_img
    src_arr = sitk.GetArrayFromImage(src_img)
    out_arr = src_arr[z_min:z_max+1, :, :]
    out_img = sitk.GetImageFromArray(out_arr)
    out_img.SetOrigin(origin)
    out_img.SetDirection(direction)
    out_img.SetSpacing(spc)
    out_file = os.path.join(outdir, os.path.basename(infile).replace(inpattern, outpattern))
    sitk.WriteImage(out_img, out_file)
    



def extract_cerebral_parenchyma_singletask(infiles, outdir, inpattern='_NCCT.nii.gz', outpattern='_brain.nii.gz'):
    for infile in tqdm(infiles):
        extract_cerebral_parenchyma_onecase(infile, outdir, inpattern, outpattern)

def extract_cerebral_parenchyma_multiprocess(indir, outdir, inpattern='_NCCT.nii.gz', outpattern='_brain.nii.gz', process_num=6):
    
    infiles = glob(os.path.join(indir, '*{}'.format(inpattern)))

    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []

    num_per_process = (len(infiles) + process_num - 1)//process_num

    for i in range(process_num):
        sub_infiles = infiles[num_per_process*i:min(num_per_process*(i+1), len(infiles))]
        print(sub_infiles)
        result = pool.apply_async(extract_cerebral_parenchyma_singletask, args=(sub_infiles, outdir, inpattern, outpattern))
        results.append(result)

    pool.close()
    pool.join()

    # extract_cerebral_parenchyma_singletask(infiles, outdir, inpattern, outpattern)




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
        sub_infiles = infiles[num_per_process*i:min(num_per_process*(i+1), len(infiles))]
        result = pool.apply_async(ncct_generate_cerebral_parenchyma_single, args=(sub_infiles, outdir))
        results.append(result)

    pool.close()
    pool.join()



# 找出脑实质的最大层，并在最大层的上下各取60层（假设层厚为0.5mm）
def ncct_generate_cerebral_parenchyma_middle_layer_onecase(infile, outfile):
    '''
    ncct_generate_cerebral_parenchyma_middle_layer_onecase('../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct/406862_first_BS_brain.nii.gz', None)
    '''
    in_img = sitk.ReadImage(infile)
    in_arr = sitk.GetArrayFromImage(in_img)
    out_arr = np.zeros(in_arr.shape, dtype=in_arr.dtype)
    # 范围限定在(5, in_arr.shape[0]-5)，因为配准时脑实质图像的上下边缘生成有问题
    for z in range(5, in_arr.shape[0]-5):
        for y in range(in_arr.shape[1]):
            x_arr = in_arr[z,y,:]
            low_thres = 0
            # low_thres = -1024
            ranges = np.where(x_arr != low_thres)
            if len(ranges[0]) > 0:
                [x_min] = np.min(ranges, axis=1)
                [x_max] = np.max(ranges, axis=1)
                out_arr[z,y,x_min:x_max+1] = 1
    
    # 在保留的断层中，mask区域扩大到和最大层面面积相等
    max_region = np.max(out_arr, axis=0)
    layers = np.sum(out_arr, axis=(1,2))
    max_layer_index = np.argmax(layers)

    layer_delta = 64
    valid_range = list(range(max(0, max_layer_index-layer_delta), max_layer_index+layer_delta))

    for z in range(in_arr.shape[0]):
        if (z not in valid_range) or (layers[z]/max(layers) < 0.5):
            out_arr[z,:,:] = 0
        else:
            out_arr[z,:,:] = 1

    out_img = sitk.GetImageFromArray(out_arr)
    out_img.CopyInformation(in_img)
    out_file = outfile
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_file)
    writer.Execute(out_img)


def ncct_generate_cerebral_parenchyma_middle_layer_single(infiles, outdir):
    os.makedirs(outdir, exist_ok=True)
    for infile in tqdm(infiles):
        outfile = os.path.join(outdir, os.path.basename(infile))
        ncct_generate_cerebral_parenchyma_middle_layer_onecase(infile, outfile)


def ncct_generate_cerebral_parenchyma_middle_layer_multiprocess(indir, outdir, inpattern, process_num=12):
    
    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []
    
    os.makedirs(outdir, exist_ok=True)
    infiles = glob(os.path.join(indir, inpattern))
    
    num_per_process = (len(infiles) + process_num - 1)//process_num

    for i in range(process_num):
        sub_infiles = infiles[num_per_process*i:min(num_per_process*(i+1), len(infiles))]
        print(sub_infiles)
        result = pool.apply_async(ncct_generate_cerebral_parenchyma_middle_layer_single, args=(sub_infiles, outdir))
        results.append(result)

    pool.close()
    pool.join()        


# 找出脑实质的最大层，并在最大层的上下各取60层（假设层厚为0.5mm），每层保留只有脑实质的部分
def ncct_generate_cerebral_parenchyma_middle_layer_only_onecase(infile, outfile):
    '''
    ncct_generate_cerebral_parenchyma_middle_layer_only_onecase('../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct/406862_first_BS_brain.nii.gz', None)
    '''
    in_img = sitk.ReadImage(infile)
    
    in_arr = sitk.GetArrayFromImage(in_img)
    out_arr = np.zeros(in_arr.shape, dtype=in_arr.dtype)
    # 范围限定在(5, in_arr.shape[0]-5)，因为配准时脑实质图像的上下边缘生成有问题
    for z in range(5, in_arr.shape[0]-5):
        tmp_arr = in_arr[z]
        out_arr[z,tmp_arr>0] = 1

    # dilation
    tmp_img = sitk.GetImageFromArray(out_arr)
    tmp_img = sitk.Cast(tmp_img, sitk.sitkInt16)
    dilation_filter = sitk.BinaryDilateImageFilter()
    dilation_filter.SetForegroundValue(1)
    dilation_filter.SetBackgroundValue(0)
    dilation_filter.SetKernelRadius(3)
    tmp_img = dilation_filter.Execute(tmp_img)
    out_arr = sitk.GetArrayFromImage(tmp_img)
    
    # 在保留的断层中，mask区域扩大到和最大层面面积相等
    max_region = np.max(out_arr, axis=0)
    layers = np.sum(out_arr, axis=(1,2))
    max_layer_index = np.argmax(layers)

    layer_delta = 64
    valid_range = list(range(max(0, max_layer_index-layer_delta), max_layer_index+layer_delta))

    for z in range(in_arr.shape[0]):
        if (z not in valid_range) or (layers[z]/max(layers) < 0.5):
            out_arr[z,:,:] = 0
        else:
            pass

    out_img = sitk.GetImageFromArray(out_arr)
    out_img.CopyInformation(in_img)
    out_file = outfile
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(out_file)
    writer.Execute(out_img)

def ncct_generate_cerebral_parenchyma_middle_layer_only_singletask(infiles, outdir):
    os.makedirs(outdir, exist_ok=True)
    for infile in tqdm(infiles):
        outfile = os.path.join(outdir, os.path.basename(infile))
        ncct_generate_cerebral_parenchyma_middle_layer_only_onecase(infile, outfile)

def ncct_generate_cerebral_parenchyma_middle_layer_only_multiprocess(indir, outdir, inpattern, process_num=12):
    
    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []
    
    os.makedirs(outdir, exist_ok=True)
    infiles = glob(os.path.join(indir, inpattern))
    
    num_per_process = (len(infiles) + process_num - 1)//process_num

    for i in range(process_num):
        sub_infiles = infiles[num_per_process*i:min(num_per_process*(i+1), len(infiles))]
        print(sub_infiles)
        result = pool.apply_async(ncct_generate_cerebral_parenchyma_middle_layer_only_singletask, args=(sub_infiles, outdir))
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
        # ranges = np.where(brain_arr > 0)
        # [z_min, y_min, x_min] = np.min(np.array(ranges), axis=1)
        # [z_max, y_max, x_max] = np.max(np.array(ranges), axis=1)
        [z_min, y_min, x_min] = [0,0,0]
        [z_max, y_max, x_max] = brain_arr.shape
        z_max -= 1
        y_max -= 1
        x_max -= 1
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


def ncct_split_train_test_according_to_rapid_result(train_config_file, test_config_file, rapid_config_file):
    '''
    debug cmd: ncct_split_train_test_according_to_rapid_result('../data/gan/hospital_4/experiment_registration2/8.2.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt', '../data/gan/hospital_4/experiment_registration2/8.2.out/config/mask_ncct_to_dwi_bxxx_test_config_file.txt', '../data/gan/hospital_4/experiment_registration3/1.rapid/config.txt')
    invoke cmd: 
    '''
    infarct_pids = []
    penumbra_pids = []
    positive_pids = []
    with open(rapid_config_file) as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            ss = line.split('\t')
            if ss[1] == 'True':
                infarct_pids.append(ss[0])
            if ss[2] == 'True':
                penumbra_pids.append(ss[0])
            if ss[1] == 'True' or ss[2] == 'True':
                positive_pids.append(ss[0])
                
    def is_test(test_pids, line):
        for pid in test_pids:
            if pid in line:
                return pid
        return None

    # test_pids = positive_pids
    test_pids = infarct_pids

    positive_list = []

    with open(train_config_file) as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            pid = is_test(test_pids, line)
            if pid is not None:
                positive_list.append(line)
            else:
                pass

    with open(test_config_file) as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            pid = is_test(test_pids, line)
            if pid is not None:
                positive_list.append(line)
            else:
                pass

    out_dir = os.path.dirname(train_config_file)
    out_file = os.path.join(out_dir, 'positive_{}'.format(os.path.basename(train_config_file)))
    with open(out_file, 'w') as f:
        f.write('\n'.join(positive_list))
    
    
    


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

def cta_extract_infos_from_xlsx(info_file):
    '''
    info_file = '../data/gan/hospital_6/2 排除约40人后新增32人基于原六院CTA2DWI.xlsx'
    invoke cmd: python utils.py cta_extract_infos_from_xlsx '../data/gan/hospital_6/2 排除约40人后新增32人基于原六院CTA2DWI.xlsx'
    debug cmd: cta_extract_infos_from_xlsx('../data/gan/hospital_6/2 排除约40人后新增32人基于原六院CTA2DWI.xlsx')
    
    file head: 
    ['批次(1/2/3)', '影像号', 
    '1:入组 或 2:排除', 
    'CTA Series Instance UID', 'DWI Series Instance UID', 
    'CTA时间(DICOM tag)', 'DWI扫描时间(DICOM tag)', 
    'CTA与DWI时间间隔(DICOM tag)', 
    '性别', '年龄', '体重', '入院NIHSS', '出院NIHSS', '入院收缩压', '入院舒张压', '糖尿病', '高血压', 
    '房颤', '卒中史', '吸烟状态', '发病时间', 'DWI复查时间', '病变血管', '梗死区域', '治疗方式', '治疗时间', 
    'NCCT Series Instance UID', 'NCCT时间(DICOM tag)', 'DWI与NCCT时间间隔(DICOM tag)']
    '''
    
    wb = xlrd.open_workbook(info_file)
    sheet_names = wb.sheet_names()

    ws = wb.sheet_by_index(0)
    print(ws.row_values(0))

    pid_index = ws.row_values(0).index('影像号')
    valid_index = ws.row_values(0).index('1:入组 或 2:排除')
    cta_index = ws.row_values(0).index('CTA Series Instance UID')
    dwi_index = ws.row_values(0).index('DWI Series Instance UID')
    cta_dwi_delta_time_index = ws.row_values(0).index('CTA与DWI时间间隔(DICOM tag)')

    patient_infos = {}
    for i_r in range(1,ws.nrows):
        pid = str(int(ws.row_values(i_r)[pid_index]))
        if len(pid) == 0 or pid is None:
            continue
        valid_flag = str(int(ws.row_values(i_r)[valid_index]))
        if valid_flag != '1':
            continue
        patient_info = {}
        cta_uid = ws.row_values(i_r)[cta_index]
        dwi_uid = ws.row_values(i_r)[dwi_index]
        cta_dwi_delta_time = ws.row_values(i_r)[cta_dwi_delta_time_index]
        patient_info['pid'] = pid
        patient_info['cta_uid'] = cta_uid
        patient_info['dwi_uid'] = dwi_uid
        patient_info['cta_dwi_delta_time'] = cta_dwi_delta_time
        patient_infos[pid] = patient_info
    return patient_infos

def cta_extract_series_to_patient(indir, out_dir, info_file):
    '''
    debug: cta_extract_series_to_patient('../data/gan/hospital_6/ori', '../data/gan/hospital_6/0.raw_dcm', '../data/gan/hospital_6/2 排除约40人后新增32人基于原六院CTA2DWI.xlsx')
    invoke: python gan_utils.py cta_extract_series_to_patient '../data/gan/hospital_6/ori', '../data/gan/hospital_6/0.raw_dcm', '../data/gan/hospital_6/2 排除约40人后新增32人基于原六院CTA2DWI.xlsx'
    
    indir:
    tree -L 1
    .
    ├── 1.2.156.112605.189250946103856.190725032121.3.5972.26584
    ├── 1.2.156.112605.189250946103856.190804032608.3.6024.213494
    ├── 1.2.156.112605.189250946103856.190903010933.3.6016.113322
    ├── 1.2.156.112605.189250946103856.191104010418.3.6364.115889
    ├── 1.2.156.112605.189250946103856.191104141450.3.6364.118948

    out_dir:
    tree -L 4
    └── 5023941
        ├── DWI
        │   └── 1.3.12.2.1107.5.2.36.40534.2020040916392479035839142.0.0.0
        │       ├── b0
        │       ├── bxxx
        │       └── not_dwi
        └── NCCT
            └── 1.2.156.112605.189250946103856.200406031912.3.5724.111797
                ├── 000001.dcm
                ├── 000002.dcm
                ├── 000003.dcm
                ├── 000004.dcm

    info_file:
    file head: 
    ['批次(1/2/3)', '影像号', 
    '1:入组 或 2:排除', 
    'CTA Series Instance UID', 'DWI Series Instance UID', 
    'CTA时间(DICOM tag)', 'DWI扫描时间(DICOM tag)', 
    'CTA与DWI时间间隔(DICOM tag)', 
    '性别', '年龄', '体重', '入院NIHSS', '出院NIHSS', '入院收缩压', '入院舒张压', '糖尿病', '高血压', 
    '房颤', '卒中史', '吸烟状态', '发病时间', 'DWI复查时间', '病变血管', '梗死区域', '治疗方式', '治疗时间', 
    'NCCT Series Instance UID', 'NCCT时间(DICOM tag)', 'DWI与NCCT时间间隔(DICOM tag)']

    '''
    patient_infos = cta_extract_infos_from_xlsx(info_file)
    for key, patient_info in patient_infos.items():
        cta_uid = patient_info['cta_uid']
        dwi_uid = patient_info['dwi_uid']
        cta_path = os.path.join(indir, cta_uid)
        dwi_path = os.path.join(indir, dwi_uid)
        if not os.path.isdir(cta_path):
            continue
        if not os.path.isdir(dwi_path):
            continue
        out_ncct_series = os.path.join(out_dir, key, 'NCCT', cta_uid)
        shutil.copytree(cta_path, out_ncct_series)
        not_exist_list = ['2602401', '3878650', '1847547', '3926192', '2123180', '3835991', '4023216']
        if key not in not_exist_list:
            continue

        out_dwi_series = os.path.join(out_dir, key, 'DWI', dwi_uid)
        cta_extract_dwi_from_raw_dwi_single(dwi_path, out_dwi_series)


def cta_split_train_test_according_to_xlsx(info_file, train_config_file, test_config_file, out_file_prefix='anno'):
    '''
    this function is hard code!!!!!!!!

    info_file = '../data/gan/hospital_6/CTA ASPECT 总表 V11.xlsx'
    train_config_file = '../data/gan/hospital_6/experiment_registration2/8.2.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt'
    test_config_file = '../data/gan/hospital_6/experiment_registration2/8.2.out/config/mask_ncct_to_dwi_bxxx_test_config_file.txt'

    debug cmd: cta_split_train_test_according_to_xlsx('../data/gan/hospital_6/CTA ASPECT 总表 V11.xlsx', '../data/gan/hospital_6/experiment_registration2/8.2.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt', '../data/gan/hospital_6/experiment_registration2/8.2.out/config/mask_ncct_to_dwi_bxxx_test_config_file.txt')
    invoke cmd: python gan_utils.py cta_split_train_test_according_to_xlsx '../data/gan/hospital_6/CTA ASPECT 总表 V11.xlsx' '../data/gan/hospital_6/experiment_registration2/8.2.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt' '../data/gan/hospital_6/experiment_registration2/8.2.out/config/mask_ncct_to_dwi_bxxx_test_config_file.txt'
    '''
    wb = xlrd.open_workbook(info_file)
    sheet_names = wb.sheet_names()

    ws = wb.sheet_by_index(0)
    print(ws.row_values(0))

    test_pids = []
    test_exist = []
    test_not_exist = []
    for i_r in range(2,71):
        pid = str(int(ws.row_values(i_r)[0]))
        test_pids.append(pid)
    train_list = []
    test_list = []

    def is_test(test_pids, line):
        for pid in test_pids:
            if pid in line:
                return pid
        return None

    with open(train_config_file) as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            pid = is_test(test_pids, line)
            if pid is not None:
                test_list.append(line)
                test_exist.append(pid)
            else:
                train_list.append(line)

    with open(test_config_file) as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            pid = is_test(test_pids, line)
            if  pid is not None:
                test_list.append(line)
                test_exist.append(pid)
            else:
                train_list.append(line)

    for t in test_pids:
        if t not in test_exist:
            test_not_exist.append(t)

    print('====> pid not exist:\t{}'.format(test_not_exist))

    print('train list:\t{}'.format(len(train_list)))
    print('test list:\t{}'.format(len(test_list)))

    outdir = os.path.dirname(train_config_file)
    out_train_config_file = os.path.join(outdir, '{}_{}'.format(out_file_prefix, os.path.basename(train_config_file)))
    out_test_config_file = os.path.join(outdir, '{}_{}'.format(out_file_prefix, os.path.basename(test_config_file)))

    with open(out_train_config_file, 'w') as f:
        f.write('\n'.join(train_list))

    with open(out_test_config_file, 'w') as f:
        f.write('\n'.join(test_list))


class SeriesInfo:
    # def __init__(self, origin, direction, spacing):
    #     self.origin = origin
    #     self.direction = direction
    #     self.spacing = spacing
    def __init__(self):
        super().__init__()
        self.image = None

    def getInfoFromImage(self, image):
        self.origin = image.GetOrigin()
        self.direction = image.GetDirection()
        self.spacing = image.GetSpacing()

    def getInfoFromImageFile(self, image_file, is_dcm=False):
        if is_dcm:
            image = read_dcm_file(image_file)
        else:
            image = sitk.ReadImage(image_file)
        self.getInfoFromImage(image)
        self.image = image

    def saveImage(self, outfile):
        if self.image is not None:
            sitk.WriteImage(self.image, outfile)



## rapid 相关操作
def rapid_extract_summary_info_dcm0_core_infarct_area(dcm_file, adc_image, patient_id, out_dir):
    '''
    核心梗死区
    1. rapid图像分辨率256x256(spacing:0.898438\0.898438), dwi/adc分辨率192x192(spacing:1.1979166269302\1.1979166269302),可以按照倍数换算；
    2. rapid的层数与adc的层数不一定一样；
    '''
    from skimage import transform

    os.makedirs(out_dir, exist_ok=True)
    image = sitk.ReadImage(dcm_file)
    arr = sitk.GetArrayFromImage(image)
    
    dwi_arr = np.zeros([20, 256, 256, 3], dtype=np.uint8)

    mask_arr = np.zeros([20, 256, 256], dtype=np.uint8)
    # 图像从中间分开, 一行20例数据，其中五例dwi, 五例mrp
    # 四行，一共20例数据, 每例数据的size:256x256
    single_h = 256
    single_w = 256

    [img_h, img_w] = arr.shape[1:3]

    assert img_w/single_w == img_w//single_w
    background_cnt = 0
    for ih in range(4):
        for iw in range(5):
            iz = ih*5+iw
            tmp_layer = arr[0, ih*single_h:(ih+1)*single_h, iw*single_w:(iw+1)*single_w, :]
            tmp_sum = np.sum(tmp_layer)
            # print('tmp_sum:\t{}'.format(tmp_sum))
            if tmp_sum < 10:
                background_cnt += 1
            dwi_arr[iz,:,:, :] = arr[0, ih*single_h:(ih+1)*single_h, iw*single_w:(iw+1)*single_w, :]
            # dwi_arr[iz,:,:, :] = arr[0, 256:512, 256:512, :]
    if background_cnt > 0:
        # print('{}\tbackground:\t{}'.format(dcm_file, background_cnt))
        return False, background_cnt
    # dwi_img = sitk.GetImageFromArray(dwi_arr)
    # dwi_img.SetOrigin(adc_info.origin)
    # dwi_img.SetDirection(adc_info.direction)
    # dwi_img.SetSpacing(adc_info.spacing)
    out_dwi_file = os.path.join(out_dir, '{}_first_FU_DWI_INFARCT.nii.gz'.format(patient_id))
    # sitk.WriteImage(dwi_img, out_dwi_file)
    [w,h,d] = adc_image.GetSize()
    resize_dwi_arr = np.zeros([d,h,w,3])
    for iz in range(dwi_arr.shape[0]):
        resize_dwi_arr[iz] = cv2.resize(dwi_arr[iz],(h,w))
    dwi_img = sitk.GetImageFromArray(resize_dwi_arr)
    dwi_img.CopyInformation(adc_image)
    sitk.WriteImage(dwi_img, out_dwi_file)
    infarct_mask_arr = dwi_arr[:,:,:,0] - dwi_arr[:,:,:,1]
    infarct_mask_arr[infarct_mask_arr != 255] = 0
    infarct_mask_arr[infarct_mask_arr == 255] = 1
    resized_infarct_mask_arr = np.zeros(adc_image.GetSize()[::-1])
    for iz in range(infarct_mask_arr.shape[0]):
        resized_infarct_mask_arr[iz] = cv2.resize(infarct_mask_arr[iz],(h,w))
    infarct_mask_img = sitk.GetImageFromArray(resized_infarct_mask_arr)
    infarct_mask_img.CopyInformation(adc_image)
    # infarct_mask_img.SetOrigin(adc_info.origin)
    # infarct_mask_img.SetDirection(adc_info.direction)
    # infarct_mask_img.SetSpacing(adc_info.spacing)
    infarct_mask_file = os.path.join(out_dir, '{}_first_FU_DWI_INFARCT_MASK.nii.gz'.format(patient_id))
    sitk.WriteImage(infarct_mask_img, infarct_mask_file)

    #检查是否存在核心梗死区
    infarct_area = np.sum(infarct_mask_arr)
    infarct_spc = infarct_mask_img.GetSpacing()
    infarct_size = infarct_mask_img.GetSize()
    infarct_volume = infarct_area * infarct_spc[0] * infarct_spc[1] * infarct_spc[2] * infarct_size[0] * infarct_size[1]/(256*256)
    # print('{} infarct volume:\t{}\t{:1d}ml'.format(dcm_file, patient_id, int(infarct_volume/1000)))
    print('infarct volume:\t{}\t{:1d}ml'.format(patient_id, int(infarct_volume/1000)))
    if infarct_area > 5:
        # print('{} infarct area:\t{}'.format(dcm_file, infarct_area))
        # print('{} infarct volume:\t{:1f}'.format(dcm_file, infarct_area))
        return True, None
    else:
        return False, None



def rapid_extract_summary_info_dcm0_ischemic_penumbra(dcm_file, adc_image, patient_id, out_dir):
    '''
    核心梗死区
    1. rapid图像分辨率256x256(spacing:0.898438\0.898438), dwi/adc分辨率192x192(spacing:1.1979166269302\1.1979166269302),可以按照倍数换算；
    2. rapid的层数与adc的层数不一定一样；
    '''
    from skimage import transform

    os.makedirs(out_dir, exist_ok=True)
    image = sitk.ReadImage(dcm_file)
    arr = sitk.GetArrayFromImage(image)
    
    dwi_arr = np.zeros([20, 256, 256, 3], dtype=np.uint8)

    mask_arr = np.zeros([20, 256, 256], dtype=np.uint8)
    # 图像从中间分开, 一行20例数据，其中五例dwi, 五例mrp
    # 四行，一共20例数据, 每例数据的size:256x256
    single_h = 256
    single_w = 256

    [img_h, img_w] = arr.shape[1:3]

    assert img_w/single_w == img_w//single_w
    background_cnt = 0
    bias_w = 256*5
    for ih in range(4):
        for iw in range(5):
            iz = ih*5+iw
            tmp_layer = arr[0, ih*single_h:(ih+1)*single_h, iw*single_w:(iw+1)*single_w, :]
            tmp_sum = np.sum(tmp_layer)
            # print('tmp_sum:\t{}'.format(tmp_sum))
            if tmp_sum < 10:
                background_cnt += 1
            dwi_arr[iz,:,:, :] = arr[0, ih*single_h:(ih+1)*single_h, iw*single_w+bias_w:(iw+1)*single_w+bias_w, :]
            # dwi_arr[iz,:,:, :] = arr[0, 256:512, 256:512, :]
    # print('background:\t{}'.format(background_cnt))
    if background_cnt > 0:
        # print('{}\tbackground:\t{}'.format(dcm_file, background_cnt))
        return False, background_cnt
    # dwi_img = sitk.GetImageFromArray(dwi_arr)
    # dwi_img.SetOrigin(adc_info.origin)
    # dwi_img.SetDirection(adc_info.direction)
    # dwi_img.SetSpacing(adc_info.spacing)
    out_dwi_file = os.path.join(out_dir, '{}_first_FU_ISCHEMIC_PENUMBRA.nii.gz'.format(patient_id))
    # sitk.WriteImage(dwi_img, out_dwi_file)
    [w,h,d] = adc_image.GetSize()
    resize_dwi_arr = np.zeros([d,h,w,3])
    for iz in range(dwi_arr.shape[0]):
        resize_dwi_arr[iz] = cv2.resize(dwi_arr[iz],(h,w))
    dwi_img = sitk.GetImageFromArray(resize_dwi_arr)
    dwi_img.CopyInformation(adc_image)
    sitk.WriteImage(dwi_img, out_dwi_file)
    infarct_mask_arr = dwi_arr[:,:,:,1] - dwi_arr[:,:,:,0]
    infarct_mask_arr[infarct_mask_arr != 255] = 0
    infarct_mask_arr[infarct_mask_arr == 255] = 1
    resized_infarct_mask_arr = np.zeros(adc_image.GetSize()[::-1])
    for iz in range(infarct_mask_arr.shape[0]):
        resized_infarct_mask_arr[iz] = cv2.resize(infarct_mask_arr[iz],(h,w))
    infarct_mask_img = sitk.GetImageFromArray(resized_infarct_mask_arr)
    infarct_mask_img.CopyInformation(adc_image)
    # infarct_mask_img.SetOrigin(adc_info.origin)
    # infarct_mask_img.SetDirection(adc_info.direction)
    # infarct_mask_img.SetSpacing(adc_info.spacing)
    infarct_mask_file = os.path.join(out_dir, '{}_first_FU_ISCHEMIC_PENUMBRA_MASK.nii.gz'.format(patient_id))
    sitk.WriteImage(infarct_mask_img, infarct_mask_file)

    #检查是否存在核心梗死区
    infarct_area = np.sum(infarct_mask_arr)
    if infarct_area > 5:
        # print('{} infarct area:\t{}'.format(dcm_file, infarct_area))
        return True, None
    else:
        return False, None


# def rapid_extract_summary_info_dcm0_ischemic_penumbra(dcm_file, adc_info, patient_id, out_dir):
#     '''
#     缺血半暗带
#     '''
#     os.makedirs(out_dir, exist_ok=True)
#     image = sitk.ReadImage(dcm_file)
#     arr = sitk.GetArrayFromImage(image)
    
#     dwi_arr = np.zeros([20, 256, 256, 3], dtype=np.uint8)

#     mask_arr = np.zeros([20, 256, 256], dtype=np.uint8)
#     # 图像从中间分开, 一行20例数据，其中五例dwi, 五例mrp
#     # 四行，一共20例数据, 每例数据的size:256x256
#     single_h = 256
#     single_w = 256

#     [img_h, img_w] = arr.shape[1:3]

#     assert img_w/single_w == img_w//single_w
#     bias_w = 256*5
#     for ih in range(4):
#         for iw in range(5):
#             iz = ih*5+iw
#             dwi_arr[iz,:,:, :] = arr[0, ih*single_h:(ih+1)*single_h, iw*single_w+bias_w:(iw+1)*single_w+bias_w, :]
#             # dwi_arr[iz,:,:, :] = arr[0, 256:512, 256:512, :]
#     dwi_img = sitk.GetImageFromArray(dwi_arr)
#     dwi_img.SetOrigin(adc_info.origin)
#     dwi_img.SetDirection(adc_info.direction)
#     dwi_img.SetSpacing(adc_info.spacing)
#     out_dwi_file = os.path.join(out_dir, '{}_first_FU_ISCHEMIC_PENUMBRA.nii.gz'.format(patient_id))
#     sitk.WriteImage(dwi_img, out_dwi_file)
#     infarct_mask_arr = dwi_arr[:,:,:,1] - dwi_arr[:,:,:,0]
#     infarct_mask_arr[infarct_mask_arr != 255] = 0
#     infarct_mask_arr[infarct_mask_arr == 255] = 1
#     infarct_mask_img = sitk.GetImageFromArray(infarct_mask_arr)
#     infarct_mask_img.SetOrigin(adc_info.origin)
#     infarct_mask_img.SetDirection(adc_info.direction)
#     infarct_mask_img.SetSpacing(adc_info.spacing)
#     infarct_mask_file = os.path.join(out_dir, '{}_first_FU_ISCHEMIC_PENUMBRA_MASK.nii.gz'.format(patient_id))
#     sitk.WriteImage(infarct_mask_img, infarct_mask_file)

#     #检查是否存在核心梗死区
#     infarct_area = np.sum(infarct_mask_arr)
#     if infarct_area > 5:
#         print('{} infarct area:\t{}'.format(dcm_file, infarct_area))
#         return True
#     else:
#         return False
    

def rapid_extract_sumary_info(rapid_series_path, adc_path, patient_id, outdir):
    '''
    rapid_series_path: '../data/gan/hospital_4/0.raw_dcm/114093/RAPID/1.3.6.1.4.1.39822.1.3.8323328.7953.1535266088.657316'
    debug: rapid_extract_sumary_info('../data/gan/hospital_4/0.raw_dcm/114093/RAPID/1.3.6.1.4.1.39822.1.3.8323328.7953.1535266088.657316', '../data/gan/hospital_4/0.raw_dcm/114093/ADC/1.3.12.2.1107.5.2.30.26961.2018082614324377426400913.0.0.0', '114093', '../data/gan/hospital_4/tmp')


    return val:
        val 0: 是否有核心梗死区
        val 1: 是否有缺血半暗带
        val 2: 是否能够和ADC/DWI图像进行匹配
    '''
    rapid_files1 = glob(os.path.join(rapid_series_path, '*.DCM'))
    rapid_files2 = glob(os.path.join(rapid_series_path, '*.dcm'))

    rapid_files = rapid_files1 + rapid_files2
    rapid_files.sort()

    rapid_file = rapid_files[0]
    adc_image = read_dcm_file(adc_path)
    is_infcrct, is_infcrct_match = rapid_extract_summary_info_dcm0_core_infarct_area(rapid_files[0], adc_image, patient_id, outdir)
    is_penumbra, is_penumbra_match = rapid_extract_summary_info_dcm0_ischemic_penumbra(rapid_files[0], adc_image, patient_id, outdir)
    out_adc_file = os.path.join(outdir, '{}_ADC.nii.gz'.format(patient_id))
    sitk.WriteImage(adc_image, out_adc_file)
    return is_infcrct, is_penumbra, is_infcrct_match

def rapid_extract_sumary_info_multiprocess(indir, outdir):
    '''
    debug cmd: rapid_extract_sumary_info_multiprocess('../data/gan/hospital_4/0.raw_dcm', '../data/gan/hospital_4/1.rapid')
    invoke cmd: python gan_utils.py rapid_extract_sumary_info_multiprocess '../data/gan/hospital_4/0.raw_dcm' '../data/gan/hospital_4/1.rapid'
    '''
    cnt = 0
    config_infos = []
    for patient_id in os.listdir(indir):
        patient_path = os.path.join(indir, patient_id)
        if not os.path.isdir(patient_path):
            continue
        rapid_path = os.path.join(os.path.join(patient_path, 'RAPID'))
        adc_path = os.path.join(os.path.join(patient_path, 'ADC'))
        if not os.path.isdir(rapid_path):
            continue
        if not os.path.isdir(adc_path):
            continue
        series_uids = os.listdir(rapid_path)
        adc_uids = os.listdir(adc_path)
        if len(series_uids) != 1:
            continue
        if len(adc_uids) != 1:
            continue
        series_uid = series_uids[0]
        series_path = os.path.join(rapid_path, series_uid)
        adc_uid = adc_uids[0]
        adc_path = os.path.join(adc_path, adc_uid)
        if not os.path.isdir(series_path):
            continue
        if not os.path.isdir(adc_path):
            continue
        # adc_info = SeriesInfo()
        # adc_info.getInfoFromImageFile(adc_path, is_dcm=True)
        # out_adc_file = os.path.join(outdir, '{}_ADC.nii.gz'.format(patient_id))
        # adc_info.saveImage(out_adc_file)
        is_infcrct, is_penumbra, is_infcrct_match = rapid_extract_sumary_info(series_path, adc_path, patient_id, outdir)
        if is_infcrct_match is not None:
            continue
        config_info = '{}\t{}\t{}'.format(patient_id, is_infcrct, is_penumbra)
        config_infos.append(config_info)
        cnt += 1
        # break
    with open(os.path.join(outdir, 'config.txt'), 'w') as f:
        f.write('\n'.join(config_infos))
    print('rapid count is:\t{}'.format(cnt))
    

# 检查是否所有有RAPID结果的数据，都同时存在MRP数据
def rapid_check_rapid_mrp_both_exist(indir):
    '''
    rapid_check_rapid_mrp_both_exist('../data/gan/hospital_4/0.raw_dcm')
    '''
    pids = os.listdir(indir)
    for pid in pids:
        patient_path = os.path.join(indir, pid)
        if not os.path.isdir(patient_path):
            continue
        mrp_path = os.path.join(patient_path, 'MRP')
        rapid_path = os.path.join(patient_path, 'RAPID')
        if not os.path.isdir(rapid_path):
            continue
        if not os.path.isdir(mrp_path):
            print(mrp_path)
            continue
        print('rapid:{}\tmrp:{}'.format(rapid_path, mrp_path))
        
# 统计包含病变的数据数目
def rapid_stat_dwi_positive_count_according_to_config(config_file):
    '''
    config file format as follows:
    pid     是否包含核心梗死区  是否包含缺血半暗带
    456926  False   False
    392112  False   False
    270200  False   False
    124639  False   False
    450950  False   False
    447276  False   False
    452468  False   False
    316872  False   False
    446273  False   True
    '''
    '''

    debug cmd: rapid_stat_dwi_positive_count_according_to_config('../data/gan/hospital_4/1.rapid/config.txt')
    invoke cmd: python gan_utils.py rapid_stat_dwi_positive_count_according_to_config '../data/gan/hospital_4/1.rapid/config.txt'

    4院第一批数据（hospital_4）的运行结果如下：

    hospital 4 ncct dwi pairs total count:  51
    hospital 4 ncct dwi pairs include infarct count:        12
    hospital 4 ncct dwi pairs include penumbra count:       25
    hospital 4 ncct dwi pairs positive count:       26


    debug cmd: rapid_stat_dwi_positive_count_according_to_config('../data/gan/hospital_4_2/1.rapid/config.txt')
    invoke cmd: python gan_utils.py rapid_stat_dwi_positive_count_according_to_config '../data/gan/hospital_4_2/1.rapid/config.txt'

    4院第二批数据（hospital_4_2）的运行结果如下：
    hospital 4 ncct dwi pairs total count:  133
    hospital 4 ncct dwi pairs include infarct count:        29
    hospital 4 ncct dwi pairs include penumbra count:       52
    hospital 4 ncct dwi pairs positive count:       60
    '''
    total_cnt = 0
    infarct_cnt = 0
    penumbra_cnt = 0
    positive_cnt = 0
    with open(config_file) as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            ss = line.split('\t')
            if ss[1] == 'True':
                infarct_cnt += 1
            if ss[2] == 'True':
                penumbra_cnt += 1
            if ss[1] == 'True' or ss[2] == 'True':
                positive_cnt += 1
            total_cnt += 1
    print('hospital 4 ncct dwi pairs total count:\t{}'.format(total_cnt))
    print('hospital 4 ncct dwi pairs include infarct count:\t{}'.format(infarct_cnt))
    print('hospital 4 ncct dwi pairs include penumbra count:\t{}'.format(penumbra_cnt))
    print('hospital 4 ncct dwi pairs positive count:\t{}'.format(positive_cnt))
    
            

# 批量修改文件名字
def change_names_batch(indir, outdir, inpattern, outpattern):
    '''
    debug cmd: change_names_batch('../data/gan/hospital_6/experiment_registration2/4 Patient_nii_unity', '../data/gan/hospital_6/experiment_registration2/4 Patient_nii_unity', '_NCCT.nii.gz', '_NCCT_bk.nii.gz')
    invoke: python gan_utils.py change_names_batch '../data/gan/hospital_6/experiment_registration2/4 Patient_nii_unity' '../data/gan/hospital_6/experiment_registration2/4 Patient_nii_unity' '_NCCT.nii.gz' '_NCCT_bk.nii.gz'
    '''
    os.makedirs(outdir, exist_ok=True)
    infiles = glob(os.path.join(indir, '*{}'.format(inpattern)))
    for infile in infiles:
        outfile = os.path.join(outdir, os.path.basename(infile).replace(inpattern, outpattern))
        os.rename(infile, outfile)


def copy_infarct_data(indir, outdir, config_file):
    '''
    indir: ../data/gan/hospital_6_crop/experiment_registration2/8.2.out
    tree -L 1
    .
    ├── ADC -> DWI_BXXX
    ├── cerebral_parenchyma
    ├── config
    ├── DWI_B0 -> DWI_BXXX
    ├── DWI_BX

    outdir: ../data/gan/hospital_6_crop/experiment_registration2/8.2.out_infarct

    debug cmd: copy_infarct_data('../data/gan/hospital_6_crop/experiment_registration2/8.2.out', '../data/gan/hospital_6_crop/experiment_registration2/8.2.out_infarct', '../data/gan/hospital_4_2_3d/1.rapid/config.txt')
    invoke cmd: python gan_utils.py copy_infarct_data '../data/gan/hospital_6_crop/experiment_registration2/8.2.out' '../data/gan/hospital_6_crop/experiment_registration2/8.2.out_infarct' '../data/gan/hospital_4_2_3d/1.rapid/config.txt'
    '''
    in_ncct_dir = os.path.join(indir, 'NCCT')
    in_dwi_bxxx_dir = os.path.join(indir, 'DWI_BXXX')
    ncct_pattern = '_first_BS_NCCT.nii.gz'
    dwi_bxxx_pattern = '_first_FU_DWI_BXXX.nii.gz'
    out_ncct_dir = os.path.join(outdir, 'NCCT')
    out_dwi_bxxx_dir = os.path.join(outdir, 'DWI_BXXX')
    os.makedirs(out_ncct_dir, exist_ok=True)
    os.makedirs(out_dwi_bxxx_dir, exist_ok=True)
    with open(config_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                if len(ss) != 3:
                    continue
                if ss[1] != 'True':
                    continue
                pid = ss[0]
                src_ncct_file = os.path.join(in_ncct_dir, '{}{}'.format(pid, ncct_pattern))
                if not os.path.isfile(src_ncct_file):
                    continue
                src_dwi_bxxx_file = os.path.join(in_dwi_bxxx_dir, '{}{}'.format(pid, dwi_bxxx_pattern))
                dst_ncct_file = os.path.join(out_ncct_dir, '{}{}'.format(pid, ncct_pattern))
                dst_dwi_bxxx_file = os.path.join(out_dwi_bxxx_dir, '{}{}'.format(pid, dwi_bxxx_pattern))
                shutil.copyfile(src_ncct_file, dst_ncct_file)
                shutil.copyfile(src_dwi_bxxx_file, dst_dwi_bxxx_file)



# 批量将三维数据的3个截面保存
def extract_mpr_one_case(infile, outdir, is_dcm=False):
    '''
    数据数据为做过resample（三个维度上spacing一致）的nii.gz格式
    debug cmd: extract_mpr('../data/gan/hospital_6/experiment_registration2/8.2.out/NCCT/1014186_first_BS_NCCT.nii.gz', '../data/gan/hospital_6/experiment_registration2/8.2.out/projection')
    debug cmd: extract_mpr('../data/gan/hospital_6/experiment_registration2/8.2.out/DWI_BXXX/1014186_first_FU_DWI_BXXX.nii.gz', '../data/gan/hospital_6/experiment_registration2/8.2.out/projection')
    '''
    os.makedirs(outdir, exist_ok=True)
    image = sitk.ReadImage(infile)
    arr = sitk.GetArrayFromImage(image)
    [z,y,x] = arr.shape
    z_plane = arr[z//2, :, :]
    y_plane = arr[:, y//2, :]
    x_plane = arr[:,:,x//2]
    
    z_name = os.path.basename(infile).split('.')[0]+'_z.jpg'
    z_name = os.path.join(outdir, z_name)
    y_name = os.path.basename(infile).split('.')[0]+'_y.jpg'
    y_name = os.path.join(outdir, y_name)
    x_name = os.path.basename(infile).split('.')[0]+'_x.jpg'
    x_name = os.path.join(outdir, x_name)
    cv2.imwrite(z_name, z_plane)
    cv2.imwrite(y_name, y_plane)
    cv2.imwrite(x_name, x_plane)

def extract_mpr_singletask(infiles, outdir):
    for infile in tqdm(infiles):
        extract_mpr_one_case(infile, outdir)

def extract_mpr_multiprocess(indir, outdir, process_num=12):
    '''
    python gan_utils.py extract_mpr_multiprocess '../data/gan/hospital_6/experiment_registration2/8.2.out/NCCT' '../data/gan/hospital_6/experiment_registration2/8.2.out/projection'
    python gan_utils.py extract_mpr_multiprocess '../data/gan/hospital_6/experiment_registration2/8.2.out/DWI_BXXX' '../data/gan/hospital_6/experiment_registration2/8.2.out/projection'
    '''

    infiles = glob(os.path.join(indir, '*.nii.gz'))
    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []

    num_per_process = (len(infiles) + process_num - 1)//process_num

    for i in range(process_num):
        sub_infiles = infiles[num_per_process*i:min(num_per_process*(i+1), len(infiles))]
        print(sub_infiles)
        result = pool.apply_async(extract_mpr_singletask, args=(sub_infiles, outdir))
        results.append(result)

    pool.close()
    pool.join()


# statistc 

def cta_stat_lesion_volume_size(indir):
    '''
    统计标注好的real dwi的病灶区域的体积（单位：ml），输出路径：os.path.join(indir, stat_result_real_dwi.csv)
    debug cmd: cta_stat_lesion_volume_size('../data/gan/hospital_6')
    '''
    raw_ct_dir = os.path.join(indir, '0.raw_dcm')
    annotation_dir = os.path.join(indir, 'annotation')
    annotation_dwi_dir = os.path.join(indir, 'annotation', 'real_dwi')
    dwi_files = glob(os.path.join(annotation_dwi_dir, '*.mha'))
    csv_file1 = os.path.join(annotation_dir, 'image_anno_TASK_2694.csv')
    csv_file2 = os.path.join(annotation_dir, 'Series_real DWI.xls')

    # annotation_dwi_dir = os.path.join(indir, 'annotation', 'fake_dwi')
    # dwi_files = glob(os.path.join(annotation_dwi_dir, '*.mha'))
    # csv_file1 = os.path.join(annotation_dir, 'image_anno_TASK_2695.csv')
    # csv_file2 = os.path.join(annotation_dir, 'Series_virtual DWI.xls')

    # annotation_dwi_dir = os.path.join(indir, 'annotation', 'fake_dwi_liaren_split', '5283')
    # dwi_files = glob(os.path.join(annotation_dwi_dir, '*.mha'))
    # csv_file1 = os.path.join(annotation_dir, 'image_anno_TASK_2823.csv')
    # csv_file2 = os.path.join(annotation_dir, 'Series_virtual DWI.xls')

    # annotation_dwi_dir = os.path.join(indir, 'annotation', 'fake_dwi_liaren_split', '5284')
    # dwi_files = glob(os.path.join(annotation_dwi_dir, '*.mha'))
    # csv_file1 = os.path.join(annotation_dir, 'image_anno_TASK_2823.csv')
    # csv_file2 = os.path.join(annotation_dir, 'Series_virtual DWI.xls')

    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_excel(csv_file2)
    imageid_to_patientid = {}
    for index, row in df1.iterrows():
        image_id = row['影像结果编号']
        patient_id = df2[df2['序列号'] == row['序列编号']]['原始路径'].values[0]
        patient_id = os.path.basename(patient_id).split('_')[0]
        imageid_to_patientid[str(image_id)] = str(patient_id)
        # if patient_id == '3901698':
        #     print(image_id)
    patient_ids = []
    lesion_volumes = []
    row_elems = []
    for dwi_file in tqdm(dwi_files):
        basename = os.path.basename(dwi_file)
        patient_id = basename.split('.')[0]
        cta_dir = os.path.join(raw_ct_dir, '{}/NCCT'.format(imageid_to_patientid[patient_id]))
        cta_series_uid = os.listdir(cta_dir)[0]
        cta_series_uid = os.path.join(cta_dir, cta_series_uid)
        cta_image = read_dcm_file(cta_series_uid)
        dwi_image = sitk.ReadImage(dwi_file)
        dwi_arr = sitk.GetArrayFromImage(dwi_image)
        spc = cta_image.GetSpacing()
        # lesion_volume = int(np.sum(dwi_arr)*spc[0]*spc[0]*spc[0]/1000)
        lesion_volume = np.round(np.sum(dwi_arr)*spc[0]*spc[0]*spc[0]/1000, 1)
        patient_ids.append(imageid_to_patientid[patient_id])
        lesion_volumes.append(lesion_volume)
        row_elems.append(np.array([imageid_to_patientid[patient_id], lesion_volume]))
        # print('{} volume is:\t{}'.format(imageid_to_patientid[patient_id], lesion_volume))
    df = pd.DataFrame(np.array(row_elems), columns=['pid', 'real dwi volumes(ml)'])
    df.to_csv(os.path.join(annotation_dir, 'stat_result_real_dwi.csv'))
    # df.to_csv(os.path.join(annotation_dir, 'stat_result_fake_dwi.csv'))
    # df.to_csv(os.path.join(annotation_dir, 'stat_result_fake_dwi_5283.csv'))
    # df.to_csv(os.path.join(annotation_dir, 'stat_result_fake_dwi_5284.csv'))

def cta_stat_calc_dice(gt_file, pred_file):
    '''
    debug cmd: cta_stat_calc_dice('../data/gan/hospital_6/annotation/anno_result/2313573.mha', '../data/gan/hospital_6/annotation/anno_result/2396943.mha')
    '''
    gt_img = sitk.ReadImage(gt_file)
    pred_img = sitk.ReadImage(pred_file)
    # print(gt_img.GetSize())
    # print(pred_img.GetSize())
    gt_arr = sitk.GetArrayFromImage(gt_img)
    pred_arr = sitk.GetArrayFromImage(pred_img)
    pred_arr = pred_arr[:gt_arr.shape[0],:,:]
    intersect = (gt_arr*pred_arr).sum()
    denominator = gt_arr.sum() + pred_arr.sum()
    smooth = 1e-8
    dice = 2 * ((intersect + smooth) / (denominator + smooth))
    return dice

def cta_stat_lesion_volume_dice(indir):
    '''
    debug cmd: cta_stat_lesion_volume_dice('../data/gan/hospital_6')
    '''
    # 找到统一patientid对应的real标注，virtual标注0，virtual标注1，virtual标注2
    annotation_dir = os.path.join(indir, 'annotation')
    csv_file1 = os.path.join(annotation_dir, 'image_anno_TASK_2694.csv')
    csv_file2 = os.path.join(annotation_dir, 'Series_real DWI.xls')
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_excel(csv_file2)
    imageid_to_patientid = {}
    patientid_to_image_id_real = {}
    for index, row in df1.iterrows():
        image_id = row['影像结果编号']
        patient_id = df2[df2['序列号'] == row['序列编号']]['原始路径'].values[0]
        patient_id = os.path.basename(patient_id).split('_')[0]
        imageid_to_patientid[str(image_id)] = str(patient_id)
        patientid_to_image_id_real[str(patient_id)] = str(image_id)

    csv_file1 = os.path.join(annotation_dir, 'image_anno_TASK_2695.csv')
    csv_file2 = os.path.join(annotation_dir, 'Series_virtual DWI.xls')
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_excel(csv_file2)
    imageid_to_patientid = {}
    patientid_to_image_id_fake0 = {}
    for index, row in df1.iterrows():
        image_id = row['影像结果编号']
        patient_id = df2[df2['序列号'] == row['序列编号']]['原始路径'].values[0]
        patient_id = os.path.basename(patient_id).split('_')[0]
        imageid_to_patientid[str(image_id)] = str(patient_id)
        patientid_to_image_id_fake0[str(patient_id)] = str(image_id)    

    csv_file1 = os.path.join(annotation_dir, 'image_anno_TASK_2823.csv')
    csv_file2 = os.path.join(annotation_dir, 'Series_virtual DWI.xls')
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_excel(csv_file2)
    imageid_to_patientid = {}
    patientid_to_image_id_fake1 = {}
    patientid_to_image_id_fake2 = {}
    for index, row in df1.iterrows():
        image_id = row['影像结果编号']
        patient_id = df2[df2['序列号'] == row['序列编号']]['原始路径'].values[0]
        patient_id = os.path.basename(patient_id).split('_')[0]
        imageid_to_patientid[str(image_id)] = str(patient_id)
        task_id = row['用户ID']
        if str(task_id) == '5283':
            patientid_to_image_id_fake1[str(patient_id)] = str(image_id)  
        if str(task_id) == '5284':
            patientid_to_image_id_fake2[str(patient_id)] = str(image_id)  
        
    print(patientid_to_image_id_real)
    print('============')
    print(patientid_to_image_id_fake0)
    print('============')
    print(patientid_to_image_id_fake1)
    print('============')
    print(patientid_to_image_id_fake2)

    print(len(patientid_to_image_id_real))
    print(len(patientid_to_image_id_fake0))
    print(len(patientid_to_image_id_fake1))
    print(len(patientid_to_image_id_fake2))

    anno_result_dir = os.path.join(indir, 'annotation', 'anno_result')
    row_elems = []
    for key, val in patientid_to_image_id_real.items():
        try:
            print('\n====> processing patient {}'.format(key))
            beg = time.time()
            gt_file = os.path.join(anno_result_dir, '{}.mha'.format(val))
            fake0_file = os.path.join(anno_result_dir, '{}.mha'.format(patientid_to_image_id_fake0[key]))
            fake1_file = os.path.join(anno_result_dir, '{}.mha'.format(patientid_to_image_id_fake1[key]))
            fake2_file = os.path.join(anno_result_dir, '{}.mha'.format(patientid_to_image_id_fake2[key]))
            dice0 = cta_stat_calc_dice(gt_file, fake0_file)
            dice1 = cta_stat_calc_dice(gt_file, fake1_file)
            dice2 = cta_stat_calc_dice(gt_file, fake2_file)
            dice0 = np.round(dice0, 3)
            dice1 = np.round(dice1, 3)
            dice2 = np.round(dice2, 3)
            print('patient id {} dice0 ({}/{}):\t{}'.format(key, val, patientid_to_image_id_fake0[key], dice0))
            print('patient id {} dice1 ({}/{}):\t{}'.format(key, val, patientid_to_image_id_fake1[key], dice1))
            print('patient id {} dice2 ({}/{}):\t{}'.format(key, val, patientid_to_image_id_fake2[key], dice2))
            print('====> finished! time elapsed {:.3f}s'.format(time.time() - beg))
            row_elems.append(np.array([key, dice0, dice1, dice2]))
        except:
            print(key)
    
    df = pd.DataFrame(np.array(row_elems), columns=['pid', 'dice0', 'dice1', 'dice2'])
    df.to_csv(os.path.join(indir, 'annotation', 'stat_dicex.csv'))

    # def calc_dice(gt_file, pred_file):
    #     gt_img = sitk.ReadImage(gt_file)
    #     pred_img = sitk.ReadImage(pred_file)
    #     gt_arr = sitk.GetArrayFromImage(gt_img)
    #     pred_arr = stik.GetArrayFromImage(pred_img)
    #     intersect = gt_arr*pred_arr
    #     denominator = (gt_arr * gt_arr).sum(-1) + (pred_arr * pred_arr).sum(-1)



    

# 将6院标注数据划分开
def split_cta2dwi_anno_data(indir):
    '''
    debug cmd: split_cta2dwi_anno_data('../data/gan/hospital_6')
    '''
    annotation_dir = os.path.join(indir, 'annotation')
    annotation_dwi_dir = os.path.join(indir, 'annotation', 'fake_dwi_liaren')
    csv_file1 = os.path.join(annotation_dir, 'image_anno_TASK_2823.csv')
    df1 = pd.read_csv(csv_file1)
    imageid_to_taskid = {}
    for index, row in df1.iterrows():
        image_id = row['影像结果编号']
        task_id = row['用户ID']
        imageid_to_taskid[str(image_id)] = str(task_id)
    annotation_dwi_dir2 = os.path.join(indir, 'annotation', 'fake_dwi_liaren_split')
    for key,val in imageid_to_taskid.items():
        tmp_dir = os.path.join(annotation_dwi_dir2, val)
        os.makedirs(tmp_dir, exist_ok=True)
        src_file = os.path.join(annotation_dwi_dir, '{}.mha'.format(key))
        dst_file = os.path.join(tmp_dir, '{}.mha'.format(key))
        shutil.copyfile(src_file, dst_file)
        print('copy from {} to {}'.format(src_file, dst_file))
        


def test_fire(t_int, t_bool, t_str):
    print(t_int+1)
    print(t_bool)
    if t_bool is True:
        print('1')
    print(True)


if __name__ =='__main__':
    # fire.Fire()
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
    # ncct_generate_cerebral_parenchyma_middle_layer_onecase('../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct/406862_first_BS_brain.nii.gz', None)
    # cta_extract_infos_from_xlsx('../data/gan/hospital_6/2 排除约40人后新增32人基于原六院CTA2DWI.xlsx')
    ## 调试：文件下有大量的CTA和DWI序列，将这些散乱的数据一一配对
    # cta_extract_series_to_patient('../data/gan/hospital_6/ori', '../data/gan/hospital_6/0.raw_dcm', '../data/gan/hospital_6/2 排除约40人后新增32人基于原六院CTA2DWI.xlsx')
    # rapid_extract_sumary_info('../data/gan/hospital_4/0.raw_dcm/114093/RAPID/1.3.6.1.4.1.39822.1.3.8323328.7953.1535266088.657316', '../data/gan/hospital_4/0.raw_dcm/114093/ADC/1.3.12.2.1107.5.2.30.26961.2018082614324377426400913.0.0.0', '114093', '../data/gan/hospital_4/tmp')
    # test_rapid_extract_sumary_info('../data/gan/hospital_4/0.raw_dcm', '../data/gan/hospital_4/1.rapid')
    # extract_cerebral_parenchyma_onecase('../data/gan/hospital_6/experiment_registration2/4 Patient_nii_unity/4495700_first_BS_NCCT.nii.gz', '../data/gan/hospital_6/experiment_registration2/tmp')
    # ncct_set_origal_point_single('../data/gan/hospital_6/experiment_registration2/1.nii_file/3833955_first_BS_NCCT.nii.gz', '../data/gan/hospital_6/experiment_registration2/2.nii_file_ori/3833955_first_BS_NCCT.nii.gz')
    # cta_split_train_test_according_to_xlsx('../data/gan/hospital_6/CTA ASPECT 总表 V11.xlsx', '../data/gan/hospital_6/experiment_registration2/8.2.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt', '../data/gan/hospital_6/experiment_registration2/8.2.out/config/mask_ncct_to_dwi_bxxx_test_config_file.txt')
    # cta_split_train_test_according_to_xlsx('../data/gan/hospital_6/CTA ASPECT 总表 V11.xlsx', '../data/gan/hospital_6_crop/experiment_registration2/8.2.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt', '../data/gan/hospital_6_crop/experiment_registration2/8.2.out/config/mask_ncct_to_dwi_bxxx_test_config_file.txt')
    # rapid_check_rapid_mrp_both_exist('../data/gan/hospital_4/0.raw_dcm')
    # rapid_extract_sumary_info_multiprocess('../data/gan/hospital_4_2/0.raw_dcm', '../data/gan/hospital_4_2/1.rapid')
    ## 调试将DWI的b0和b1000数据区分开
    # cta_extract_dwi_from_raw_dwi_single('../data/gan/hospital_6/0.raw_dcm/3926192/DWI/1.3.46.670589.11.17277.5.0.5008.2017041707551310621/not_dwi', None)
    # rapid_stat_dwi_positive_count_according_to_config('../data/gan/hospital_4_2/1.rapid/config.txt')
    # ncct_split_train_test_according_to_rapid_result('../data/gan/hospital_4_2/experiment_registration2/8.8.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt', '../data/gan/hospital_4_2/experiment_registration2/8.8.out/config/mask_ncct_to_dwi_bxxx_test_config_file.txt', '../data/gan/hospital_4_2/experiment_registration3/1.rapid/config.txt')
    # ncct_generate_cerebral_parenchyma_middle_layer_only_onecase('../data/gan/hospital_4_2/experiment_registration2/5 dwi_rigid_align_ncct/486499_first_BS_brain.nii.gz', None)
    cta_stat_lesion_volume_size('../data/gan/hospital_6')
    # split_cta2dwi_anno_data('../data/gan/hospital_6')
    # cta_stat_lesion_volume_dice('../data/gan/hospital_6')
    # cta_stat_calc_dice('../data/gan/hospital_6/annotation/anno_result/2313573.mha', '../data/gan/hospital_6/annotation/anno_result/2396943.mha')