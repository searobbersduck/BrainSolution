import os
import sys 

from glob import glob
from tqdm import tqdm
import time
import pandas as pd
import json
import numpy as np
import fire
import pydicom

import fire

import SimpleITK as sitk
abs_dir = os.getcwd()
work_dir = os.path.abspath(os.path.join(abs_dir,os.path.pardir))
sys.path.append(work_dir)
work_dir = os.path.abspath(os.path.join(abs_dir,os.path.pardir,os.path.pardir))
sys.path.append(work_dir)


from cerebral_parenchyma.train.train import inference, extract_region_by_mask, extract_region_by_mask1

def extract_cerebral_parenchyma(indir, outdir, inpattern):
    os.makedirs(outdir, exist_ok=True)
    for series_uid in tqdm(os.listdir(indir)):
        series_path = os.path.join(indir, series_uid)
        if not os.path.isdir(series_path):
            continue
        sitk_mask = inference(series_path, '../../cerebral_parenchyma/train/model/extract_cerebral_parenchyma/extract_cerebral_parenchyma_0056_best_loss_0.011.pth', None, is_dcm=True)
        
        series_reader = sitk.ImageSeriesReader()
        dicomfilenames = series_reader.GetGDCMSeriesFileNames(series_path)
        series_reader.SetFileNames(dicomfilenames)

        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()
        
        image = series_reader.Execute()

        masked_image = extract_region_by_mask1(image, sitk_mask)

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

        filtered_image = masked_image
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


        out_dcm_path = os.path.join(outdir, series_uid)
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



class AneurysmLocation:
    def __init__(self, p1, p2):
        '''
        p1 wise: z, y, x
        '''
        self.p1 = p1
        self.p2 = p2
        self.diameter = np.linalg.norm(self.p2-self.p1)
        self.center = (p1+p2)/2
        self.boundary_min = self.center-self.diameter/2
        self.boundary_max = self.center+self.diameter/2
        self.boundary_min_int = np.array(np.floor(self.boundary_min), dtype=np.int)
        self.boundary_max_int = np.array(np.ceil(self.boundary_max), dtype=np.int)
        self.center_int = np.array(self.center, dtype=np.int)
        

    def is_in(self, region):
        '''
        region: min_z, max_z, min_y, max_y, min_x, max_x
        '''
        # if np.all(self.boundary_max < np.array([region[0], region[2], region[4]])) or np.all(self.boundary_min > np.array([region[1], region[3], region[5]])):
        #     return 0
        if np.all(self.boundary_max <= np.array([region[1], region[3], region[5]])) and np.all(self.boundary_min >= np.array([region[0], region[2], region[4]])):
            return 1
        elif np.all(self.center <= np.array([region[1], region[3], region[5]])) and np.all(self.center >= np.array([region[0], region[2], region[4]])):
            return 2
        else:
            return 0

    def get_bias(self, region):
        return self.center - np.array([region[0], region[2], region[4]])

    # generate mask points according to given bounding box(center and radius) 
    # point:(z,y, x)
    # points: [[z,y,x], [z,y,x], ...]
    def generate_mask_points(self):
        points = []
        for iz in range(self.boundary_min_int[0], self.boundary_max_int[0]):
            for iy in range(self.boundary_min_int[1], self.boundary_max_int[1]):
                for ix in range(self.boundary_min_int[2], self.boundary_max_int[2]):
                    points.append([iz, iy, ix])
        return points
    



# check if the Instance number and slice location are monotone increasing
def FindZDirection(folderName):
    dcm_files = glob(os.path.join(folderName, '*.dcm'))
    metadata = pydicom.dcmread(dcm_files[0])
    slice_loc1  = metadata.SliceLocation
    xuhao1 = metadata.InstanceNumber
    metadata = pydicom.dcmread(dcm_files[1])
    slice_loc2  = metadata.SliceLocation
    xuhao2 = metadata.InstanceNumber
    p = (xuhao1-xuhao2)*(slice_loc1-slice_loc2)
    z_flip = False
    if p < 0:
        z_flip = True
    return z_flip

def generate_block_pairs(in_arr, series_uid, block_size=[128,128,128], stride=[64, 64, 64], target_locations=[]):
    '''
    block_size = [block_size_z, block_size_y, block_size_x]
    stride = [stride_z, stride_y, stride_x]
    将数据分成block_size大小的若干块，按照stride大小进行步进，并且根据肿瘤的位置，给出被切出的块中是否含有肿瘤；
    输出标签：0：无肿瘤， 1：有肿瘤的部分， 2：包含整个肿瘤
    '''
    [z, y, x] = in_arr.shape
    blocks = []
    blocks_label_list = []
    blocks_name_list = []
    index = 0
    for iz in range(0, z, stride[0]):
        for iy in range(0, y, stride[1]):
            for ix in range(0, x, stride[2]):
                z_min = iz
                z_max = min(iz+block_size[2], z)
                y_min = iy
                y_max = min(iy+block_size[1], y)
                x_min = ix
                x_max = min(ix+block_size[0], x)
                block = np.zeros(block_size)
                block[:(z_max-z_min), :(y_max-y_min), :(x_max-x_min)] = in_arr[z_min:z_max, y_min:y_max, x_min:x_max]
                is_in = -1
                for loc in target_locations:
                    if is_in != 1:
                        is_in = loc.is_in([z_min, z_max, y_min, y_max, x_min, x_max])
                        bias_coord = loc.get_bias([z_min, z_max, y_min, y_max, x_min, x_max])
                block_name = '{}_{:.1f}_{:.1f}_{:.1f}_{}.npy'.format(series_uid, bias_coord[2], bias_coord[1], bias_coord[0], index)
                index += 1
                blocks_label_list.append(is_in)
                blocks_name_list.append(block_name)
                blocks.append(block)
    return blocks, blocks_label_list, blocks_name_list

def generate_block_pairs_WD_fir(anns, series_uid, root_dir, out_dir):
    ann_infos = anns
    series_uid = series_uid
    series_path = os.path.join(root_dir, series_uid)
    if not os.path.isdir(series_path):
        return None, None
    series_reader = sitk.ImageSeriesReader()
    dicomfilenames = series_reader.GetGDCMSeriesFileNames(series_path)
    series_reader.SetFileNames(dicomfilenames)

    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    
    is_filp = FindZDirection(series_path)
    print('is flip:\t{}'.format(is_filp))

    image = series_reader.Execute()

    in_arr = sitk.GetArrayFromImage(image)
    target_locs = []
    for ann_info in ann_infos:
        p1 = np.array([float(ann_info['point1']['z']), float(ann_info['point1']['y']), float(ann_info['point1']['x'])])
        p2 = np.array([float(ann_info['point2']['z']), float(ann_info['point2']['y']), float(ann_info['point2']['x'])])

        if is_filp:
            z_max = len(dicomfilenames)
            p1 = np.array([float(ann_info['point1']['z']), float(ann_info['point1']['y']), z_max-float(ann_info['point1']['x'])])
            p2 = np.array([float(ann_info['point2']['z']), float(ann_info['point2']['y']), z_max-float(ann_info['point2']['x'])])
        else:
            p1 = np.array([float(ann_info['point1']['z']), float(ann_info['point1']['y']), float(ann_info['point1']['x'])-1])
            p2 = np.array([float(ann_info['point2']['z']), float(ann_info['point2']['y']), float(ann_info['point2']['x'])-1])

        target_loc = AneurysmLocation(p1, p2)
        target_locs.append(target_loc)


    blocks, blocks_label_list, blocks_name_list = generate_block_pairs(in_arr, series_uid, target_locations=target_locs)

    out_block_name_list = []

    for i, block in enumerate(blocks):
        # block_name = os.path.join(out_dir, '{}_{}.npy'.format(series_uid, i))
        block_name = os.path.join(out_dir, blocks_name_list[i])
        print('====> save to {}'.format(block_name))
        np.save(block_name, block)
        out_block_name_list.append(os.path.basename(block_name))

    return out_block_name_list, blocks_label_list

def generate_block_pairs_WD_fir_singlefolder(anns, series_uids, root_dir, out_dir):
    names = []
    labels = []
    for i in tqdm(range(len(anns))):
        name, label = generate_block_pairs_WD_fir(anns[i], series_uids[i], root_dir, out_dir)
        if name is not None:
            names += name
            labels += label
    return names, labels


def test_generate_block_pairs():
    csv_file = '../data/source_img/csvs/WD_fir.csv'
    root_dir = '../data/source_img/img_WD_fir'
    out_dir = '../data/block_pairs/WD_fir'
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    print(df.head())
    print('finish test_generate_block_pairs!')
    all_names = []
    all_labels = []
    for index, row in df.iterrows():
        ann_info = json.loads(row['影像结果'])
        series_uid = row['序列编号']
        series_path = os.path.join(root_dir, series_uid)
        if not os.path.isdir(series_path):
            continue
        series_reader = sitk.ImageSeriesReader()
        dicomfilenames = series_reader.GetGDCMSeriesFileNames(series_path)
        series_reader.SetFileNames(dicomfilenames)

        series_reader.MetaDataDictionaryArrayUpdateOn()
        series_reader.LoadPrivateTagsOn()
        
        image = series_reader.Execute()

        in_arr = sitk.GetArrayFromImage(image)
        p1 = np.array([float(ann_info['point1']['z']), float(ann_info['point1']['y']), float(ann_info['point1']['x'])])
        p2 = np.array([float(ann_info['point2']['z']), float(ann_info['point2']['y']), float(ann_info['point2']['x'])])
        target_loc = AneurysmLocation(p1, p2)


        blocks, blocks_label_list = generate_block_pairs(in_arr, target_locations=[target_loc])

        block_name_list = []

        for i, block in enumerate(blocks):
            block_name = os.path.join(out_dir, '{}_{}.npy'.format(series_uid, i))
            print('====> save to {}'.format(block_name))
            np.save(block_name, block)
            block_name_list.append(os.path.basename(block_name))

        all_names += block_name_list
        all_labels += blocks_label_list

        # print('hello world')
    with open(os.path.join(out_dir, 'config.txt'), 'w') as f:
        for i in range(len(all_names)):
            f.write('{}\t{}\n'.format(all_names[i], all_labels[i]))
    print('hello world')

def test_generate_block_pairs_singleprocessing():
    csv_file = '../data/source_img/csvs/WD_fir.csv'
    root_dir = '../data/source_img/img_WD_fir'
    out_dir = '../data/block_pairs/WD_fir'
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    print(df.head())
    print('finish test_generate_block_pairs!')
    all_names = []
    all_labels = []
    ann_info_list = []
    series_uid_list = []
    for index, row in df.iterrows():
        ann_info = json.loads(row['影像结果'])
        series_uid = row['序列编号']
        ann_info_list.append(ann_info)
        series_uid_list.append(series_uid)

    names, labels = generate_block_pairs_WD_fir_singlefolder(ann_info_list, series_uid_list, root_dir, out_dir)

        # print('hello world')
    with open(os.path.join(out_dir, 'config.txt'), 'w') as f:
        for i in range(len(all_names)):
            f.write('{}\t{}\n'.format(all_names[i], all_labels[i]))
    print('hello world')


def generate_block_pairs_multiprocessing(root_dir, csv_file, out_dir, train_split_ratio=0.9, process_num=8):    
    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []


    # csv_file = '../data/source_img/csvs/WD_fir.csv'
    # root_dir = '../data/source_img/img_WD_fir'
    # out_dir = '../data/source_img/block_pairs/WD_fir_coord'
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    print(df.head())
    print('finish test_generate_block_pairs!')
    all_names = []
    all_labels = []
    ann_info_list = []
    series_uid_list = []
    series_dict = {}
    for index, row in df.iterrows():
        ann_info = json.loads(row['影像结果'])
        series_uid = row['序列编号']
        ann_info_list.append(ann_info)
        series_uid_list.append(series_uid)
        if series_uid in series_dict:
            series_dict[series_uid] += [ann_info]
        else:
            series_dict[series_uid] = [ann_info]

    ann_info_list = []
    series_uid_list = []
    for key,val in series_dict.items():
        series_uid_list.append(key)
        ann_info_list.append(val)

    train_all_names = []
    train_all_labels = []
    train_ann_info_list = []
    train_series_uid_list = []

    ratio = train_split_ratio
    index = list(range(len(ann_info_list)))
    train_pos = int(ratio*len(ann_info_list))
    np.random.shuffle(index)
    train_index = index[:train_pos]
    val_index = index[train_pos:]

    # train_ann_info_list = ann_info_list[train_index]
    # train_series_uid_list = series_uid_list[train_index]
    train_ann_info_list = [ann_info_list[i] for i in train_index]
    train_series_uid_list = [series_uid_list[i] for i in train_index]

    val_ann_info_list = [ann_info_list[i] for i in val_index]
    val_series_uid_list = [series_uid_list[i] for i in val_index]

    num_per_process = (len(train_ann_info_list) + process_num - 1)//process_num

    train_out_dir = os.path.join(out_dir, 'train')
    os.makedirs(train_out_dir, exist_ok=True)
    for i in range(process_num):
        sub_anns = train_ann_info_list[num_per_process*i:min(num_per_process*(i+1), len(ann_info_list)-1)]
        sub_uids = train_series_uid_list[num_per_process*i:min(num_per_process*(i+1), len(series_uid_list)-1)]
        result = pool.apply_async(generate_block_pairs_WD_fir_singlefolder, args=(sub_anns, sub_uids, root_dir, train_out_dir))
        results.append(result)

    pool.close()
    pool.join()

    for result in results:
        result = result.get()
        train_all_names += result[0]
        train_all_labels += result[1]

        # print('hello world')
    with open(os.path.join(train_out_dir, 'config.txt'), 'w') as f:
        for i in range(len(train_all_names)):
            f.write('{}\t{}\n'.format(train_all_names[i], train_all_labels[i]))

    val_out_dir = os.path.join(out_dir, 'val')
    os.makedirs(val_out_dir, exist_ok=True)
    val_all_names, val_all_labels = generate_block_pairs_WD_fir_singlefolder(val_ann_info_list, val_series_uid_list, root_dir, val_out_dir)

    with open(os.path.join(val_out_dir, 'config.txt'), 'w') as f:
        for i in range(len(val_all_names)):
            f.write('{}\t{}\n'.format(val_all_names[i], val_all_labels[i]))

    print('hello world')

def WD_fir_generate_block_pairs_multiprocessing(process_num=8):
    generate_block_pairs_multiprocessing('../data/source_img/img_WD_fir', 
        '../data/source_img/csvs/WD_fir.csv', '../data/source_img/block_pairs/WD_fir_coord', 
        train_split_ratio=0.9, process_num=8)


def XY_fir_generate_block_pairs_multiprocessing(process_num=8):
    generate_block_pairs_multiprocessing('../data/source_img/img_XY_fir', 
        '../data/source_img/csvs/XY_fir.csv', '../data/source_img/block_pairs/XY_fir_coord', 
        train_split_ratio=0.9, process_num=8)


def XY_sec_generate_block_pairs_multiprocessing(process_num=8):
    generate_block_pairs_multiprocessing('../data/source_img/img_XY_sec', 
        '../data/source_img/csvs/XY_sec.csv', '../data/source_img/block_pairs/XY_sec_coord', 
        train_split_ratio=0.9, process_num=8)
        

def test_generate_block_pairs_multiprocessing(process_num=8):    
    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []


    csv_file = '../data/source_img/csvs/WD_fir.csv'
    root_dir = '../data/source_img/img_WD_fir'
    out_dir = '../data/source_img/block_pairs/WD_fir_coord'
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    print(df.head())
    print('finish test_generate_block_pairs!')
    all_names = []
    all_labels = []
    ann_info_list = []
    series_uid_list = []
    series_dict = {}
    for index, row in df.iterrows():
        ann_info = json.loads(row['影像结果'])
        series_uid = row['序列编号']
        ann_info_list.append(ann_info)
        series_uid_list.append(series_uid)
        if series_uid in series_dict:
            series_dict[series_uid] += [ann_info]
        else:
            series_dict[series_uid] = [ann_info]

    ann_info_list = []
    series_uid_list = []
    for key,val in series_dict.items():
        series_uid_list.append(key)
        ann_info_list.append(val)

    train_all_names = []
    train_all_labels = []
    train_ann_info_list = []
    train_series_uid_list = []

    ratio = 0.9
    index = list(range(len(ann_info_list)))
    train_pos = int(ratio*len(ann_info_list))
    np.random.shuffle(index)
    train_index = index[:train_pos]
    val_index = index[train_pos:]

    # train_ann_info_list = ann_info_list[train_index]
    # train_series_uid_list = series_uid_list[train_index]
    train_ann_info_list = [ann_info_list[i] for i in train_index]
    train_series_uid_list = [series_uid_list[i] for i in train_index]

    val_ann_info_list = [ann_info_list[i] for i in val_index]
    val_series_uid_list = [series_uid_list[i] for i in val_index]

    num_per_process = (len(train_ann_info_list) + process_num - 1)//process_num

    train_out_dir = os.path.join(out_dir, 'train')
    os.makedirs(train_out_dir, exist_ok=True)
    for i in range(process_num):
        sub_anns = train_ann_info_list[num_per_process*i:min(num_per_process*(i+1), len(ann_info_list)-1)]
        sub_uids = train_series_uid_list[num_per_process*i:min(num_per_process*(i+1), len(series_uid_list)-1)]
        result = pool.apply_async(generate_block_pairs_WD_fir_singlefolder, args=(sub_anns, sub_uids, root_dir, train_out_dir))
        results.append(result)

    pool.close()
    pool.join()

    for result in results:
        result = result.get()
        train_all_names += result[0]
        train_all_labels += result[1]

        # print('hello world')
    with open(os.path.join(train_out_dir, 'config.txt'), 'w') as f:
        for i in range(len(train_all_names)):
            f.write('{}\t{}\n'.format(train_all_names[i], train_all_labels[i]))

    val_out_dir = os.path.join(out_dir, 'val')
    os.makedirs(val_out_dir, exist_ok=True)
    val_all_names, val_all_labels = generate_block_pairs_WD_fir_singlefolder(val_ann_info_list, val_series_uid_list, root_dir, val_out_dir)

    with open(os.path.join(val_out_dir, 'config.txt'), 'w') as f:
        for i in range(len(val_all_names)):
            f.write('{}\t{}\n'.format(val_all_names[i], val_all_labels[i]))

    print('hello world')


def extract_block_with_aneurysm(root_dir, config_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with open(config_file) as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            ss = line.split('\t')
            if len(ss) != 2:
                continue
            if int(ss[1]) != 1:
                continue
            outfile = os.path.join(out_dir, ss[0].replace('.npy', '.nii.gz'))
            arr = np.load(os.path.join(root_dir, ss[0]))
            image = sitk.GetImageFromArray(arr)
            sitk.WriteImage(image, outfile)

def convert_image_format(infile, inpattern, outpattern):
    '''
    infile: 
    inpattern: '.nii.gz'
    outpattern: '.mhd'

    python aneurysm_utils.py convert_image_format '../data/source_img/block_pairs/WD_fir_coord/train_visualize/1.2.840.113619.2.334.3.2831179063.409.1509096762.87_115.3_109.1_19.0_402.nii.gz' '.nii.gz' '.mhd'
    python aneurysm_utils.py convert_image_format '../../gan/data/brain/gan/ncct2dwi/experiment_registration2/2.nii_file_ori/137611_first_BS_NCCT.nii.gz' '.nii.gz' '.mhd'
    '''
    image = sitk.ReadImage(infile)
    # sitk.WriteImage(image, infile.replace(inpattern, outpattern))

    # series_reader = sitk.ImageSeriesReader()
    # dicomfilenames = series_reader.GetGDCMSeriesFileNames(infile)
    # series_reader.SetFileNames(dicomfilenames)

    # series_reader.MetaDataDictionaryArrayUpdateOn()
    # series_reader.LoadPrivateTagsOn()
    
    # image = series_reader.Execute()

    sitk.WriteImage(image, 'test.mhd')

def test_extract_block_with_aneurysm():
    root_dir = '../data/source_img/block_pairs/WD_fir_coord/train'
    out_dir = '../data/source_img/block_pairs/WD_fir_coord/train_visualize'
    config_file = '../data/source_img/block_pairs/WD_fir_coord/train/config.txt'
    extract_block_with_aneurysm(root_dir, config_file, out_dir)








# for segmentation

# 将数据生成nii.gz格式和对应的nii.gz格式的mask
def generate_aneurysm_mask_one_case(anns, series_uid, root_dir, out_dir):
    ann_infos = anns
    series_uid = series_uid
    series_path = os.path.join(root_dir, series_uid)
    if not os.path.isdir(series_path):
        return None, None
    series_reader = sitk.ImageSeriesReader()
    dicomfilenames = series_reader.GetGDCMSeriesFileNames(series_path)
    series_reader.SetFileNames(dicomfilenames)

    series_reader.MetaDataDictionaryArrayUpdateOn()
    series_reader.LoadPrivateTagsOn()
    
    is_filp = FindZDirection(series_path)
    print('is flip:\t{}'.format(is_filp))

    image = series_reader.Execute()

    in_arr = sitk.GetArrayFromImage(image)

    target_locs = []
    for ann_info in ann_infos:
        p1 = np.array([float(ann_info['point1']['z']), float(ann_info['point1']['y']), float(ann_info['point1']['x'])])
        p2 = np.array([float(ann_info['point2']['z']), float(ann_info['point2']['y']), float(ann_info['point2']['x'])])

        if is_filp:
            z_max = len(dicomfilenames)
            p1 = np.array([float(ann_info['point1']['z']), float(ann_info['point1']['y']), z_max-float(ann_info['point1']['x'])])
            p2 = np.array([float(ann_info['point2']['z']), float(ann_info['point2']['y']), z_max-float(ann_info['point2']['x'])])
        else:
            p1 = np.array([float(ann_info['point1']['z']), float(ann_info['point1']['y']), float(ann_info['point1']['x'])-1])
            p2 = np.array([float(ann_info['point2']['z']), float(ann_info['point2']['y']), float(ann_info['point2']['x'])-1])

        target_loc = AneurysmLocation(p1, p2)
        target_locs.append(target_loc)

    mask_arr = np.zeros(in_arr.shape, dtype=np.uint8)
    centers = []
    for target_loc in target_locs:
        centers.append(target_loc.center_int)
        points = target_loc.generate_mask_points()
        for point in points:
            mask_arr[point[0], point[1], point[2]] = 1

    out_image_file = os.path.join(out_dir, '{}_image.nii.gz'.format(series_uid))
    out_mask_file = os.path.join(out_dir, '{}_mask.nii.gz'.format(series_uid))

    sitk.WriteImage(image, out_image_file)
    mask = sitk.GetImageFromArray(mask_arr)
    sitk.WriteImage(mask, out_mask_file)
    # 将centers转换为str, 格式为：'z1\ty1\tx1\tz2...'
    center_str = ''
    for center in centers:
        center = list(center)
        for v in center:
            center_str += str(v) + '\t'
    center_str = center_str[:-1]

    return series_uid, center_str


def generate_aneurysm_mask_singlefolder(anns, series_uids, root_dir, out_dir):
    centers = []
    names = []
    for i in tqdm(range(len(anns))):
        series_uid, center = generate_aneurysm_mask_one_case(anns[i], series_uids[i], root_dir, out_dir)
        names.append(series_uid)
        centers.append(center)
    return names, centers


def generate_aneurysm_mask_multiprocessing(root_dir, csv_file, out_dir, train_split_ratio=0.9, process_num=8):    
    import multiprocessing
    from multiprocessing import Process
    multiprocessing.freeze_support()

    pool = multiprocessing.Pool()
    results = []


    # csv_file = '../data/source_img/csvs/WD_fir.csv'
    # root_dir = '../data/source_img/img_WD_fir'
    # out_dir = '../data/source_img/block_pairs/WD_fir_coord'
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    print(df.head())
    print('finish test_generate_block_pairs!')
    all_names = []
    all_labels = []
    ann_info_list = []
    series_uid_list = []
    series_dict = {}
    for index, row in df.iterrows():
        ann_info = json.loads(row['影像结果'])
        series_uid = row['序列编号']
        ann_info_list.append(ann_info)
        series_uid_list.append(series_uid)
        if series_uid in series_dict:
            series_dict[series_uid] += [ann_info]
        else:
            series_dict[series_uid] = [ann_info]

    ann_info_list = []
    series_uid_list = []
    for key,val in series_dict.items():
        series_uid_list.append(key)
        ann_info_list.append(val)

    train_all_names = []
    train_all_labels = []
    train_ann_info_list = []
    train_series_uid_list = []

    ratio = train_split_ratio
    index = list(range(len(ann_info_list)))
    train_pos = int(ratio*len(ann_info_list))
    np.random.shuffle(index)
    train_index = index[:train_pos]
    val_index = index[train_pos:]

    # train_ann_info_list = ann_info_list[train_index]
    # train_series_uid_list = series_uid_list[train_index]
    train_ann_info_list = [ann_info_list[i] for i in train_index]
    train_series_uid_list = [series_uid_list[i] for i in train_index]

    val_ann_info_list = [ann_info_list[i] for i in val_index]
    val_series_uid_list = [series_uid_list[i] for i in val_index]

    num_per_process = (len(train_ann_info_list) + process_num - 1)//process_num

    train_out_dir = os.path.join(out_dir, 'train')
    os.makedirs(train_out_dir, exist_ok=True)
    for i in range(process_num):
        sub_anns = train_ann_info_list[num_per_process*i:min(num_per_process*(i+1), len(ann_info_list)-1)]
        sub_uids = train_series_uid_list[num_per_process*i:min(num_per_process*(i+1), len(series_uid_list)-1)]
        result = pool.apply_async(generate_aneurysm_mask_singlefolder, args=(sub_anns, sub_uids, root_dir, train_out_dir))
        results.append(result)

    pool.close()
    pool.join()

    for result in results:
        result = result.get()
        train_all_names += result[0]
        train_all_labels += result[1]

        # print('hello world')
    with open(os.path.join(train_out_dir, 'config.txt'), 'w') as f:
        for i in range(len(train_all_names)):
            f.write('{}\t{}\n'.format(train_all_names[i], train_all_labels[i]))

    val_out_dir = os.path.join(out_dir, 'val')
    os.makedirs(val_out_dir, exist_ok=True)
    val_all_names, val_all_labels = generate_aneurysm_mask_singlefolder(val_ann_info_list, val_series_uid_list, root_dir, val_out_dir)

    with open(os.path.join(val_out_dir, 'config.txt'), 'w') as f:
        for i in range(len(val_all_names)):
            f.write('{}\t{}\n'.format(val_all_names[i], val_all_labels[i]))

    print('hello world')

def WD_fir_generate_aneurysm_mask_multiprocessing(process_num=8):
    generate_aneurysm_mask_multiprocessing('../data/source_img/img_WD_fir', 
        '../data/source_img/csvs/WD_fir.csv', '../data/source_img/seg/WD_fir', 
        train_split_ratio=0.9, process_num=process_num)


def test_generate_aneurysm_mask_singlefolder():
    csv_file = '../data/source_img/csvs/WD_fir.csv'
    root_dir = '../data/source_img/img_WD_fir'
    out_dir = '../data/source_img/seg/WD_fir'
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(csv_file)
    print(df.head())
    print('finish test_generate_block_pairs!')
    all_names = []
    all_labels = []
    ann_info_list = []
    series_uid_list = []
    for index, row in df.iterrows():
        ann_info = json.loads(row['影像结果'])
        series_uid = row['序列编号']
        ann_info_list.append([ann_info])
        series_uid_list.append(series_uid)

    names, labels = generate_aneurysm_mask_singlefolder(ann_info_list, series_uid_list, root_dir, out_dir)

        # print('hello world')
    with open(os.path.join(out_dir, 'config.txt'), 'w') as f:
        for i in range(len(all_names)):
            f.write('{}\t{}\n'.format(all_names[i], all_labels[i]))
    print('hello world')





if __name__ == '__main__':
    fire.Fire()
    # extract_cerebral_parenchyma('../data/source_img/img_WD_fir', '../data/source_img/test', '*.nii.gz')
    # test_generate_block_pairs()
    # test_extract_block_with_aneurysm()
    # test_generate_block_pairs_singleprocessing()
    # convert_image_format('../data/source_img/block_pairs/WD_fir_coord/train_visualize/1.2.840.113619.2.334.3.2831179063.409.1509096762.87_115.3_109.1_19.0_402.nii.gz', '.nii.gz', '.mhd')
    # test_generate_aneurysm_mask_singlefolder()