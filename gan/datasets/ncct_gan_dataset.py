import os
import sys
from glob import glob
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
import nibabel
from scipy import ndimage

from tqdm import tqdm


global_ncct_error_list = ['250238', '317892', '462086', '456640', '475170', '372829', '462630', '417357', '458192', '429884', '456831', 
    '5016897', '3466686', '3772244', '3839904', '4728962']



class NCCT_GAN_DS(Dataset):
    def __init__(self, root_dir, config_file, phase, crop_size, scale_size, debug=False):
        super().__init__()
        self.src_list = []
        self.dst_list = []
        self.mask_list = []
        self.root_dir = root_dir
        self.phase = phase
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.debug = debug
        # self.error_indexs = ['475170', '372829', '463311']
        self.error_indexs = global_ncct_error_list
        with open(config_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                if len(ss) < 8:
                    continue
                src_file = os.path.join(root_dir, ss[0])
                dst_file = os.path.join(root_dir, ss[1])
                if not os.path.isfile(src_file):
                    continue
                if not os.path.isfile(dst_file):
                    continue
                # 去除事先查看过的有问题的数据
                index = os.path.basename(ss[0]).split('_')[0]
                if index in self.error_indexs:
                    continue
                d = int(ss[3])-int(ss[2])
                h = int(ss[5])-int(ss[4])
                w = int(ss[7])-int(ss[6])
                if d < self.crop_size[0] or h < self.crop_size[1] or w < self.crop_size[2]:
                    # continue
                    pass
                self.src_list.append(src_file)
                self.dst_list.append(dst_file)
            print('====> data count is:\t{}'.format(len(self.src_list)))
    
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
        return len(self.src_list)

    def __getitem__(self, idx):
        if self.phase == 'train':
            src_file = self.src_list[idx]
            dst_file = self.dst_list[idx]
            src_img = sitk.ReadImage(src_file)
            src_data = sitk.GetArrayFromImage(src_img)
            dst_img = sitk.ReadImage(dst_file)
            dst_data = sitk.GetArrayFromImage(dst_img)
            if np.random.rand() < 0.8:
                cropped_src, cropped_dst = self.__random_crop_data(src_data, dst_data, self.crop_size)
            else:
                cropped_src, cropped_dst = self.__center_crop_data(src_data, dst_data, self.crop_size)

            if self.debug:
                mid_dir = os.path.join(self.root_dir, 'tmp')
                os.makedirs(mid_dir, exist_ok=True)
                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(mid_dir, 'src_index_{}.nii.gz'.format(idx)))
                writer.Execute(sitk.GetImageFromArray(cropped_src))

                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(mid_dir, 'dst_index_{}.nii.gz'.format(idx)))
                writer.Execute(sitk.GetImageFromArray(cropped_dst))

            cropped_src = torch.from_numpy(cropped_src).float()
            cropped_src = torch.unsqueeze(cropped_src, axis=0)
            cropped_dst = torch.from_numpy(cropped_dst).float()
            cropped_dst = torch.unsqueeze(cropped_dst, axis=0)
            return cropped_src, cropped_dst, cropped_src, os.path.basename(src_file), os.path.basename(dst_file)


class CroppedBoundary():
    def __init__(self, boundary_d_min, boundary_d_max, boundary_h_min, boundary_h_max, boundary_w_min, boundary_w_max):
        self.boundary_d_min = boundary_d_min
        self.boundary_d_max = boundary_d_max
        self.boundary_h_min = boundary_h_min
        self.boundary_h_max = boundary_h_max
        self.boundary_w_min = boundary_w_min
        self.boundary_w_max = boundary_w_max




class NCCT_GAN_MASK_DS(Dataset):
    def __init__(self, root_dir, config_file, phase, crop_size, scale_size, debug=False):
        super().__init__()
        self.src_list = []
        self.dst_list = []
        self.mask_list = []
        self.boundary_list = []
        self.root_dir = root_dir
        self.phase = phase
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.debug = debug
        self.error_indexs = global_ncct_error_list
        # self.error_indexs = ['475170', '372829', '463311']
        # self.error_indexs = ['250238', '317892', '462086', '456640', '475170', '372829', '462630', '417357', '458192', '429884', '456831'] # 462630有问题需要验证，选取图像出的问题
        # 配准有问题
        # hospital_4_2: '417357', '458192'， '429884':中间有断层？, '456831', '175928'疑似, 
        # hospital_4: '462630'
        with open(config_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                if len(ss) < 9:
                    continue
                src_file = os.path.join(root_dir, ss[0])
                dst_file = os.path.join(root_dir, ss[1])
                mask_file = os.path.join(root_dir, ss[8])
                if not os.path.isfile(src_file):
                    print('{} is not file!'.format(src_file))
                    continue
                if not os.path.isfile(dst_file):
                    print('{} is not file!'.format(dst_file))
                    continue
                # 去除事先查看过的有问题的数据
                index = os.path.basename(ss[0]).split('_')[0]
                if index in self.error_indexs:
                    continue
                d = int(ss[3])-int(ss[2])
                h = int(ss[5])-int(ss[4])
                w = int(ss[7])-int(ss[6])
                cropped_boundary = CroppedBoundary(int(ss[2]), int(ss[3]), int(ss[4]), int(ss[5]), int(ss[6]), int(ss[7]))
                if d < self.crop_size[0] or h < self.crop_size[1] or w < self.crop_size[2]:
                    # 有些NCCT图像，只采集了几层，需要去掉
                    continue
                    # pass
                self.src_list.append(src_file)
                self.dst_list.append(dst_file)
                self.mask_list.append(mask_file)
                self.boundary_list.append(cropped_boundary)
            print('====> data count is:\t{}'.format(len(self.src_list)))
    
    def __random_crop_data(self, ct_data, dwi_data, mask_data, size, cropped_boundary):
        # [img_d, img_h, img_w] = dwi_data.shape
        padding = 1
        [img_d, img_h, img_w] = [cropped_boundary.boundary_d_max+padding, cropped_boundary.boundary_h_max+padding, cropped_boundary.boundary_w_max+padding]
        [input_d, input_h, input_w] = size
        # assert np.all(np.less_equal(size, dwi_data.shape))
        z_min_upper = img_d - input_d
        y_min_upper = img_h - input_h
        x_min_upper = img_w - input_w

        # Z_min = np.random.randint(0, z_min_upper)
        # Y_min = np.random.randint(0, y_min_upper)
        # X_min = np.random.randint(0, x_min_upper)


        # print('cropped_boundary.boundary_d_min-padding:\t', cropped_boundary.boundary_d_min-padding)
        # print('z_min_upper\t', z_min_upper)
        # print('cropped_boundary.boundary_h_min-padding\t', cropped_boundary.boundary_h_min-padding)
        # print('y_min_upper\t', y_min_upper)
        # print('cropped_boundary.boundary_w_min-padding\t', cropped_boundary.boundary_w_min-padding)
        # print('x_min_upper\t', x_min_upper)
        Z_min = np.random.randint(cropped_boundary.boundary_d_min, z_min_upper)
        Y_min = np.random.randint(cropped_boundary.boundary_h_min, y_min_upper)
        X_min = np.random.randint(cropped_boundary.boundary_w_min, x_min_upper)

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

        # print('__random_crop_data\tZ_min:{}\tY_min:{}\tX_min:{}'.format(Z_min, Y_min, X_min))

        return ct_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], dwi_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], mask_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __center_crop_data(self, ct_data, dwi_data, mask_data, size, cropped_boundary):
        # [img_d, img_h, img_w] = dwi_data.shape
        padding = 1
        [img_d, img_h, img_w] = [cropped_boundary.boundary_d_max+padding, cropped_boundary.boundary_h_max+padding, cropped_boundary.boundary_w_max+padding]
        center_d =  (cropped_boundary.boundary_d_max+padding + cropped_boundary.boundary_d_min) // 2
        center_h =  (cropped_boundary.boundary_h_max+padding + cropped_boundary.boundary_h_min) // 2
        center_w =  (cropped_boundary.boundary_w_max+padding + cropped_boundary.boundary_w_min) // 2
        [input_d, input_h, input_w] = size
        # assert np.all(np.less_equal(size, dwi_data.shape))
        # Z_min = img_d//2-input_d//2
        # Y_min = img_h//2-input_h//2
        # X_min = img_w//2-input_w//2

        Z_min = center_d-input_d//2
        Y_min = center_h-input_h//2
        X_min = center_w-input_w//2

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

        # print('__center_crop_data\tZ_min:{}\tY_min:{}\tX_min:{}'.format(Z_min, Y_min, X_min))
        return ct_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], dwi_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], mask_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):
        if self.phase == 'train':
            src_file = self.src_list[idx]
            # print(src_file)
            dst_file = self.dst_list[idx]
            mask_file = self.mask_list[idx]
            src_img = sitk.ReadImage(src_file)
            src_data = sitk.GetArrayFromImage(src_img)
            dst_img = sitk.ReadImage(dst_file)
            dst_data = sitk.GetArrayFromImage(dst_img)
            mask_img = sitk.ReadImage(mask_file)
            mask_data = sitk.GetArrayFromImage(mask_img)
            
            if np.random.rand() < 0.9:
                cropped_src, cropped_dst, cropped_mask = self.__random_crop_data(src_data, dst_data, mask_data, self.crop_size, self.boundary_list[idx])
            else:
                cropped_src, cropped_dst, cropped_mask = self.__center_crop_data(src_data, dst_data, mask_data, self.crop_size, self.boundary_list[idx])

            if self.debug:
                mid_dir = os.path.join(self.root_dir, 'tmp')
                os.makedirs(mid_dir, exist_ok=True)
                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(mid_dir, 'src_index_{}.nii.gz'.format(idx)))
                writer.Execute(sitk.GetImageFromArray(cropped_src))

                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(mid_dir, 'dst_index_{}.nii.gz'.format(idx)))
                writer.Execute(sitk.GetImageFromArray(cropped_dst))

                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(mid_dir, 'mask_index_{}.nii.gz'.format(idx)))
                writer.Execute(sitk.GetImageFromArray(cropped_mask))

            cropped_src = torch.from_numpy(cropped_src).float()
            cropped_src = torch.unsqueeze(cropped_src, axis=0)
            cropped_dst = torch.from_numpy(cropped_dst).float()
            cropped_dst = torch.unsqueeze(cropped_dst, axis=0)
            cropped_mask = torch.from_numpy(cropped_mask).float()
            cropped_mask = torch.unsqueeze(cropped_mask, axis=0)
            # print('cropped_src:\t{}'.format(cropped_src.shape))
            # print('cropped_dst:\t{}'.format(cropped_dst.shape))
            # print('cropped_mask:\t{}'.format(cropped_mask.shape))
            return cropped_src, cropped_dst, cropped_mask, os.path.basename(src_file), os.path.basename(dst_file)


# for predict
class NCCT_GAN_PREDICT_UTILS:
    def __init__(self):
        super().__init__()

    def get_image_tensors(self, infile, crop_size, is_dcm=False):
        src_img = sitk.ReadImage(infile)
        src_data = sitk.GetArrayFromImage(src_img)

        # padding to 32xn/Nxn
        padding = 32
        [pd, ph, pw] = crop_size
        [d,h,w] = src_data.shape
        new_d = ((d+pd-1)//pd)*pd
        new_h = ((h+ph-1)//ph)*ph
        new_w = ((w+pw-1)//pw)*pw

        if not np.all([d,h,w]==np.array([new_d, new_h, new_w])):
            new_arr = np.zeros([new_d, new_h, new_w])
            new_arr[:d,:h,:w] = src_data
        else:
            new_arr = src_data

        cropped_srcs = []
        d_cnt = (d+pd-1)//pd
        h_cnt = (h+ph-1)//ph
        w_cnt = (w+pw-1)//pw
        for iz in range(d_cnt):
            for iy in range(h_cnt):
                for ix in range(w_cnt):
                    cropped_src = new_arr[iz*pd:(iz+1)*pd, iy*ph:(iy+1)*ph, ix*pw:(ix+1)*pw]
                    cropped_src = torch.from_numpy(cropped_src).float()
                    cropped_src = torch.unsqueeze(cropped_src, axis=0)
                    cropped_src = torch.unsqueeze(cropped_src, axis=0)
                    cropped_srcs.append(cropped_src)
        
        return cropped_srcs, d_cnt, h_cnt, w_cnt

    def compose_arrays_to_image(self, arr, blocks_dim, crop_size):
        assert len(arr) == blocks_dim[0] * blocks_dim[1] * blocks_dim[2]
        dim = np.array(blocks_dim)*np.array(crop_size)
        dst_arr = np.zeros(dim)
        [d_cnt, h_cnt, w_cnt] = blocks_dim
        [pd, ph, pw] = crop_size
        for iz in range(d_cnt):
            for iy in range(h_cnt):
                for ix in range(w_cnt):
                    dst_arr[iz*pd:(iz+1)*pd, iy*ph:(iy+1)*ph, ix*pw:(ix+1)*pw] = arr[iz*h_cnt*w_cnt+iy*w_cnt+ix]
        return dst_arr


# for test crop logic
class NCCT_GAN_MASK_DS_X(Dataset):
    def __init__(self, root_dir, config_file, phase, crop_size, scale_size, debug=False):
        super().__init__()
        self.src_list = []
        self.dst_list = []
        self.mask_list = []
        self.boundary_list = []
        self.root_dir = root_dir
        self.phase = phase
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.debug = debug
        self.error_indexs = global_ncct_error_list
        # self.error_indexs = ['475170', '372829', '463311']
        # self.error_indexs = ['250238', '317892', '462086', '456640', '475170', '372829', '462630', '417357', '458192', '429884', '456831'] # 462630有问题需要验证，选取图像出的问题
        # 配准有问题
        # hospital_4_2: '417357', '458192'， '429884':中间有断层？, '456831', '175928'疑似, 
        # hospital_4: '462630'
        with open(config_file, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                if line is None or len(line) == 0:
                    continue
                ss = line.split('\t')
                if len(ss) < 9:
                    continue
                src_file = os.path.join(root_dir, ss[0])
                dst_file = os.path.join(root_dir, ss[1])
                mask_file = os.path.join(root_dir, ss[8])
                if not os.path.isfile(src_file):
                    print('{} is not file!'.format(src_file))
                    continue
                if not os.path.isfile(dst_file):
                    print('{} is not file!'.format(dst_file))
                    continue
                # 去除事先查看过的有问题的数据
                index = os.path.basename(ss[0]).split('_')[0]
                if index in self.error_indexs:
                    continue
                d = int(ss[3])-int(ss[2])
                h = int(ss[5])-int(ss[4])
                w = int(ss[7])-int(ss[6])
                cropped_boundary = CroppedBoundary(int(ss[2]), int(ss[3]), int(ss[4]), int(ss[5]), int(ss[6]), int(ss[7]))
                if d < self.crop_size[0] or h < self.crop_size[1] or w < self.crop_size[2]:
                    # 有些NCCT图像，只采集了几层，需要去掉
                    continue
                    # pass
                self.src_list.append(src_file)
                self.dst_list.append(dst_file)
                self.mask_list.append(mask_file)
                self.boundary_list.append(cropped_boundary)
            print('====> data count is:\t{}'.format(len(self.src_list)))

        self.test_arr = np.zeros([700, 512, 512])
    
    def __random_crop_data(self, ct_data, dwi_data, mask_data, size, cropped_boundary):
        # [img_d, img_h, img_w] = dwi_data.shape
        padding = 1
        [img_d, img_h, img_w] = [cropped_boundary.boundary_d_max+padding, cropped_boundary.boundary_h_max+padding, cropped_boundary.boundary_w_max+padding]
        [input_d, input_h, input_w] = size
        # assert np.all(np.less_equal(size, dwi_data.shape))
        z_min_upper = img_d - input_d
        y_min_upper = img_h - input_h
        x_min_upper = img_w - input_w

        # Z_min = np.random.randint(0, z_min_upper)
        # Y_min = np.random.randint(0, y_min_upper)
        # X_min = np.random.randint(0, x_min_upper)


        # print('cropped_boundary.boundary_d_min-padding:\t', cropped_boundary.boundary_d_min-padding)
        # print('z_min_upper\t', z_min_upper)
        # print('cropped_boundary.boundary_h_min-padding\t', cropped_boundary.boundary_h_min-padding)
        # print('y_min_upper\t', y_min_upper)
        # print('cropped_boundary.boundary_w_min-padding\t', cropped_boundary.boundary_w_min-padding)
        # print('x_min_upper\t', x_min_upper)
        Z_min = np.random.randint(cropped_boundary.boundary_d_min-padding, z_min_upper)
        Y_min = np.random.randint(cropped_boundary.boundary_h_min-padding, y_min_upper)
        X_min = np.random.randint(cropped_boundary.boundary_w_min-padding, x_min_upper)

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

        return ct_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], dwi_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], mask_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __center_crop_data(self, ct_data, dwi_data, mask_data, size, cropped_boundary):
        # [img_d, img_h, img_w] = dwi_data.shape
        padding = 1
        [img_d, img_h, img_w] = [cropped_boundary.boundary_d_max+padding, cropped_boundary.boundary_h_max+padding, cropped_boundary.boundary_w_max+padding]
        [input_d, input_h, input_w] = size
        # assert np.all(np.less_equal(size, dwi_data.shape))
        Z_min = img_d//2-input_d//2
        Y_min = img_h//2-input_h//2
        X_min = img_w//2-input_w//2
        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

        return ct_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], dwi_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max], mask_data[Z_min: Z_max, Y_min: Y_max, X_min: X_max]

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):
        if self.phase == 'train':
            src_file = self.src_list[idx]
            print(src_file)
            dst_file = self.dst_list[idx]
            mask_file = self.mask_list[idx]
            # src_img = sitk.ReadImage(src_file)
            # src_data = sitk.GetArrayFromImage(src_img)
            # dst_img = sitk.ReadImage(dst_file)
            # dst_data = sitk.GetArrayFromImage(dst_img)
            # mask_img = sitk.ReadImage(mask_file)
            # mask_data = sitk.GetArrayFromImage(mask_img)
            src_data = self.test_arr
            dst_data = self.test_arr
            mask_data = self.test_arr
            
            if np.random.rand() < 0.8:
                cropped_src, cropped_dst, cropped_mask = self.__random_crop_data(src_data, dst_data, mask_data, self.crop_size, self.boundary_list[idx])
            else:
                cropped_src, cropped_dst, cropped_mask = self.__center_crop_data(src_data, dst_data, mask_data, self.crop_size, self.boundary_list[idx])

            if self.debug:
                mid_dir = os.path.join(self.root_dir, 'tmp')
                os.makedirs(mid_dir, exist_ok=True)
                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(mid_dir, 'src_index_{}.nii.gz'.format(idx)))
                writer.Execute(sitk.GetImageFromArray(cropped_src))

                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(mid_dir, 'dst_index_{}.nii.gz'.format(idx)))
                writer.Execute(sitk.GetImageFromArray(cropped_dst))

                writer = sitk.ImageFileWriter()
                writer.SetFileName(os.path.join(mid_dir, 'mask_index_{}.nii.gz'.format(idx)))
                writer.Execute(sitk.GetImageFromArray(cropped_mask))

            # cropped_src = torch.from_numpy(cropped_src).float()
            # cropped_src = torch.unsqueeze(cropped_src, axis=0)
            # cropped_dst = torch.from_numpy(cropped_dst).float()
            # cropped_dst = torch.unsqueeze(cropped_dst, axis=0)
            # cropped_mask = torch.from_numpy(cropped_mask).float()
            # cropped_mask = torch.unsqueeze(cropped_mask, axis=0)
            print('cropped_src:\t{}'.format(cropped_src.shape))
            print('cropped_dst:\t{}'.format(cropped_dst.shape))
            print('cropped_mask:\t{}'.format(cropped_mask.shape))
            return cropped_src, cropped_dst, cropped_mask, os.path.basename(src_file), os.path.basename(dst_file)



def test_NCCT_GAN_MASK_DS():

    class Options():
        def __init__(self):
            self.lr = 2e-4
            self.beta1 = 0.5
            self.gan_mode = 'lsgan'
            self.direction = 'AtoB'
            self.lambda_L1 = 2
            self.epochs = 1000
            self.num_workers = 2
            self.batch_size = 2
            self.pin_memory = True
            self.display = 2
            self.save_interval = 10
            self.intermidiate_result_root = '../data/gan/hospital_6/experiment_registration2/8.1.out/train_result/intermidiate_result_{}'.format(__file__.split('.')[0])
            # add patch discriminator
            self.patch_D = False
            self.num_patches_D = 5
            self.patch_size_D = [64, 64, 64]
            # crop_size
            self.crop_size = [32, 448, 448]

            self.root_dir = '../data/gan/hospital_6/experiment_registration2/8.1.out'
            self.config_file = '../data/gan/hospital_6/experiment_registration2/8.1.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt'
            self.check_point = None

    opt = Options()
    ds = NCCT_GAN_MASK_DS(opt.root_dir, opt.config_file, 'train', opt.crop_size, opt.crop_size, debug=False)
    data_loader = DataLoader(ds, num_workers=opt.num_workers, batch_size=opt.batch_size, pin_memory=True, shuffle=True)

    for _ in range(100):
        for i, (srcs, dsts, masks, _, _) in tqdm(enumerate(data_loader)):
            print(srcs.shape)
            print(dsts.shape)


def test_NCCT_GAN_MASK_DS_X():
    
    class Options():
        def __init__(self):
            self.lr = 2e-4
            self.beta1 = 0.5
            self.gan_mode = 'lsgan'
            self.direction = 'AtoB'
            self.lambda_L1 = 2
            self.epochs = 1000
            self.num_workers = 8
            self.batch_size = 2
            self.pin_memory = True
            self.display = 2
            self.save_interval = 10
            self.intermidiate_result_root = '../data/gan/hospital_6/experiment_registration2/8.1.out/train_result/intermidiate_result_{}'.format(__file__.split('.')[0])
            # add patch discriminator
            self.patch_D = False
            self.num_patches_D = 5
            self.patch_size_D = [64, 64, 64]
            # crop_size
            self.crop_size = [32, 448, 448]

            self.root_dir = '../data/gan/hospital_6/experiment_registration2/8.1.out'
            self.config_file = '../data/gan/hospital_6/experiment_registration2/8.1.out/config/mask_ncct_to_dwi_bxxx_train_config_file.txt'
            self.check_point = None

    opt = Options()
    ds = NCCT_GAN_MASK_DS_X(opt.root_dir, opt.config_file, 'train', opt.crop_size, opt.crop_size, debug=False)
    data_loader = DataLoader(ds, num_workers=opt.num_workers, batch_size=opt.batch_size, pin_memory=True, shuffle=True)

    for _ in range(100):
        for i, (srcs, dsts, masks, _, _) in tqdm(enumerate(data_loader)):
            print(srcs.shape)
            print(dsts.shape)


def test_NCCT_GAN_PREDICT_UTILS():
    utils = NCCT_GAN_PREDICT_UTILS()
    crop_size = [128,256,256]
    image_tensors,d_cnt,h_cnt,w_cnt = utils.get_image_tensors('../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct/114093_first_BS_NCCT.nii.gz', crop_size)
    print(len(image_tensors))
    print(image_tensors[0].shape)

    arr = []
    for img_tensor in image_tensors:
        sub_arr = torch.squeeze(img_tensor).numpy()
        arr.append(sub_arr)
    
    dst_arr = utils.compose_arrays_to_image(arr, [d_cnt, h_cnt, w_cnt], crop_size)



    print('finish test_NCCT_GAN_PREDICT_UTILS!')

if __name__ == '__main__':
    test_NCCT_GAN_MASK_DS()
    # test_NCCT_GAN_PREDICT_UTILS()