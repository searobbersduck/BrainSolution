import os
import numpy as np


class CroppedBoundary():
    def __init__(self, boundary_d_min, boundary_d_max, boundary_h_min, boundary_h_max, boundary_w_min, boundary_w_max):
        self.boundary_d_min = boundary_d_min
        self.boundary_d_max = boundary_d_max
        self.boundary_h_min = boundary_h_min
        self.boundary_h_max = boundary_h_max
        self.boundary_w_min = boundary_w_min
        self.boundary_w_max = boundary_w_max

class CropUtils:
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_region_3d_random_crop(size, cropped_boundary):
        '''
        random sample one size block from cropped_boundary region
        '''
        padding = 1
        [img_d, img_h, img_w] = [cropped_boundary.boundary_d_max+padding, cropped_boundary.boundary_h_max+padding, cropped_boundary.boundary_w_max+padding]
        [input_d, input_h, input_w] = size

        z_min_upper = img_d - input_d
        y_min_upper = img_h - input_h
        x_min_upper = img_w - input_w

        Z_min = np.random.randint(cropped_boundary.boundary_d_min, z_min_upper)
        Y_min = np.random.randint(cropped_boundary.boundary_h_min, y_min_upper)
        X_min = np.random.randint(cropped_boundary.boundary_w_min, x_min_upper)

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

        return Z_min, Z_max, Y_min, Y_max, X_min, X_max


    @staticmethod
    def get_region_3d_center_crop(size, cropped_boundary):
        padding = 1
        [img_d, img_h, img_w] = [cropped_boundary.boundary_d_max+padding, cropped_boundary.boundary_h_max+padding, cropped_boundary.boundary_w_max+padding]
        center_d =  (cropped_boundary.boundary_d_max+padding + cropped_boundary.boundary_d_min) // 2
        center_h =  (cropped_boundary.boundary_h_max+padding + cropped_boundary.boundary_h_min) // 2
        center_w =  (cropped_boundary.boundary_w_max+padding + cropped_boundary.boundary_w_min) // 2
        [input_d, input_h, input_w] = size

        Z_min = center_d-input_d//2
        Y_min = center_h-input_h//2
        X_min = center_w-input_w//2

        Z_max = Z_min + input_d
        Y_max = Y_min + input_h
        X_max = X_min + input_w

        return Z_min, Z_max, Y_min, Y_max, X_min, X_max


    @staticmethod
    def get_region_3d_random_skip_layers_crop(size, cropped_boundary):
        '''
        random select z layers from d-direction
        '''
        padding = 1
        [img_d, img_h, img_w] = [cropped_boundary.boundary_d_max+padding, cropped_boundary.boundary_h_max+padding, cropped_boundary.boundary_w_max+padding]
        [input_d, input_h, input_w] = size
        # assert np.all(np.less_equal(size, dwi_data.shape))
        z_min_upper = img_d - input_d
        y_min_upper = img_h - input_h
        x_min_upper = img_w - input_w

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

        # random select z from [cropped_boundary.boundary_d_min, cropped_boundary.boundary_d_max]
        z_index_list = list(range(cropped_boundary.boundary_d_min, cropped_boundary.boundary_d_max+1))
        z_index = np.random.choice(z_index_list, size[0])
        z_index.sort()

        return z_index, Y_min, Y_max, X_min, X_max