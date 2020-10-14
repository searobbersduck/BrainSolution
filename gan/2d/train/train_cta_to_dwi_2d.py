import os
import sys

curdir = os.path.abspath(os.curdir) # gan/2d/train
root_dir=os.path.join(os.path.dirname(curdir))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'external_lib/pytorch-CycleGAN-and-pix2pix'))

from models.pix2pix_model import Pix2PixModel
from options.train_options import TrainOptions
from options.test_options import TestOptions
from util.visualizer import save_images
from util import html

import time
from models import create_model
from util.visualizer import Visualizer

from datasets.cta_gan_datasets import CTA_GAN_DS
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import SimpleITK as sitk
import torch
import numpy as np

from glob import glob

def train():
    opt = TrainOptions().parse()

    data_root = '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/train'
    config_file = '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/config/cta_to_dwi_2d_train.txt'
    crop_size = [512, 512]

    ds = CTA_GAN_DS(data_root, config_file, 'train', crop_size, crop_size)
    data_loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)

    total_iters = 0                # the total number of training iterations

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        for i, (srcs, dsts, _, _) in tqdm(enumerate(data_loader)):
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            data = {}
            data['A'] = srcs
            data['B'] = dsts
            data['A_paths'] = 'A'
            data['B_paths'] = ['B']

            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / len(data_loader), losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
            
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)


def predict_onecase(infile_A, infile_B, outdir, model=None):
    '''
    debug cmd:  predict_onecase('../../data/task2/1.1.raw/1.2.156.112605.14038007945377.191013010825.3.5228.61295_1.2.156.112605.14038007945377.191013011003.3.5228.104694/m_ptrRawImage.nii.gz', '../../data/task2/1.1.raw/1.2.156.112605.14038007945377.191013010825.3.5228.61295_1.2.156.112605.14038007945377.191013011003.3.5228.104694/diff.nii.gz', '../../data/tmp/1.nii.gz')
    debug cmd:  predict_onecase('../../data/gan/hospital_6/experiment_registration2/8.2.out/NCCT/1014186_first_BS_NCCT.nii.gz', '../../data/gan/hospital_6/experiment_registration2/8.2.out/DWI_BXXX/1014186_first_FU_DWI_BXXX.nii.gz', '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/tmp')
    '''
    opt = TestOptions().parse()
    if model is None:
        # opt = TestOptions().parse()
        model = create_model(opt)
        model.setup(opt)
        model.eval()

    series_uid = os.path.basename(os.path.dirname(infile_A))
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch), '{}'.format(series_uid))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    os.makedirs(outdir, exist_ok=True)
    in_img_A = sitk.ReadImage(infile_A)
    in_arr_A = sitk.GetArrayFromImage(in_img_A)
    in_img_B = sitk.ReadImage(infile_B)
    in_arr_B = sitk.GetArrayFromImage(in_img_B)
    
    out_arr_B = np.zeros(in_arr_B.shape)

    assert in_arr_A.shape == in_arr_B.shape

    for i in range(in_arr_A.shape[0]):
        slice_arr_A = in_arr_A[i]
        slice_arr_B = in_arr_B[i]
        srcs = torch.from_numpy(slice_arr_A).unsqueeze(0).unsqueeze(0).float()
        dsts = torch.from_numpy(slice_arr_B).unsqueeze(0).unsqueeze(0).float()
        data = {}
        data['A'] = srcs
        data['B'] = dsts
        data['A_paths'] = ['{}_{}_A'.format(111, i)]
        data['B_paths'] = ['{}_{}_B'.format(111, i)]
        model.set_input(data)
        model.test()
        fake_B = model.fake_B.detach().cpu().numpy().squeeze()
        out_arr_B[i] = fake_B
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
    webpage.save()
    out_arr_B = np.array(out_arr_B, dtype=np.int16)
    out_img_B = sitk.GetImageFromArray(out_arr_B)
    in_img_B.CopyInformation(in_img_A)
    out_img_B.CopyInformation(in_img_A)
    # sitk.WriteImage(out_img_B, os.path.join(outdir, 'fake_B.nii.gz'))
    # sitk.WriteImage(in_img_A, os.path.join(outdir, 'real_A.nii.gz'))
    # sitk.WriteImage(in_img_B, os.path.join(outdir, 'real_B.nii.gz'))
    out_real_a_name = os.path.basename(infile_A)
    out_real_b_name = os.path.basename(infile_B)
    out_fake_b_name = out_real_a_name.replace('.nii.gz', '_fake_2d.nii.gz')
    
    os.makedirs(os.path.join(outdir, 'FAKE_DWI_2D'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'CTA'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'REAL_DWI'), exist_ok=True)
    
    sitk.WriteImage(out_img_B, os.path.join(outdir, 'FAKE_DWI_2D', out_fake_b_name))
    sitk.WriteImage(in_img_A, os.path.join(outdir, 'CTA', out_real_a_name))
    sitk.WriteImage(in_img_B, os.path.join(outdir, 'REAL_DWI', out_real_b_name))


def predict_singletask(root_dir, config_file, outdir):
    '''
    debug cmd: predict_singletask('../../data/gan/hospital_6/experiment_registration2/8.2.out', '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/config/cta_to_dwi_2d_test.txt', '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/tmp')
    '''
    cta_files = []
    dwi_files = []
    with open(config_file) as f:
        for line in f.readlines():
            line = line.strip()
            if line is None or len(line) == 0:
                continue
            ss = line.split('\t')
            cta_file = os.path.join(root_dir, ss[0])
            dwi_file = os.path.join(root_dir, ss[1])
            cta_files.append(cta_file)
            dwi_files.append(dwi_file)

    for i in tqdm(range(len(cta_files))):
        predict_onecase(cta_files[i], dwi_files[i], outdir)


def predict_singletask2(root_dir, test_gen_dir, out_dir):
    '''
    root_dir:       ../../data/gan/hospital_6/experiment_registration2/10.predict_retain
    test_gen_dir:   ../../data/gan/hospital_6/experiment_registration2/10.predict_retain/CTA
    out_dir:        ../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/tmp

    debug cmd: predict_singletask2('../../data/gan/hospital_6/experiment_registration2/10.predict_retain', '../../data/gan/hospital_6/experiment_registration2/10.predict_retain/CTA', '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/tmp')
    '''
    cta_pattern = '_first_BS_NCCT.nii.gz'
    dwi_pattern = '_first_FU_DWI_BXXX.nii.gz'
    cta_files = glob(os.path.join(test_gen_dir, '*{}'.format(cta_pattern)))
    pids = [os.path.basename(i).split('_')[0] for i in cta_files]
    cta_root = os.path.join(root_dir, 'CTA')
    dwi_root = os.path.join(root_dir, 'REAL_DWI')
    cta_files = [os.path.join(cta_root, '{}{}'.format(i, cta_pattern)) for i in pids]
    dwi_files = [os.path.join(dwi_root, '{}{}'.format(i, dwi_pattern)) for i in pids]

    opt = TestOptions().parse()
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    for i in tqdm(range(len(cta_files))):
        predict_onecase(cta_files[i], dwi_files[i], out_dir, model)



if __name__ == '__main__':
    # train()
    # predict_onecase('../../data/gan/hospital_6/experiment_registration2/8.2.out/NCCT/1014186_first_BS_NCCT.nii.gz', '../../data/gan/hospital_6/experiment_registration2/8.2.out/DWI_BXXX/1014186_first_FU_DWI_BXXX.nii.gz', '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/tmp')
    # predict_singletask('../../data/gan/hospital_6/experiment_registration2/8.2.out', '../../data/gan/hospital_6/experiment_registration2/8.2.out/config/anno_mask_ncct_to_dwi_bxxx_test_config_file.txt', '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/tmp')
    predict_singletask2('../../data/gan/hospital_6/experiment_registration2/10.predict_retain', '../../data/gan/hospital_6/experiment_registration2/10.predict_retain/CTA', '../../data/gan/hospital_6/experiment_registration2/8.2.out/slice_2d/tmp')