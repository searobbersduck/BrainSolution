import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from skimage.feature import local_binary_pattern
import numpy as np
from torchvision.models.vgg import vgg16

class PerceptionLoss(nn.Module):
    def __init__(self, device):
        super(PerceptionLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        loss_network = loss_network.to(device)
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        
    def forward(self, real_tensor, fake_tensor):
        '''
        real_tensor, cuda
        fake_tensor, cuda
        '''
        real_axial = torch.repeat_interleave(real_tensor[:,:,real_tensor.shape[2]//2, :, :], 3, dim=1)
        real_coronal = torch.repeat_interleave(real_tensor[:,:,:,real_tensor.shape[3]//2, :], 3, dim=1)
        real_sagital = torch.repeat_interleave(real_tensor[:,:,:,:,real_tensor.shape[4]//2], 3, dim=1)
        fake_axial = torch.repeat_interleave(fake_tensor[:,:,fake_tensor.shape[2]//2, :, :], 3, dim=1)
        fake_coronal = torch.repeat_interleave(fake_tensor[:,:,:,fake_tensor.shape[3]//2, :], 3, dim=1)
        fake_satital = torch.repeat_interleave(fake_tensor[:,:,:,:,fake_tensor.shape[4]//2], 3, dim=1)
        
        real_axial_f = self.loss_network(real_axial)
        fake_axial_f = self.loss_network(fake_axial)
        
        real_coronal_f = self.loss_network(real_coronal)
        fake_coronal_f = self.loss_network(fake_coronal)

        real_sagital_f = self.loss_network(real_sagital)
        fake_sagital_f = self.loss_network(fake_satital)

        loss_axial = self.mse_loss(real_axial_f, fake_axial_f)
        loss_coronal = self.mse_loss(real_coronal_f, fake_coronal_f)
        loss_sagital = self.mse_loss(real_sagital_f, fake_sagital_f)

        self.loss_axial = loss_axial
        self.loss_coronal = loss_coronal
        self.loss_sagital = loss_sagital

        return (loss_axial + loss_coronal + loss_sagital)/3


def test_PerceptionLoss():
    print('====> test_PerceptionLoss begin:')
    import SimpleITK as sitk
    import time
    real_file = '../data/gan/hospital_6_crop/experiment_registration2/8.2.out/train_result/intermidiate_result_ddp_train_cta_to_dwi_bxxx_hospital6_nonmask_skip_20200805/epoch_480_index_38_dst_real_4495700_first_FU_DWI_BXXX.nii.gz'
    fake_file = '../data/gan/hospital_6_crop/experiment_registration2/8.2.out/train_result/intermidiate_result_ddp_train_cta_to_dwi_bxxx_hospital6_nonmask_skip_20200805/epoch_480_index_38_dst_fake_4495700_first_FU_DWI_BXXX.nii.gz'

    real_img = sitk.ReadImage(real_file)
    fake_img = sitk.ReadImage(fake_file)

    real_arr = sitk.GetArrayFromImage(real_img)
    fake_arr = sitk.GetArrayFromImage(fake_img)

    real_tensor = torch.from_numpy(real_arr).unsqueeze(0).unsqueeze(0)
    fake_tensor = torch.from_numpy(fake_arr).unsqueeze(0).unsqueeze(0)

    local_rank = 0
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    real_tensor = real_tensor.cuda()
    fake_tensor = fake_tensor.cuda()

    perception_loss_obj = PerceptionLoss(device)
    loop_num = 20
    for i in range(loop_num):
        beg = time.time()
        perception_loss = perception_loss_obj(real_tensor, fake_tensor)
        print('perception loss:\t{}, time elapsed:\t{:.3f}s'.format(perception_loss, time.time()-beg))
    print('====> test_PerceptionLoss finish!')


if __name__ == '__main__':
    test_PerceptionLoss()


