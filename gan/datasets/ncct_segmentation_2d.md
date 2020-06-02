# NCCT分割（2d）

## NCCT-INFARCT(核心梗死区)分割

### 数据准备

```

python ncct_segmentation2d_dataset.py ncct_convert_3d_2d '../data/gan/hospital_4_2/experiment_registration3/5 dwi_rigid_align_ncct' '../data/gan/hospital_4_2/experiment_seg_2d/infarct' '../data/gan/hospital_4_2/experiment_registration3/1.rapid/config.txt' _BS_NCCT.nii.gz _FU_DWI_INFARCT_MASK.nii.gz 0.9 0.1
python ncct_segmentation2d_dataset.py generate_config_file '../data/gan/hospital_4_2/experiment_seg_2d/infarct/train' '../data/gan/hospital_4_2/experiment_seg_2d/infarct/config' 'train'

python ncct_segmentation2d_dataset.py generate_config_file '../data/gan/hospital_4_2/experiment_seg_2d/infarct/val' '../data/gan/hospital_4_2/experiment_seg_2d/infarct/config' 'val'

python ncct_segmentation2d_dataset.py generate_config_file '../data/gan/hospital_4_2/experiment_seg_2d/infarct/test' '../data/gan/hospital_4_2/experiment_seg_2d/infarct/config' 'test'
```