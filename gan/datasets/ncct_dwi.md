## 数据处理
- [x] 0. 从原始的病人数据中，提取相应的NCCT、DWI、ADC数据，并copy到相应的路径下
    * 调用命令`python gan_utils.py ncct_extract_series_from_raw_series '../data/gan/ncct2dwi/siyuan_dcm_with_pid' '../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm' '../data/gan/ncct2dwi/experiment_registration2/config/V1 四院NCCT-DWI-ADC-RAPID.xlsx'`
    * 生成数据的路径位于:`'../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm'`

```
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

```
- [x] 1. 将dcm数据生成相应的nii.gz文件以便后续处理
    * 确保数据存在，如不存在，请执行上一步骤；
    * 调用命令：`python gan_utils.py ncct_convert_dcm_to_niigz '../data/gan/ncct2dwi/experiment_registration2/0.raw_dcm' '../data/gan/ncct2dwi/experiment_registration2/1.nii_file'`
    * 生成数据的路径位于:`'../data/gan/ncct2dwi/experiment_registration2/1.nii_file'`
    * ***遇到问题：NCCT数据经过itk读入之后的direction信息有错误，需要注意校正***；

- [x] 2. 将数据的origin统一设置到(0,0,0),以便有序在看图软件中对比查看
    * 调用命令：`python gan_utils.py ncct_set_original_point '../data/gan/ncct2dwi/experiment_registration2/1.nii_file' '../data/gan/ncct2dwi/experiment_registration2/2.nii_file_ori'`
    * 生成数据的路径位于:`'../data/gan/ncct2dwi/experiment_registration2/2.nii_file_ori'`

- [x] 3. 提取出NCCT图像中的脑实质部分
    * `ln -s 2.nii_file_ori 4\ Patient_nii_unity`
    * 调用命令：`python xxx_test_registration.py`
    * 生成数据存在于`'../data/gan/ncct2dwi/experiment_registration2/4 Patient_nii_unity'`, 其中的`*_brain.nii.gz`文件即为脑实质文件
    * (**推荐**)或者调用命令：`python gan_utils.py extract_cerebral_parenchyma_multiprocess '../data/gan/ncct2dwi/experiment_registration2/4 Patient_nii_unity' '../data/gan/ncct2dwi/experiment_registration2/4 Patient_nii_unity' _NCCT.nii.gz _brain.nii.gz`


- [x] 4. 刚性配准生成DWI_B0、DWI_BXXX、ADC、brain(脑实质部分)、NCCT等图像
    * 配准过程以NCCT数据作为基准
    * 调用命令：`python simpleinfer_projects/CTA_ASPECTS/gan_ncct_rigid_align.py`
    * 生成数据存在于`'../data/gan/ncct2dwi/experiment_registration2/5 dwi_rigid_align_ncct'`， 路径为配准之后的重采样数据
    * 注意：
      * 目前配准对于每种模态使用的是不同的配准transform,***之后应该考虑合并***
      * 目前在做配准变换时，采用的是linear插值，***之后需要尝试nearest***
      * 在加载图像时，直接利用sitk.ReadImage(xxx, sitk.sitkFloat32)读取图像并进行配准的结果，和正常加载并在过程中利用sitk.Cast转换为sitkFloat32再做配准的结果不一样，目前采用的是直接利用sitk.ReadImage(xxx, sitk.sitkFloat32)方式进行加载；
      * 第一次配准：肉眼进行比对***BXXX和脑实质***图像发现如下问题图像， ***其它的GAN需要分别进行比对，todo***
        * 配准有小差别：250238， 317892， 462086
        * 配准有较大差别，图像本身就有问题：456640，475170， 372829

- [x] 5. 提取脑实质部分的mask
    * 调用命令`python gan_utils.py ncct_generate_cerebral_parenchyma '../data/gan/ncct2dwi/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/ncct2dwi/experiment_registration2/8.out/cerebral_parenchyma' *_brain.nii.gz`
    * 或者调用多线程命令`python gan_utils.py ncct_generate_cerebral_parenchyma_multiprocess '../data/gan/ncct2dwi/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/ncct2dwi/experiment_registration2/8.out/cerebral_parenchyma' *_brain.nii.gz 12`
    * 此处的mask，所有有效层面(***当前层面和最大层面的比值>0.5***)统一到和最大层面的mask相同；

- [ ] 5.1 计算CT和DWI图像的边界， 在做配准和重采样运算时，会因为图像旋转等，产生没有值的区域，这部分区域一般会被设置为默认值，在实际进行生成模型的训练时，该区域不能加入运算，因此需要事先计算出这部分区域


- [ ] 5.2 取中间层的上下64层数据
    * 调用命令：`python gan_utils.py ncct_generate_cerebral_parenchyma_middle_layer_multiprocess '../data/gan/ncct2dwi/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/ncct2dwi/experiment_registration2/8.1.out/cerebral_parenchyma' *_brain.nii.gz 12`

- [x] 6. 根据脑实质的mask，将相应的CTA和DWI部分截取出来
    * 提取NCCT，调用命令：`python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/ncct2dwi/experiment_registration2/8.out/cerebral_parenchyma' '../data/gan/ncct2dwi/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/ncct2dwi/experiment_registration2/8.out/NCCT' *brain*.nii.gz *NCCT.nii.gz`
    * 提取DWI_B0，调用命令：`python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/ncct2dwi/experiment_registration2/8.out/cerebral_parenchyma' '../data/gan/ncct2dwi/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/ncct2dwi/experiment_registration2/8.out/DWI_B0' *brain*.nii.gz *DWI_B0.nii.gz`
    * 提取DWI_BXXX，调用命令：`python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/ncct2dwi/experiment_registration2/8.out/cerebral_parenchyma' '../data/gan/ncct2dwi/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/ncct2dwi/experiment_registration2/8.out/DWI_BXXX' *brain*.nii.gz *DWI_BXXX.nii.gz`
    * 提取ADC，调用命令：`python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/ncct2dwi/experiment_registration2/8.out/cerebral_parenchyma' '../data/gan/ncct2dwi/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/ncct2dwi/experiment_registration2/8.out/ADC' *brain*.nii.gz *ADC.nii.gz`

- [x] 7. 根据脑实质mask，生成配置文件
    * 生成原始分辨率的配置文件，调用命令：`python gan_utils.py ncct_genereate_cta2dwi_config_file_with_cerebral_parenchyma ../data/gan/ncct2dwi/experiment_registration2/8.out ../data/gan/ncct2dwi/experiment_registration2/8.out/config`

- [x] 统计数据统计xyz三个方向上的有效范围，并记录最大和最小范围，以确定网络的crop边界，即crop的范围不能超出最小边界
    * 此处以脑实质的mask作为参考，即所有`mask=1`的区域为有效区域
    * 调用命令: `python cta_to_dwi_dataset.py calculate_crop_range ../data/gan/ncct2dwi/experiment_registration2/8.out/config/ncct_to_dwi_b0_train_config_file.txt`
    * 结果：
    ```
    min depth: 196
    max depth: 287
    min height: 289
    max height: 428
    min width: 240
    max width: 345
    ```
    ```
    min depth: 162
    max depth: 237
    min height: 289
    max height: 428
    min width: 233
    max width: 345
    ```    
    * 训练神经网络时，crop size的初步取值[160, 256, 224]



## 4院数据处理
- [ ] 1. 将原始文件夹转换为按照patient_id分类的文件夹
    * 调用命令: `python gan_utils.py ncct_extract_from_hospital_folder_all`, 输出到文件夹`../data/gan/hospital_4/0.ori`
- [ ] 1.1. 生成表格
    * 生成所有的病人+series相关的表格，调用命令：`python gan_utils.py ncct_generate_table_all_series '../data/gan/hospital_4/0.ori' '../data/gan/hospital_4/0.table'`, 第二批数据： `python gan_utils.py ncct_generate_table_all_series '../data/gan/hospital_4_2/0.ori' '../data/gan/hospital_4_2/0.table'`
    * 生成ncct\dwi\adc配对的表格，调用命令：`python gan_utils.py ncct_generate_table_all_ncct_dwi_adc_pairs '../data/gan/hospital_4/0.ori' '../data/gan/hospital_4/0.table' 10800`, 其中参数`10800`代表采集时间`3h`; 第二批数据： `python gan_utils.py ncct_generate_table_all_ncct_dwi_adc_pairs '../data/gan/hospital_4_2/0.ori' '../data/gan/hospital_4_2/0.table' 10800`
- [ ] 2. 提取出要做后续处理的DWI、ADC、NCCT配对的数据
    * 调用命令: `python gan_utils.py ncct_extract_infos_from_patients_all '../data/gan/hospital_4/0.ori' '../data/gan/hospital_4/0.raw_dcm'`

- [ ] 批处理, 处理逻辑参照上文：
```
python utils.py ncct_extract_infos_from_patients_all '../data/gan/hospital_4/0.ori' '../data/gan/hospital_4/0.raw_dcm'
python gan_utils.py ncct_convert_dcm_to_niigz_multiprocess '../data/gan/hospital_4/0.raw_dcm' '../data/gan/hospital_4/experiment_registration2/1.nii_file'
python gan_utils.py ncct_set_original_point '../data/gan/hospital_4/experiment_registration2/1.nii_file' '../data/gan/hospital_4/experiment_registration2/2.nii_file_ori'
ln -s ../data/gan/hospital_4/experiment_registration2/2.nii_file_ori ../data/gan/hospital_4/experiment_registration2/4\ Patient_nii_unity
python gan_utils.py extract_cerebral_parenchyma_multiprocess '../data/gan/hospital_4/experiment_registration2/4 Patient_nii_unity' '../data/gan/hospital_4/experiment_registration2/4 Patient_nii_unity' _NCCT.nii.gz _brain.nii.gz

python gan_ncct_rigid_align.py

python gan_utils.py ncct_generate_cerebral_parenchyma_multiprocess '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.out/cerebral_parenchyma' *_brain.nii.gz 12

python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_4/experiment_registration2/8.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.out/NCCT' *brain*.nii.gz *NCCT.nii.gz
python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_4/experiment_registration2/8.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.out/DWI_B0' *brain*.nii.gz *DWI_B0.nii.gz
python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_4/experiment_registration2/8.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.out/DWI_B0' *brain*.nii.gz *DWI_BXXX.nii.gz
python cta_to_dwi_dataset.py extract_region_by_mask '../data/gan/hospital_4/experiment_registration2/8.out/cerebral_parenchyma' '../data/gan/hospital_4/experiment_registration2/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration2/8.out/DWI_B0' *brain*.nii.gz *ADC.nii.gz

python gan_utils.py ncct_genereate_cta2dwi_config_file_with_cerebral_parenchyma ../data/gan/hospital_4/experiment_registration2/8.out ../data/gan/hospital_4/experiment_registration2/8.out/config

python cta_to_dwi_dataset.py calculate_crop_range ../data/gan/hospital_4/experiment_registration2/8.out/config/ncct_to_dwi_b0_train_config_file.txt
```


## 核心梗死区和半暗带相关信息提取

- [ ] 1. 提取核心梗死区和半暗带的mask
    * 调用命令`invoke cmd: python gan_utils.py rapid_extract_sumary_info_multiprocess '../data/gan/hospital_4/0.raw_dcm' '../data/gan/hospital_4/experiment_registration3/1.rapid'`
- [ ] 2. 将起点设置为0
    * 调用命令`python gan_utils.py ncct_set_original_point '../data/gan/hospital_4/experiment_registration3/1.rapid' '../data/gan/hospital_4/experiment_registration3/2.nii_file_ori'`
- [ ] 3. 配准
    * 调用命令`ln -s ../data/gan/hospital_4/experiment_registration3/2.nii_file_ori ../data/gan/hospital_4/experiment_registration3/4 Patient_nii_unity`
    * 调用命令`python gan_ncct_rapid_rigid_align.py`, 生成的数据位于`../data/gan/hospital_4/experiment_registration3/5 dwi_rigid_align_ncct`
- [ ] 4. 生成界面图片以便查看配准是否正确
    * 调用命令`python gan_utils.py extract_mpr_multiprocess '../data/gan/hospital_4/experiment_registration3/5 dwi_rigid_align_ncct' '../data/gan/hospital_4/experiment_registration3/8.2.out/projection'`
    * 经查看hospital_4_2中问题图片:'398774', '448646', '458192'