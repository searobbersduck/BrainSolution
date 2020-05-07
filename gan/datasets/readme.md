## task
---
### cta to dwi dataset

- [x] 查看所有数据的边界并记录，计算三个方向的最大边界
    * 调用命令：`python cta_to_dwi_dataset.py extract_valid_volume_all ../data/gan/cta2dwi/case_178_forHuang/CTA ../data/gan/cta2dwi/case_178_forHuang/DWI`
    * 此处ct的阈值设置为-100， dwi的阈值范围为100，即所有大于阈值的区域为有效区域
    * 统计结果为：` ====> max range:       [79, 477, 386]`
  
- [x] 将数据resize到统一的分辨率，这里统一使用xy方向上的spacing作为基准
    * 调用命令：`python cta_to_dwi_dataset.py rescale_data_for_cta2dwi ../data/gan/cta2dwi/case_178_forHuang ../data/gan/cta2dwi/case_178_forHuang_rescale`

- [x] 将数据resize到统一的分辨率，这里统一使用xy方向上的spacing作为基准, resize到256
    * 调用命令：`python cta_to_dwi_dataset.py rescale_data_for_cta2dwi ../data/gan/cta2dwi/case_178_forHuang ../data/gan/cta2dwi/case_178_forHuang_rescale 256`

- [x] 查看所有rescale数据的边界并记录，计算三个方向的最大边界
    * 调用命令：`python cta_to_dwi_dataset.py extract_valid_volume_all ../data/gan/cta2dwi/case_178_forHuang_rescale/CTA ../data/gan/cta2dwi/case_178_forHuang_rescale/DWI`
    * 此处ct的阈值设置为-100， dwi的阈值范围为100，即所有大于阈值的区域为有效区域

- [x] 根据rescale数据统计xyz三个方向上的有效范围并生成配置文件
   * 调用命令：`python cta_to_dwi_dataset.py genereate_cta2dwi_range_config_file ../data/gan/cta2dwi/case_178_forHuang_rescale/CTA ../data/gan/cta2dwi/case_178_forHuang_rescale/DWI ../data/gan/cta2dwi/case_178_forHuang_rescale/config/config_file_1.txt`

- [x] 统计rescale数据统计xyz三个方向上的有效范围，并记录最大和最小范围，以确定网络的crop边界，即crop的范围不能超出最小边界
    * 此处ct的阈值设置为-100， dwi的阈值范围为100，即所有大于阈值的区域为有效区域
    * 调用命令: `python cta_to_dwi_dataset.py calculate_crop_range ../data/gan/cta2dwi/case_178_forHuang_rescale/config/config_file_1.txt`
    * 结果：
    ```
        min depth: 182
        max depth: 326
        min height: 287
        max height: 477
        min width: 239
        max width: 386
    ```
    * 训练神经网络时，crop size的初步取值[160, 256, 224]

- [x] 编写pytorch的data generator, 并输出中间结果
    * 调用命令：`python cta_to_dwi_dataset.py check_CTA2DWI_GAN_DS_middle_result`


- [x] 在itk-snap上查看，病灶区域的dwi值，并据此确定z方向上的范围，尽可能包含病灶区域

- [x] 查看原始cta和dwi的direction/spacing/size信息
    * 调用命令：`python cta_to_dwi_dataset.py analyze_cta2dwi_ori_data '../data/gan/cta2dwi/Atlas-crec-CTA-ASPECT/2 Patient_dcm_sorted' ../data/gan/cta2dwi/Atlas-crec-CTA-ASPECT/analysis/cta_and_dwi_info.txt`
    * 信息结果存储于`../data/gan/cta2dwi/Atlas-crec-CTA-ASPECT/analysis/cta_and_dwi_info.txt`

- [x] 
---

### ncct to dwi

- [x] 查看所有数据的边界并记录，计算三个方向的最大边界
    * 调用命令：`python cta_to_dwi_dataset.py extract_valid_volume_all ../data/gan/ncct2dwi/siyuan_NCCT-DWI-align/dwi_rigid_align_ncct/NCCT ../data/gan/ncct2dwi/siyuan_NCCT-DWI-align/dwi_rigid_align_ncct/DWI`
    * 此处ct的阈值设置为-100， dwi的阈值范围为100，即所有大于阈值的区域为有效区域
    * 统计结果为：`  ====> max range:       [133, 511, 511]`

- [x] 将数据resize到统一的分辨率，这里统一使用xy方向上的spacing作为基准
    * 调用命令：`python cta_to_dwi_dataset.py rescale_data_for_ncct2dwi ../data/gan/ncct2dwi/siyuan_NCCT-DWI-align/dwi_rigid_align_ncct ../data/gan/ncct2dwi/siyuan_NCCT-DWI-align/dwi_rigid_align_ncct_rescale`

- [x] 将数据resize到统一的分辨率，这里统一使用xy方向上的spacing作为基准, resize到256
    * 调用命令：`python cta_to_dwi_dataset.py rescale_data_for_ncct2dwi ../data/gan/ncct2dwi/siyuan_NCCT-DWI-align/dwi_rigid_align_ncct ../data/gan/ncct2dwi/siyuan_NCCT-DWI-align/dwi_rigid_align_ncct_rescale 256`

- [x] 查看所有数据的边界并记录，计算三个方向的最大边界
    * 调用命令：`python cta_to_dwi_dataset.py extract_valid_volume_all ../data/gan/ncct2dwi/siyuan_NCCT-DWI-align/dwi_rigid_align_ncct_rescale/NCCT ../data/gan/ncct2dwi/siyuan_NCCT-DWI-align/dwi_rigid_align_ncct_rescale/DWI`
    * 此处ct的阈值设置为-100， dwi的阈值范围为100，即所有大于阈值的区域为有效区域
    * 统计结果为：`  ====> max range:       [382, 511, 511]`


- [x] 根据rescale数据统计xyz三个方向上的有效范围并生成配置文件
   * 调用命令：`python cta_to_dwi_dataset.py genereate_cta2dwi_range_config_file ../data/gan/ncct2dwi/siyuan_NCCT-DWI-align/dwi_rigid_align_ncct_rescale/NCCT ../data/gan/ncct2dwi/siyuan_NCCT-DWI-align/dwi_rigid_align_ncct_rescale/DWI ../data/gan/ncct2dwi/siyuan_NCCT-DWI-align/dwi_rigid_align_ncct_rescale/config/config_file_1.txt *_NCCT.nii* _first_FU_DWI.nii.gz`



---
---
思路：先提取出cta图像的脑实质部分，分别做cta和dwi图像在脑实质部分的mask
- [ ] 提取脑实质部分的mask
    * 调用命令`python cta_to_dwi_dataset.py generate_cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration/cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_mask/cerebral_parenchyma *_brain_rigid_aligned.nii.gz`
    * 暂时无用，对于脑实质靠下的部分，脑实质区域较小，会影响判断

- [x] 将数据resize到统一的分辨率，这里统一使用xy方向上的spacing作为基准
    * 调用命令：`python cta_to_dwi_dataset.py rescale_data_for_cta2dwi ../data/gan/cta2dwi/experiment_data1/rigid_registration ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512`

- [x] 将数据resize到统一的分辨率，这里统一使用xy方向上的spacing作为基准, resize到512
    * 调用命令：`python cta_to_dwi_dataset.py rescale_data_for_cta2dwi ../data/gan/cta2dwi/experiment_data1/rigid_registration ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512 512`


- [x] 将数据resize到统一的分辨率，这里统一使用xy方向上的spacing作为基准, resize到256
    * 调用命令：`python cta_to_dwi_dataset.py rescale_data_for_cta2dwi ../data/gan/cta2dwi/experiment_data1/rigid_registration ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_256 256`

- [x] 提取脑实质部分的mask
    * 调用命令`python cta_to_dwi_dataset.py generate_cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale/cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_mask/cerebral_parenchyma *_brain_rigid_aligned.nii.gz`
    * 调用命令`python cta_to_dwi_dataset.py generate_cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512/cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512_mask/cerebral_parenchyma *_brain_rigid_aligned.nii.gz`
    * 调用命令`python cta_to_dwi_dataset.py generate_cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_256/cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_256_mask/cerebral_parenchyma *_brain_rigid_aligned.nii.gz`

- [ ] 根据脑实质的mask，将相应的CTA和DWI部分截取出来
    * 提取CTA，调用命令：`python cta_to_dwi_dataset.py extract_region_by_mask ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_mask/cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale/CTA ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_mask/CTA *brain*.nii.gz *CTA*.nii.gz`
    * 提取CTA 512分辨率，调用命令：`python cta_to_dwi_dataset.py extract_region_by_mask ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512_mask/cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512/CTA ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512_mask/CTA *brain*.nii.gz *CTA*.nii.gz`
    * 提取CTA 256分辨率，调用命令：`python cta_to_dwi_dataset.py extract_region_by_mask ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_256_mask/cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_256/CTA ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_256_mask/CTA *brain*.nii.gz *CTA*.nii.gz`
    * 提取DWI，调用命令：`python cta_to_dwi_dataset.py extract_region_by_mask ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_mask/cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale/DWI ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_mask/DWI *brain*.nii.gz *DWI*.nii.gz`
    * 提取DWI 512分辨率，调用命令：`python cta_to_dwi_dataset.py extract_region_by_mask ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512_mask/cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512/DWI ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512_mask/DWI *brain*.nii.gz *DWI*.nii.gz`
    * 提取DWI 256分辨率，调用命令：`python cta_to_dwi_dataset.py extract_region_by_mask ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_256_mask/cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_256/DWI ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_256_mask/DWI *brain*.nii.gz *DWI*.nii.gz`

- [ ] 根据脑实质mask，生成配置文件
    * 生成原始分辨率的配置文件，调用命令：`python cta_to_dwi_dataset.py genereate_cta2dwi_config_file_with_cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_mask ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_mask/config`
    * 生成512分辨率的配置文件，调用命令：`python cta_to_dwi_dataset.py genereate_cta2dwi_config_file_with_cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512_mask ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_512_mask/config`
    * 生成256分辨率的配置文件，调用命令：`python cta_to_dwi_dataset.py genereate_cta2dwi_config_file_with_cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_256_mask ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale_256_mask/config`
  
  ---
  利用脑实质mask，提取出去骨的脑部CT
  ---

配准之前：
- [x] 提取脑实质部分的mask
    * 调用命令`python cta_to_dwi_dataset.py generate_cerebral_parenchyma_dilation '../data/gan/cta2dwi/Atlas-crec-CTA-ASPECT/4 Patient_nii_unity' ../data/gan/cta2dwi/experiment_data1/rigid_registration_skull_stripper/cerebral_parenchyma *_brain.nii.gz`

- [ ] 根据脑实质的mask，将相应的CTA和DWI部分截取出来
    * 提取CTA，调用命令：`python cta_to_dwi_dataset.py extract_region_by_mask ../data/gan/cta2dwi/experiment_data1/rigid_registration_skull_stripper/cerebral_parenchyma '../data/gan/cta2dwi/Atlas-crec-CTA-ASPECT/4 Patient_nii_unity' ../data/gan/cta2dwi/experiment_data1/rigid_registration_skull_stripper/CTA *brain.nii.gz *CTA.nii.gz`

配准之后：
- [x] 提取脑实质部分的mask
    * 调用命令`python cta_to_dwi_dataset.py generate_cerebral_parenchyma_dilation ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale/cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_skull_stripper/cerebral_parenchyma *_brain_rigid_aligned.nii.gz`

- [ ] 根据脑实质的mask，将相应的CTA和DWI部分截取出来
    * 提取CTA，调用命令：`python cta_to_dwi_dataset.py extract_region_by_mask ../data/gan/cta2dwi/experiment_data1/rigid_registration_skull_stripper/cerebral_parenchyma ../data/gan/cta2dwi/experiment_data1/rigid_registration_rescale/CTA ../data/gan/cta2dwi/experiment_data1/rigid_registration_skull_stripper/CTA *brain*.nii.gz *CTA*.nii.gz`
