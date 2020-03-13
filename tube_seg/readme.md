# 脑血管分割

## 初始数据集生成
### 源数据（头颈部）
1. 剪影数据
2. 相对应的混合电压数据
3. 高电压数据
4. 低电压数据

### 处理数据
#### 1. 预处理剪影数据
1. 将剪影数据生成`mhd`格式，并生成相应的初分割脚本：`../tube_seg/notebook/0.henan_data_module_generate_mhd.ipynb`
2. 利用[smistad/Tube-Segmentation-Framework](https://github.com/smistad/Tube-Segmentation-Framework), 对数据进行处理，生成相应的`mhd`文件, 在编译好的code的路径下，执行1生成的脚本，注意脚本中的数据路径及输出结果路径；
3. 将2生成的数据，处理生成nii格式，以便查看；参照：`tube_seg/notebook/0.henan_data_module_convert_mhd_to_nii.ipynb`


#### 2. 预处理混合电压数据（同1）
1. 将剪影数据生成`mhd`格式，并生成相应的初分割脚本：`../tube_seg/notebook/0.henan_data_module_generate_mix_mhd.ipynb`
2. 利用[smistad/Tube-Segmentation-Framework](https://github.com/smistad/Tube-Segmentation-Framework), 对数据进行处理，生成相应的`mhd`文件, 在编译好的code的路径下，执行1生成的脚本，注意脚本中的数据路径及输出结果路径；
3. 将2生成的数据，处理生成nii格式，以便查看; 参照：`tube_seg/notebook/0.henan_data_module_convert_mhd_to_nii.ipynb`


