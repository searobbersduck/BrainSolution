## 数据处理
1. 处理WD_fir数据：
   1. `python aneurysm_utils.py test_generate_block_pairs_multiprocessing`, 处理结果存放于`../data/source_img/block_pairs/WD_fir_coord`, 原始配置文件位于`../data/source_img/csvs/WD_fir.csv`, ***注意配置文件中的z坐标需要根据数据的direction进行调整***；
   2. 数据处理后的配置文件位于：`../data/source_img/block_pairs/WD_fir_coord/train/config.txt`
   3. `config.txt`中的每项如`1.2.840.113619.2.416.179073062726235549312612285271935335680_284.1_191.9_481.0_0.npy    0`所示，记录了每个块的是否有动脉瘤以及动脉瘤相对于该块的位置；
   4. 为了能够利用看图软件查看切块是否正确，调用`test_extract_block_with_aneurysm`,结果存于`../data/source_img/block_pairs/WD_fir_coord/train_visualize`
   5. 现在使用的一款看图小工具只支持`.mhd`格式，可以通过如下代码转换`python aneurysm_utils.py convert_image_format '../data/source_img/block_pairs/WD_fir_coord/train_visualize/1.2.840.113619.2.334.3.2831179063.409.1509096762.87_115.3_109.1_19.0_402.nii.gz' '.nii.gz' '.mhd'`
   6. 在WD_fir val上的测试效果：
    ```
    ====> test accuracy is 0.963
    ===> Confusion Matrix:
    [[2735   96]
    [  12   37]]
    ====> end to test!
    ``` 
    7. 在XY_fir train上的测试效果：
    ```
    ====> test accuracy is 0.952
    ===> Confusion Matrix:
    [[45236  2161]
    [  124   223]]
    ====> end to test!

    ```
     8. 在XY_fir val上的测试效果：
    ```
    ====> test accuracy is 0.945
    ===> Confusion Matrix:
    [[5072  280]
    [  17    7]]
    ====> end to test!
    ```

2. 处理XY_fir数据：
   1. 数据的原始配置文件与WD_fir一致；
   2. 调用命令`python aneurysm_utils.py XY_fir_generate_block_pairs_multiprocessing`

3. 处理XY_sec数据：
   1. 数据的原始配置文件与WD_sec一致；
   2. 调用命令`python aneurysm_utils.py XY_sec_generate_block_pairs_multiprocessing`


## 分割数据处理

1. 处理WD_fir数据：
   1. `python aneurysm_utils.py WD_fir_generate_aneurysm_mask_multiprocessing`, 处理结果存放于`../data/source_img/seg/WD_fir`, 原始配置文件位于`../data/source_img/csvs/WD_fir.csv`, ***注意配置文件中的z坐标需要根据数据的direction进行调整***；