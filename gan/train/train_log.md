### 2020.4.2

#### cta转dwi

调用命令：`python train_tmp_20200402.py`
参数：
```
class Options():
    def __init__(self):
        self.lr = 1e-3
        self.beta1 = 0.5
        self.gan_mode = 'lsgan'
        self.direction = 'AtoB'
        self.lambda_L1 = 2
        self.epochs = 1000
        self.num_workers = 8
        self.batch_size = 3
        self.pin_memory = True
        self.display = 20
        self.save_interval = 10
        self.intermidiate_result_root = '../data/gan/cta2dwi/case_178_forHuang_rescale/intermidiate_result_{}'.format(os.path.basename(os.getcwd()).split('.')[0])
        # add patch discriminator
        self.patch_D = True
        self.num_patches_D = 5
        self.patch_size_D = [32, 64, 64]
        # crop_size
        self.crop_size = [160, 224, 288]
```
本次修改：
1. 添加：patch discriminator
    * 相应添加参数：
    ```
        self.patch_D = True
        self.num_patches_D = 5
        self.patch_size_D = [64, 64, 64]
    ```
2. 添加：每100epoch，保存所有生成图
3. 修改参数：`self.lambda_L1 = 100`修改为`self.lambda_L1 = 2`

结果：
1. 临时需要重新分配GPU，试验终止
2. 试验进行到700+，效果比不加patch discriminator要好，但整体效果依然不佳
3. 试验结果位于`../data/gan/cta2dwi/case_178_forHuang_rescale/intermidiate_result_train_tmp_20200402`, 对比结果位于`../data/gan/cta2dwi/case_178_forHuang_rescale/intermidiate_result1`, 此处对比的时`700+epoch`和`900+epoch`的结果；

### 2020.4.3

#### ncct转dwi
调用命令：`CUDA_VISIBLE_DEVICES=6,7 python train_tmp_20200403.py|tee train_tmp_20200403.log`

参数：
```
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
        self.display = 20
        self.save_interval = 10
        self.intermidiate_result_root = '../data/gan/ncct2dwi/siyuan_NCCT-DWI-align/dwi_rigid_align_ncct_rescale/train_result/intermidiate_result_{}'.format(__file__.split('.')[0])
        # add patch discriminator
        self.patch_D = False
        self.num_patches_D = 5
        self.patch_size_D = [64, 64, 64]
        # crop_size
        self.crop_size = [160, 224, 288]
```
本次修改：
1. 模型数据由cta2dwi,变为ncct2dwi数据
2. 参数调整
    ```
    self.lr = 2e-4
    self.display = 10
    ```

2次修改：
1. 发现三例问题数据：475170、372829、463311，此处在训练集中去掉这3例数据；后续需要对配准算法进行优化





结果：
显示效果不佳，
### 2020.4.6

#### cta生成dwi

调用命令：`CUDA_VISIBLE_DEVICES=3,4,5 python train_tmp_20200406.py | tee train_tmp_20200406.log`

参数：
```
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
        self.display = 10
        self.save_interval = 10
        self.intermidiate_result_root = '../data/gan/cta2dwi/case_178_forHuang_rescale/train_result/intermidiate_result_{}'.format(__file__.split('.')[0])
        # add patch discriminator
        self.patch_D = False
        self.num_patches_D = 5
        self.patch_size_D = [64, 64, 64]
        # crop_size
        self.crop_size = [160, 224, 288]
```

本次修改：
相对于上次训练：
1. batch_size：3变为2
2. lr:1e-3变为2e-4

结果：
临时需要重新分配GPU，试验终止

---
---
2020.4.7
---

#### cta2dwi
试验目的：验证单batch_size是否有比多batch更好的效果

参考结果：

调用命令：`CUDA_VISIBLE_DEVICES=0 python train_tmp_20200407_cta2dwi_3d.py | tee train_tmp_20200407_cta2dwi_3d.log`