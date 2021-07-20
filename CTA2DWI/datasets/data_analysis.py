import os
import sys

import numpy as np
from glob import glob
from tqdm import tqdm
import pandas as pd

import fire

CTA2DWI_ROOT = os.path.join(os.path.dirname(__file__), os.path.pardir)
sys.path.append(CTA2DWI_ROOT)

from external_lib.MedCommon.utils.dicom_tag_utils import DicomTagUtils

'''
    $BrainSolution/CTA2DWI/data/gan/hospital_6_multi/CTA2DWI-多中心-20201102

    tree -L 2
    .
    ├── CTA阴性（108例）
    │   ├── 六院-CTA阴性（69）
    │   ├── 南通大学-阴性-血管(14)
    │   └── 闵中心-阴性-血管（25）
    └── 阳性-闭塞(188例）
        ├── 六院-DWI闭塞病例(105)
        ├── 六院-阳性-血管闭塞（25）
        ├── 六院-阳性-血管闭塞（37）
        ├── 南通大学-阳性-血管闭塞（5）
        └── 闵中心-阳性-血管闭塞（16）


    tree -L 3
    .
    ├── CTA阴性（108例）
    │   ├── 六院-CTA阴性（69）
    │   │   ├── 1124013
    │   │   ├── 1140092
    │   │   ├── 1195207
    │   │   ├── 1399063
    │   │   ├── 1424031
    │   │   ├── 1534457
    │   │   ├── 1870593
    │   │   ├── 1944927
    │   │   ├── 2030869
    │   │   ├── 2052446
    │   │   ├── 2059698
    │   │   ├── 2202873
    │   │   ├── 2262510
    │   │   ├── 2375643
    │   │   ├── 2432960
    │   │   ├── 2555294
    │   │   ├── 2589844
    │   │   ├── 2659165
    │   │   ├── 2827839
    │   │   ├── 3247448
    │   │   ├── 3461840
    │   │   ├── 3670506
    │   │   ├── 3780357
    │   │   ├── 3783684
    │   │   ├── 3849168
    │   │   ├── 4032540
    │   │   ├── 4253761
    │   │   ├── 4360408
    │   │   ├── 4449513
    │   │   ├── 4562734
    │   │   ├── 4562774
    │   │   ├── 4991624
    │   │   ├── 4997258
    │   │   ├── 4997700
    │   │   ├── 5000919
    │   │   ├── 5001429
    │   │   ├── 5004860
    │   │   ├── 5005636
    │   │   ├── 5022525
    │   │   ├── 5023998
    │   │   ├── 5037960
    │   │   ├── 5043789
    │   │   ├── 5055446
    │   │   ├── 5056709
    │   │   ├── 5056770
    │   │   ├── 5058950
    │   │   ├── 5061619
    │   │   ├── 5061653
    │   │   ├── 5063234
    │   │   ├── 5064024
    │   │   ├── 5065086
    │   │   ├── 5077804
    │   │   ├── 5086573
    │   │   ├── 5086941
    │   │   ├── 5104076
    │   │   ├── 5104855
    │   │   ├── 5108293
    │   │   ├── 5110542
    │   │   ├── 5110660
    │   │   ├── 5111489
    │   │   ├── 5111554
    │   │   ├── 5112438
    │   │   ├── 5117938
    │   │   ├── 5121258
    │   │   ├── 5122603
    │   │   ├── 5133546
    │   │   ├── 5150123
    │   │   ├── 5151483
    │   │   └── 5153387
    │   ├── 南通大学-阴性-血管(14)
    │   │   ├── 1717906
    │   │   ├── 1741090
    │   │   ├── 1747371
    │   │   ├── 1750639
    │   │   ├── 1790350
    │   │   ├── 1822436
    │   │   ├── 1830948
    │   │   ├── 1934654
    │   │   ├── 1952707
    │   │   ├── 2078915
    │   │   ├── 2104557
    │   │   ├── 2110080
    │   │   ├── 2123279
    │   │   └── 2124256
    │   └── 闵中心-阴性-血管（25）
    │       ├── 102812640
    │       ├── 102850872
    │       ├── 102871629
    │       ├── CAO HONGLIN
    │       ├── CHEN YA
    │       ├── GONG CHUNLONG
    │       ├── GONG HONGLAN
    │       ├── HANG FAJUN
    │       ├── HU GENHUA
    │       ├── JIAO HONGMAO
    │       ├── LI BIYAO
    │       ├── LU HUANYI
    │       ├── SHEN QIANRU
    │       ├── SHEN YUZHEN
    │       ├── SONG HAIRONG
    │       ├── SUN HONGWEI
    │       ├── SUN ZHANWU
    │       ├── WANG KEZHAN
    │       ├── XIE GUIZHEN
    │       ├── XU XIANGYUN
    │       ├── XU XINGQIN
    │       ├── YE ZHANGLIU
    │       ├── ZHANG QUFEI
    │       ├── ZHENG YI
    │       └── ZHOU GUOHUA
    └── 阳性-闭塞(188例）
        ├── 六院-DWI闭塞病例(105)
        │   ├── 1014186
        │   ├── 1029629
        │   ├── 1035948
        │   ├── 1074375
        │   ├── 1231754
        │   ├── 1358935
        │   ├── 1445543
        │   ├── 1502032
        │   ├── 1616503
        │   ├── 1686783
        │   ├── 1703560
        │   ├── 1712912
        │   ├── 1902661
        │   ├── 1935168
        │   ├── 2001558
        │   ├── 2006878
        │   ├── 2083121
        │   ├── 2094368
        │   ├── 2182657
        │   ├── 2280689
        │   ├── 2389792
        │   ├── 2452420
        │   ├── 2454141
        │   ├── 2602401
        │   ├── 2622185
        │   ├── 2639090
        │   ├── 2670896
        │   ├── 2942930
        │   ├── 2954952
        │   ├── 2998848
        │   ├── 3094942
        │   ├── 3220683
        │   ├── 3331290
        │   ├── 3356522
        │   ├── 3440192
        │   ├── 3466686
        │   ├── 3617570
        │   ├── 3624765
        │   ├── 3728208
        │   ├── 3736505
        │   ├── 3784996
        │   ├── 3869885
        │   ├── 3874653
        │   ├── 3878650
        │   ├── 3901698
        │   ├── 3902268
        │   ├── 3924933
        │   ├── 3926192
        │   ├── 3942366
        │   ├── 3959073
        │   ├── 3987577
        │   ├── 4014155
        │   ├── 4023857
        │   ├── 4062211
        │   ├── 4068352
        │   ├── 4079485
        │   ├── 4118402
        │   ├── 4119126
        │   ├── 4140286
        │   ├── 4157025
        │   ├── 4185501
        │   ├── 4196642
        │   ├── 4203734
        │   ├── 4238407
        │   ├── 4246235
        │   ├── 4260454
        │   ├── 4285331
        │   ├── 4291274
        │   ├── 4298912
        │   ├── 4299369
        │   ├── 4304964
        │   ├── 4366985
        │   ├── 4381668
        │   ├── 4386552
        │   ├── 4402594
        │   ├── 4407082
        │   ├── 4440733
        │   ├── 4451397
        │   ├── 4455178
        │   ├── 4457593
        │   ├── 4465419
        │   ├── 4479986
        │   ├── 4490737
        │   ├── 4503839
        │   ├── 4504615
        │   ├── 4527569
        │   ├── 4531560
        │   ├── 4552561
        │   ├── 4574956
        │   ├── 4581095
        │   ├── 4582240
        │   ├── 4597717
        │   ├── 4600198
        │   ├── 4605196
        │   ├── 4624787
        │   ├── 4634411
        │   ├── 4646244
        │   ├── 4669297
        │   ├── 4677636
        │   ├── 4692992
        │   ├── 4730391
        │   ├── 4765366
        │   ├── 4781742
        │   ├── 4804837
        │   └── 4835944
        ├── 六院-阳性-血管闭塞（25）
        │   ├── 1094560
        │   ├── 1430993
        │   ├── 1650297
        │   ├── 1710336
        │   ├── 1894930
        │   ├── 2264829
        │   ├── 2349661
        │   ├── 2365327
        │   ├── 2811875
        │   ├── 2966561
        │   ├── 4198311
        │   ├── 4268802
        │   ├── 5023628
        │   ├── 5023941
        │   ├── 5026848
        │   ├── 5043836
        │   ├── 5046928
        │   ├── 5087053
        │   ├── 5109460
        │   ├── 5115015
        │   ├── 5117999
        │   ├── 5120184
        │   ├── 5126214
        │   ├── 5135202
        │   └── 5153326
        ├── 六院-阳性-血管闭塞（37）
        │   ├── 1180002
        │   ├── 1237062
        │   ├── 1306107
        │   ├── 1792160
        │   ├── 1959911
        │   ├── 2278692
        │   ├── 2459328
        │   ├── 2693475
        │   ├── 2835569
        │   ├── 2954159
        │   ├── 2999343
        │   ├── 3101695
        │   ├── 3129838
        │   ├── 3906343
        │   ├── 4356296
        │   ├── 4590771
        │   ├── 4641052
        │   ├── 4715940
        │   ├── 4797646
        │   ├── 4848080
        │   ├── 4853010
        │   ├── 4873022
        │   ├── 4890509
        │   ├── 4962062
        │   ├── 4972753
        │   ├── 4979526
        │   ├── 4981609
        │   ├── 4984637
        │   ├── 4999020
        │   ├── 5001653
        │   ├── 5006173
        │   ├── 5014734
        │   ├── 5014850
        │   ├── 5016897
        │   ├── 5023571
        │   ├── 5023628
        │   └── 5024308
        ├── 南通大学-阳性-血管闭塞（5）
        │   ├── 1464395
        │   ├── 1718144
        │   ├── 2085089
        │   ├── 2091458
        │   └── 2107569
        └── 闵中心-阳性-血管闭塞（16）
            ├── 101878640
            ├── 102512839-101477685
            ├── 102661445
            ├── 102869917
            ├── 102987728
            ├── FENG DEXING
            ├── JIN ZHANGGEN
            ├── LIANG YONGSHENG
            ├── QIAN YONGZHEN
            ├── SHEN DESHENG
            ├── YE WEI
            ├── YOU XIUYING
            ├── YU GENYE
            ├── ZHANG HUOZHEN
            ├── ZHANG MEIHUA
            └── ZHOU CHUNYOU

'''


# 将多中心的数据统一copy到统一目录下，目录的构成如下

def copy_multi_centers_data_to_one_folder(multi_center_root, dst_root):
    '''
    multi_center_root: '../data/gan/hospital_6_multi/CTA2DWI-多中心-20201102/阳性-闭塞(188例）/六院-DWI闭塞病例(105)'
    multi_center_root: '../data/gan/hospital_6_multi/CTA2DWI-多中心-20201102/阳性-闭塞(188例）/六院-阳性-血管闭塞（37）'
    multi_center_root: '../data/gan/hospital_6_multi/CTA2DWI-多中心-20201102/阳性-闭塞(188例）/六院-阳性-血管闭塞（25）'
    '''
    pass

def cta2dwi_extract_patient_info(data_root, outdir):
    '''
    data_root: '../data/gan/hospital_6_multi/CTA2DWI-多中心-20201102/阳性-闭塞(188例）/六院-DWI闭塞病例(105)'
    data_root: '../data/gan/hospital_6_multi/CTA2DWI-多中心-20201102/阳性-闭塞(188例）/六院-阳性-血管闭塞（37）'
    data_root: '../data/gan/hospital_6_multi/CTA2DWI-多中心-20201102/阳性-闭塞(188例）/六院-阳性-血管闭塞（25）'

    outdir: '../data/gan/hospital_6_multi'

    debug cmd: cta2dwi_extract_patient_info('../data/gan/hospital_6_multi/CTA2DWI-多中心-20201102/阳性-闭塞(188例）/六院-DWI闭塞病例(105)', '../data/gan/hospital_6_multi/config')
    debug cmd: cta2dwi_extract_patient_info('../data/gan/hospital_6_multi/CTA2DWI-多中心-20201102/阳性-闭塞(188例）/六院-阳性-血管闭塞（37）', '../data/gan/hospital_6_multi/config')
    debug cmd: cta2dwi_extract_patient_info('../data/gan/hospital_6_multi/CTA2DWI-多中心-20201102/阳性-闭塞(188例）/六院-阳性-血管闭塞（25）', '../data/gan/hospital_6_multi/config')

    debug cmd: cta2dwi_extract_patient_info('../data/gan/hospital_6/ori_neg', '../data/gan/hospital_6_multi/config')
    ''' 
    os.makedirs(outdir, exist_ok=True)
    basename = os.path.basename(data_root)
    outfile = os.path.join(outdir, '{}.csv'.format(basename))
    patient_ids = os.listdir(data_root)

    except_list = []
    row_elems = []
    column_names = ['pid', 'cta_series_instance_uid', 'dwi_series_instance_uid', 'cta_age', 'cta_sex', 'cta_acq_time', 'dwi_acq_time']
    for pid in tqdm(patient_ids):
        try:
            patient_path = os.path.join(data_root, pid)
            cta_path = os.path.join(patient_path, 'CTA')
            if not os.path.isdir(cta_path):
                cta_path = os.path.join(patient_path, 'CTA1')
            dwi_path = os.path.join(patient_path, 'DWI')
            if not os.path.isdir(dwi_path):
                dwi_path = os.path.join(patient_path, 'MR')

            cta_meta = DicomTagUtils.load_metadata(cta_path, is_series=True)
            cta_info = DicomTagUtils.get_basic_info(cta_meta)

            dwi_meta = DicomTagUtils.load_metadata(dwi_path, is_series=True)
            dwi_info = DicomTagUtils.get_basic_info(dwi_meta)

            patient_info = [pid, cta_info['series_uid'], dwi_info['series_uid'], cta_info['age'], cta_info['sex'], str(cta_info['acq_time']), str(dwi_info['acq_time'])]
            row_elems.append(np.array(patient_info))

        except:
            except_list.append(pid)
    
    df = pd.DataFrame(np.array(row_elems), columns=column_names)
    df.to_csv(outfile)
    print(except_list)



if __name__ == '__main__':
    # cta2dwi_extract_patient_info('../data/gan/hospital_6_multi/CTA2DWI-多中心-20201102/阳性-闭塞(188例）/六院-DWI闭塞病例(105)', '../data/gan/hospital_6_multi/config')
    cta2dwi_extract_patient_info('../data/gan/hospital_6/ori_neg', '../data/gan/hospital_6_multi/config')