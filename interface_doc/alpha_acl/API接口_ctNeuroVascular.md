## Version:1.0
### 请求路径

### 返回码

### 子功能

#### 自动分割

1. 输入接口说明：

|名称|字段|类型|必须|说明|
|-|-|-|-|-|
|序列号|data/seriesUid|string|Y||
|序列路径|data/seriesPath|string|Y|待分割的脑血管序列的路径（请确保为脑部ct数据，该算法不提供内部检测）|
|混合电压序列路径|data/mixSeriesPath|string|Y||
|输出mask路径|data/seg_mask_path|string|Y|请确保输出路径存在，并保证有写入权限,输出格式：(nii.gz, nii, mha)|

2. 输入接口json示例

```
{
  "method": "neuro_vascular_coarse_segmentation",
  "ver": "1.0",
  "requestId": 44,
  "data":
  {
    "seriesPath" : "/data/zhangwd/data/examples/brain/bystudy/1.3.12.2.1107.5.1.4.60320.30000016072900171834200000026/1.3.12.2.1107.5.99.2.9594.30000016072913081812500000799",
    "mixSeriesPath" : "/data/zhangwd/data/examples/brain/bystudy/1.3.12.2.1107.5.1.4.60320.30000016072900171834200000026/1.3.12.2.1107.5.99.2.9594.30000016072913081812500000799",
    "seriesUid":"
/data/zhangwd/data/examples/brain/bystudy/1.3.12.2.1107.5.1.4.60320.30000016072900171834200000026/1.3.12.2.1107.5.99.2.9594.30000016072913081812500000799",
    "seg_mask_path":"/data/brain/t2.nii.gz"
  }
}
```

3. 输出接口说明：

|名称|字段|类型|必须|说明|
|-|-|-|-|-|
|序列号|data/seriesUid|string|Y||
|是否成功|data/isSucceed|bool|Y|分割程序是否顺利运行|


4. 输出接口json示例

```
{
    "requestId": 44,
    "code": 0,
    "beginTime": 1582789567,
    "endTime": 1582789602,
    "msg": "",
    "data": {
        "seriesUid": "\r\n/data/zhangwd/data/examples/brain/bystudy/1.3.12.2.1107.5.1.4.60320.30000016072900171834200000026/1.3.12.2.1107.5.99.2.9594.30000016072913081812500000799",
        "isSucceed": true
    }
}
```


#### 单点区域增长

1. 输入接口说明：

|名称|字段|类型|必须|说明|
|-|-|-|-|-|
|序列号|data/seriesUid|string|Y||
|序列路径|data/seriesPath|string|Y|待分割的脑血管序列的路径（请确保为脑部ct数据，该算法不提供内部检测）|
|混合电压序列路径|data/mixSeriesPath|string|Y||
|输出mask路径|data/seg_mask_path|string|Y|请确保输出路径存在，并保证有写入权限，输出格式：(nii.gz, nii, mha)|
|种子点|data/iSeed|int[3]|Y||

2. 输入接口json示例

```
{
  "method": "neuro_vascular_single_point_region_grow",
  "ver": "1.0",
  "requestId": 44,
  "data":
  {
    "seriesPath" : "/data/zhangwd/data/examples/brain/bystudy/1.3.12.2.1107.5.1.4.60320.30000016072900171834200000026/1.3.12.2.1107.5.99.2.9594.30000016072913081812500000799",
    "mixSeriesPath" : "/data/zhangwd/data/examples/brain/bystudy/1.3.12.2.1107.5.1.4.60320.30000016072900171834200000026/1.3.12.2.1107.5.99.2.9594.30000016072913081812500000799",
    "seriesUid":"
/data/zhangwd/data/examples/brain/bystudy/1.3.12.2.1107.5.1.4.60320.30000016072900171834200000026/1.3.12.2.1107.5.99.2.9594.30000016072913081812500000799",
    "seg_mask_path":"/data/brain/tt2.nii.gz",
	"iSeed":[195,333,250]
  }
}
```


3. 输出接口说明：

|名称|字段|类型|必须|说明|
|-|-|-|-|-|
|序列号|data/seriesUid|string|Y||
|是否成功|data/isSucceed|bool|Y|分割程序是否顺利运行|
|是否成功|data/growedVessel|vector<int>|Y|分割结果，输出索引位置数组|


4. 输出接口json示例

```
{
    "requestId": 44,
    "code": 0,
    "beginTime": 1582789567,
    "endTime": 1582789602,
    "msg": "",
    "data": {
        "seriesUid": "\r\n/data/zhangwd/data/examples/brain/bystudy/1.3.12.2.1107.5.1.4.60320.30000016072900171834200000026/1.3.12.2.1107.5.99.2.9594.30000016072913081812500000799",
        "isSucceed": true
    }
}
```


#### 修改前后mask度量——轮廓线周长计算

1. 输入接口说明：

|名称|字段|类型|必须|说明|
|-|-|-|-|-|
|初始mask路径|data/initMask|string|Y||
|修改后的mask路径|data/modifiedMask|string|Y||
|初始mask的label值|data/initLabels|int[]|Y||
|输出mask的label值|data/modifiedLabels|int[]|Y||

2. 输入接口json示例

```
{
  "method": "calculate_mask_perimeter",
  "ver": "1.0",
  "requestId": 44,
  "data":
  {
    "initMask" : "/data/zhangwd/data/examples/MASKfeikuaceng1.nii",
    "modifiedMask":"/data/zhangwd/data/examples/MASKfeikuaceng2.nii",
    "initLabels":[1],
	"modifiedLabels":[1]
  }
}
```


3. 输出接口说明：

|名称|字段|类型|必须|说明|
|-|-|-|-|-|
|周长|data/peremeter|double|Y||



4. 输出接口json示例
```
{
    "requestId": 44,
    "code": 0,
    "beginTime": 1583744015,
    "endTime": 1583744022,
    "msg": "",
    "data": {
        "peremeter": 36.0000000
    }
}
```

### 返回码说明

返回码：
```
	OK = 0,
	CodeParamMissing = 1901,
	CodeSeriesPathNotExist = 1902,
	CodeParamIllegal = 1903,
	CodeUnknown = 1904
```