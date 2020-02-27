## 2020.2.7
- [ ] 跑通心脏血管增强UT
- [x] 尝试心脏冠脉标注工具，并记录
- [x] 跟进ct部位识别，进度和效果
  * 把层数比较大的文件插值成200张，不改变spacing（应该改变spacing），胸腹连扫数据的准确率只有50%，该方式不通过
- [ ] 实现itk分割的python code
  * 利用种子点生成分割mask，需要手动标注，太麻烦，暂不使用
  * 利用阈值生成分割mask
- [x] 跟进dr相关项目
  * 帮助测试跑通肺结核train code
  * 帮助测试跑通predict code


## 2020.2.10
- [x] 通过alpha的medialness接口，处理生成脑部ct数据
  * 经查看，无法直接利用该数据提取脑部血管mask，需要通过其它方式
- [x] 脑ct在去骨数据的基础上，只做阈值操作
  * 无法生成比较好的血管mask
- [x] 协助跑通dr predict

## 2020.2.11


## 2020.2.12
1. 在medialness滤波基础上，做threshold+ConnectedComponent操作，结果存于`/home/zhangwd/code/work/BrainSolution/ct_dual_energy_substraction/itk_algo/connected1/`
	* 目测效果不佳
2. 在原图像基础上，根据窗宽窗位，做threshold+ConnectedComponent操作，结果存于`/home/zhangwd/code/work/BrainSolution/ct_dual_energy_substraction/itk_algo/connected/`
	* 目测效果不佳
3. 在原图像基础上，根据窗宽窗位，只做threshold操作，结果存于`/home/zhangwd/code/work/BrainSolution/ct_dual_energy_substraction/itk_algo/threshold/`
	* 目测结果比上两种效果好，个人认为可以在此基础上，进行其它操作


## 2020.2.13
1. 尝试在如下操作：
	1. 未去骨图像与去骨图像做差值，提取出头骨图像
	2. 做阈值二值化后，腐蚀膨胀提取出头骨二值化图像（这一部分膨胀有比腐蚀更大的力度）
	3. 在原图像基础上，根据窗宽窗位，只做阈值操作（腐蚀膨胀，kernel=1），并去掉与头骨有交集的部分
目前还在测试中。


## 2020.2.14
1. 复现冠脉区域增长的效果
   1. 查找冠脉数据保存到``
   2. 利用1中数据生成相应的medialness结果， 保存到``
   3. 构建1中数据相应的初始化mask文件
   4. 复现UT:`ctPiplineRegionGrowTest`的结果