# AI

## 学习理论

[https://github.com/guxiaozhe/NOTE/blob/master/ML/%E5%AD%A6%E4%B9%A0%E7%90%86%E8%AE%BA/BASIC.md](https://github.com/guxiaozhe/NOTE/blob/master/ML/学习理论/BASIC.md)



##数据增强

 [https://github.com/guxiaozhe/NOTE/blob/master/MLliterature/Data%20Augmentation/DataAug.md](https://github.com/guxiaozhe/NOTE/blob/master/MLliterature/Data Augmentation/DataAug.md)

#  Information



熵 https://github.com/guxiaozhe/NOTE/blob/master/Information/Info.md

























# Cifar 10 Experiment Result

##90 epoches

aug_alpha=1

KD alpha=0.5

KD T=5

###学生：Resnet 8

|          | NOKD  | KD   |      |      |      |
| :------: | ----- | ---- | ---- | ---- | ---- |
|  NOAUG   | 88.79 |      |      |      |      |
|  MIXUP   | 86.69 |      |      |      |      |
|  CUTMIX  | 85.97 |      |      |      |      |
| CUTMIXKD | 86.46 |      |      |      |      |

 

###学生：Resnet 18

|          | NOKD  | KD   |      |      |      |
| :------: | ----- | ---- | ---- | ---- | ---- |
|  NOAUG   | 94.24 |      |      |      |      |
|  MIXUP   | 94.93 |      |      |      |      |
|  CUTMIX  | 95.20 |      |      |      |      |
| CUTMIXKD | 95.22 |      |      |      |      |

 

## 200 epoches



aug_alpha=1

KD alpha=0.5

KD T=5

###学生：Resnet 8

|          | NOKD  | KD    |      |      |      |
| :------: | ----- | ----- | ---- | ---- | ---- |
|  NOAUG   |       | 90.69 |      |      |      |
|  MIXUP   |       | 90.17 |      |      |      |
|  CUTMIX  |       | 89.96 |      |      |      |
| CUTMIXKD | 87.67 | 89.99 |      |      |      |

 

###学生：Resnet 18

|          | NOKD  | KD   |      |      |      |
| :------: | ----- | ---- | ---- | ---- | ---- |
|  NOAUG   | 95.26 |      |      |      |      |
|  MIXUP   | 95.74 |      |      |      |      |
|  CUTMIX  | 96.05 |      |      |      |      |
| CUTMIXKD | 95.95 |      |      |      |      |

 



aug_alpha=0.5

KD alpha=0.5

KD T=5

###学生：Resnet 8

|          | NOKD  | KD   |      |      |      |
| :------: | ----- | ---- | ---- | ---- | ---- |
|  NOAUG   |       |      |      |      |      |
|  MIXUP   | 89.73 |      |      |      |      |
|  CUTMIX  | 89.77 |      |      |      |      |
| CUTMIXKD | 90.17 |      |      |      |      |



###  学生：Resnet 20

|          | NOKD  | KD   |      |      |      |
| :------: | ----- | ---- | ---- | ---- | ---- |
|  NOAUG   | 92.41 |      |      |      |      |
|  MIXUP   | 92.63 |      |      |      |      |
|  CUTMIX  | 92.59 |      |      |      |      |
| CUTMIXKD | 93.04 |      |      |      |      |

###  学生：Resnet 18

|          | NOKD  | KD   |      |      |      |
| :------: | ----- | ---- | ---- | ---- | ---- |
|  NOAUG   | 95.03 |      |      |      |      |
|  MIXUP   |       |      |      |      |      |
|  CUTMIX  | 95.86 |      |      |      |      |
| CUTMIXKD | 95.74 |      |      |      |      |



###  学生：Resnet 26

|          | NOKD  | KD   |      |      |      |
| :------: | ----- | ---- | ---- | ---- | ---- |
|  NOAUG   |       |      |      |      |      |
|  MIXUP   |       |      |      |      |      |
|  CUTMIX  | 93.41 |      |      |      |      |
| CUTMIXKD | 93.21 |      |      |      |      |

