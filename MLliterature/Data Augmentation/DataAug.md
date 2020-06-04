#  2020 ICLR- AUGMIX: "A SIMPLE DATA PROCESSING METHOD TO IMPROVE ROBUSTNESS AND UNCERTAINTY"

##  Overview

本文主要想解决的是training data and testing data的分布不同的问题。 一般的数据增广方法倾向于让模型去记住这些增加的数据，而不是真的学习到里面的规律。 而且很多的data augmentation 方法 会有 clean accuracy , computational compleixty, robustness 与 uncertainty estimation之间的trade-off吧。



然后相比于连续的做一系列变换引起变化后的图像与原始图像差距太大并且多样性不够， 所以采用随机选取一些transform operation 来解决这问题。

>**Flow**
>
>1. 从Dirichlet分布采样权值每一步变化的权重$w_i$
>2. 从transform pool $\mathcal O$   随机采样operation $O$
>3. 第k步的图像 $x_k=x_{k-1}+w_k\times O(x_{orig})$
>4. 循环K步后再和原始图片插值 $x_{aug}=m.x_{orig}+(1-m)x_K$ , 其中权重m从beta分布采样
>
>









#  2019 ICCV  -Cutmix: "Regularization strategy to train strong classifiers with localizable features”

## Overview

随机把另一个图像的patch 覆盖的当前图像。主要解决随机擦除方法里 单幅图片的信息量是减少的问题。



# 2019 Patch Gaussian augments  "Improving robustness without sacrificing accuracy with patch Gaussian augmentation”

## Overview

随机对某个patch data添加噪声

## Motivation

当对整个图片增加随机噪声时候， clean accuracy 会下降， robustness会增加。 random cutout 会增加clean accuracy, 但是会降低随机鲁棒性。 所以把它们结合起来。



# 2017 ICLR - Mixup "mixup: Beyond empirical   risk minimization”

## Overview

随机选取两个图像 并按比例插值



# 2017 Random  Cutout "Improved regularization of convolutional neural networks with Cutout".

## Overview

随机擦除一个patch 图像。 原理可以解释成如果一个让模型通过图片的不同部分，而不是仅仅局限在某个最显著特征（比如head）来做出决策。 然后在cutmix里指出，这样的缺陷是 每幅图片的信息量是减少的


