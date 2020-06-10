#  分析



## 2020 CVPR:    REVISIT KNOWLEDGE DISTILLATION: A TEACHERFREE FRAMEWORK

### Overview

主要是通过发现 1） student and teacher reverse training可以提高performance 以及2）a poorly trained teacher 可以提高student performancce 得出结论： 1) KD是一种可学习的label smoothing regularization    2 ) LS 提供了一个虚拟老师。 因而完全可以找到一个适合的 virtual teacher with zero similarity information  来提高model performance.

论文提出了两个方法

1. self-training 和born again 类似 
2. 手动设计一个targets distribution 



###  KD 以及 LR

* $p(k)$ : 样本x的category k 的输出
* $q(k)$ :  group truth

**原始的损失函数**

>$$
>H(q,p)=-\sum_{k}^K q(k)\log p(k)
>$$
>
>



**LS 损失函数**

>$$
>q'(k)=q(k)\times (1-\alpha)+\alpha \mu(k)\\\\
>通常\mu(k)=1/K\\\\
>\Rightarrow H(q',p)=-(1-\alpha)\sum_k^K  q(k)\log p(k)- \alpha\sum_k^K \mu(k)\log p(k)\\\\
>=(1-\alpha)H(q,p)+\alpha(D_{KL}(\mu||p)+H(\mu))\\\\
>=(1-\alpha)H(q,p)+\alpha D_{KL}(\mu||p)\\\\=(1-\alpha)H(q,p)+\alpha H(\mu,p)
>$$

**KD损失函数**

>$$
>\mathcal L_{KD}=(1-\alpha) H(q,p)+\alpha D_{KL}(p^T_{\tau}||p^S_{\tau})\\\\
>=(1-\alpha) H(q,p)+\alpha( \alpha D_{KL}(p^T_{\tau}||p^S_{\tau})+H(p^T_{\tau}))-H(p_{\tau}^T)\\\\
>(1-\alpha) H(q,p)+ \alpha H(p^T_{\tau},p^S_{\tau})
>$$
>
>

如果$\tau=1$, 那么 KD 相对于 LS with
$$
\mu(k)=p^T(k)\Rightarrow \\\\
q'(k)=(1-\alpha)q(k)+\alpha p^T(k)
$$
特别的，当温度$\tau$ 很高， 则teacher output相当于更接近LS的uniform distribution.



LS让output 的distribution 除了符合label外， 同时让output distribution 更uniform and soft,

而KD是让output distribution 更soft同时，更接近teacher （teacher output本身比较soft）。 





### Virtual Teacher

$$
p^d(k)=\begin{cases}a\mbox{ if }k=c\\\\
(1-a)/(K-1) \alpha\mbox{ otherwise }
\\ \end{cases}\\\\
LS: q'(k)=(1-\alpha)q(k)+\alpha/K\\\\
VKD: q'(k)=\begin{cases}(1-\alpha)q(k)+\alpha a \mbox{ if }k=c\\\\
(1-\alpha)q(k)+(1-a)/(K-1) \alpha\mbox{ otherwise }
\\ \end{cases}\\\\
$$



# 蒸馏方式

## 2020 CVPR: The Knowledge Within: Methods for Data-Free Model Compression

### Overview

针对情景是**在不接触原始数据情况下**， 用一个full precision teacher来生成人造样本帮助fine tune/calibrate一个量化的student。 主要是先用teacher 包含的知识来生成synthetic样本来帮助被压缩的模型精力减少performance 损失。 同时提出利用BN层的统计数据在不使用原始数据情况下， 估算数据集之间的相似度。



### 生成样本方法



> **Gaussian scheme** ： 随机高斯采样,   然后在训练中需要冻结student BN层 来防止BN出现大的偏差

> **Inception scheme:**： 最大化teacher output logit， 


>**BN-Statistics scheme:**   前两种方法容易导致生成数据和真实数据的internal statistics’ divergence， >所以可能会损害性能。 因此提出了BN-Stat， i.e., 直接较小生成数据与BN 统计信息之间的divergence
>
>
>
>* 损失函数：  给定BN Stats:   $\hat \mu, \hat \sigma $,    生成样本BN Stats $\tilde \mu(D), \tilde \sigma(D)$
>
>$$
>BNS(D,\hat \mu, \hat \sigma )=D_{KL}(\mathcal N(\hat \mu, \hat \sigma )||\mathcal N(\tilde \mu(D), \tilde \sigma(D)))\\
>=\log \frac{\tilde \sigma}{\hat \sigma}-0.5(1-\frac{\hat \sigma^2+(\hat\mu-\tilde \mu)^2}{\tilde \sigma^2})
>$$
>
>


















#  学习方式

## CVPR 2020  Neural Networks Are More Productive Teachers Than Human Raters: Active Mixup for Data-Efficient Knowledge Distillation from a Blackbox Model



### Overview

针对少量无标签样本$X$ 情况。  论文采用mixup 先构造一个样本池， 用teacher 提供原始样本以及mixup的样本的标签。  首先在原始样本上预训练student。  然后每个循环中， 从mixup samples选取出一些 student model is most uncertian，i.e., $\max P(y|x) $ 比较小的。  因为mixup构造中涉及到$x^{i,j}=\lambda x^i+(1-\lambda )x^j$  ,  需要确保$\lambda$ 选取使得student  score $C(x_i,x_j)=\min_{\lambda} \max P(y|x^{i,j})$   来训练student.  经过多次迭代， student 会学习到越来越多的uncertian mixup samples 以此提高performance.



相比于传统KD， 针对是数据不充足情况， 也并没有比较在样本充足情况下，是否比KD效果好。







