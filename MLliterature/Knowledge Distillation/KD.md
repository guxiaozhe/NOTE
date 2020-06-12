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





## CVPR 2020 : Regularizing Class-wise Predictions via Self-knowledge Distillation

### Overview 

主要想法就是减少同个label的不同sample之间output distribution差异。 那么不管对于correct or misclassified sample， 它们之间的差异都是相似的。 Regularization loss也简单， sample 两组有**相同label的样本**， 然后想KD一样减少它们**soft output KL散度**。 最后还添加了一项原始样本x与曾广样本x’的soft输出之间的KL散度



PS：motivation是不是成立，i.e.,强制同个label的不同样本之间的输出分布相同是不是成立。 和KD本身比并没有优势。

##  CVPR 2020:  Heterogeneous Knowledge Distillation using Information Flow Modeling

### Overview

主要考虑解决两个问题1） student在训练的不同阶段 需要不同的老师 2）利用中间层的KD通常无法应对不同结构的teacher and student，需要block之间的1-to-1match。

为了解决问题， student and teacher 的某层输出与label之间的互相学习MI 作为蒸馏目标  ，这样的好处是1）不必考虑网络结构的不同 与2）**避免了over-regularization**



### Formulation

* $g^l(x)$:   表示student layer l输出

* $f^l(x)$ :  teacher layer l输出

* $I(g^l/f^l,Y)$  ：l层输出与目标变量互信息

* $\mathbf  w_t=[I(f^1,Y)....I(f^{L_T},Y)]$ :  teacher 的互信息向量

* $\mathbf  w_s=[I(g^1,Y)....I(g^{L_S},Y)]$：student 互信息向量

  

>  **蒸馏Loss**
> $$
> \sum_{l=1}^{L_S} (\mathbf w_s[l]-\mathbf w_t[*])^2\\
> $$
> 这里* 表示$\arg\min_k (\mathbf w_s[l]-\mathbf w_t[k])^2$。  因此teacher 的某一层可能汇合student 的多个层匹配，而某些层的信息可能又全用不上。 同时如何选择不同的layer 匹配又会影响性能。






> **Quadratic Mutual Information**  来用来计算MI： 
>
>  student’s conditional probability distribution 定义如下， teacher 也类似。
>
> 
> $$
> p_{i|j}^{(s,l)}=\frac{K(g^l(x_i),g^l(x_j))}{\sum_{i\neq j}^N K(g^l(x_i),g^l(x_j))}
> $$
> 其中K是一个kernel function。那么student /teacher 的layer-wise MI 用 Jeffreys divergence 计算, 
> $$
> \mathcal L^{(l_t,l_s)}=\mathcal D(\mathcal P^{t,l_t}||\mathcal P^{s,l_s})=\sum_{i,j\neq i}(p_{i|j}^{(t,l_t)}-p_{i|j}^{(s,l_s)})(\log p_{i|j}^{(t,l_t)}-\log p_{i|j}^{(s,l_s)})\\\\
> =D_{KL}(p_{i|j}^{(t,l_t)}||p_{i|j}^{(s,l_s)})-D_{KL}(p_{i|j}^{(s,l_s)}||p_{i|j}^{(t,l_t)})
> $$
> 
>
> 


> **辅助网络** ： 为了解决层与层的匹配问题，使用一个与student结构类似（同样的层数，但是更宽）的辅助网络。 
>
> 先用传统的KD，来训练辅助网络，然后再用MI 的divergence来训练student
> $$
> \mathcal L=\alpha_i \mathcal D(\mathcal P^{t,l_i}||\mathcal P^{s,l_i})
> $$
>
> 



>**Critical Period-aware 蒸馏**  ：主要是针对不同epoch ， student处于不同状态。 比如刚初始化时候，网络的可塑性强，后来就变弱了。本文采取的策略就是在更重要时期，给与更高的权重$\alpha_i$
>$$
>\alpha_i=\begin{cases}
>1& \mbox{ if } i=L_S\\\\
>\alpha_{initial}\times \gamma^k&\mbox{ otherwise}
>\end{cases}
>$$
>
>
>其中 k表示训练的epoch，$\gamma$是一个衰减系数
>
>





PS: 实验中的student采用了一个很简单的3层卷积结构。 然后对KD的超参数设计 为0.1和 T=2 也不是大多数论文里的比较优解。 





#  学习方式

## CVPR 2020  Neural Networks Are More Productive Teachers Than Human Raters: Active Mixup for Data-Efficient Knowledge Distillation from a Blackbox Model



### Overview

针对少量无标签样本$X$ 情况。  论文采用mixup 先构造一个样本池， 用teacher 提供原始样本以及mixup的样本的标签。  首先在原始样本上预训练student。  然后每个循环中， 从mixup samples选取出一些 student model is most uncertian，i.e., $\max P(y|x) $ 比较小的。  因为mixup构造中涉及到$x^{i,j}=\lambda x^i+(1-\lambda )x^j$  ,  需要确保$\lambda$ 选取使得student  score $C(x_i,x_j)=\min_{\lambda} \max P(y|x^{i,j})$   来训练student.  经过多次迭代， student 会学习到越来越多的uncertian mixup samples 以此提高performance.



PS: 相比于传统KD， 针对是数据不充足情况， 也并没有比较在样本充足情况下，是否比KD效果好。







## CVPR 2020:  Highlight Every Step: Knowledge Distillation via Collaborative Teaching

### Overview

采用了两个teacher。 其中一个teacher 是和student 一起step by step 的训练， 并用当前的输出让student 模仿teacher 训练中的每一步状态。另一个teacher类似于attention KD 提供sample的attention map 让学生模仿。



实验数据表明显示这种协同训练有少量提升，但是发现作为baseline KD 本身的结果非常差。 怀疑是否也只是在当前实验设定下才work。







## CVPR 2020:  Few Sample Knowledge Distillation for Efficient Network Compression

### Overview

首先是从一个预训练teacher 模型prune一个student 模型，并保证他们在各个block上的feature map size一致。  然后在student模型对应teacher的block上添加1x1的卷积，来匹配student-teacher 的block输出。最后把这个1x1卷积merge到前面的卷积层里， 因为1x1 conv相当于feature map 的线性组合， 所以也是可以实现的。  最后在block wise 的student /teacher 拟合中， 是逐层进行的，即先用样本 对第一个block输出拟合，然后对第二层，依次类推。



## CVPR 2020:  Search to Distill: Pearls are Everywhere but not the Eyes

### Overview

从不同结构teacher适合不同结构的student角度（因为teacher的结构性知识 是不能传递给student的），来搜索适合的student 结构。给定teacher, 用RNN 的强化学习来在predefined 结构用KD 知道的accuracy作为reward搜索适合的结构。





## AAAI  2019: Rocket Launching: A Universal and Efficient Framework for Training Well-performing Light Net

### Overview

基本思路是teacher 模型和student 模型共享前几层的网络参数， 然后同时用label训练teacher 和student 并保持teacher 与student之间的logits 比较相似。

从实验结果看对WRN， 对比原始KD有一定提升，但是这时候KD本身效果极差， 相信因该也是特殊实验设定的结果。





