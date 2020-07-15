



#  分析

##  CVPR 2020:    REVISIT KNOWLEDGE DISTILLATION: A TEACHERFREE FRAMEWORK

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

##   CVPR 2020  On the Demystification of Knowledge Distillation: A Residual Network Perspective 

### Overview

根据 即使很烂的teacher 也能指导student的事实， 说明teacher 其实并不是单纯起着指导的作用，我们对它为什么work还是不清楚啊。因而我们也不知道是什么样才是好老师，什么才是好学生。事实上，学生的好坏 和教师本身无关了。



因而作者提出了个**很有趣的假设**（和我们之前的猜想很像）

>教师并不是在传到自己的结构性知识之类， 而是在训练中，帮助学生到达一个更加优秀的初始化， 然后student 在一个 a well-behaved  non-chaotic  region of loss-landscape  收敛。

**这点其实和NAS里的lucky ticket非常像： 给定初始化，需要一个lucky ticket 结构。  给定一个结构，需要一个lucky ticket initialization**

猜想如果正确： 那么是不是在几个epoch后，我们完全可以直接用hard label来训练？ 

>衍生的假设：
>
>* 正则化的student 从老师这边收益更多，除非是当老师也是高度正则化
>* 当老师高度正则化（说明老师给与学生的hint是比较speciliazed的）， 那么student表达能力高的更适应这种hint，收益效果更高。
>
>理解： 老师的hint 可能并不是适合学生的。 特别是比较特殊的hint。 当然，即**使没有正则化的老师的hint，在两者之间差异过大时候。也不一定能帮助学生。** ： 比如（有实验）resnet8 的resnet18,20,26老师中最适合它的是26， 但是换成wideresnet，即使没有regulization， 它的效果也差。
>
>* 从loss-landscape and feature reuse，  KD可以代替residual connection for 浅层网络。
>
>



**实验发现**：

>选用了Resnet 作为teacher, 然后把skip connetion remove掉的结构作为student.
>
>对于浅层的网络， KD能很好恢复性能 ，甚至超越，。
>
>随着层数变深，虽然学生网络本身性能提升， 但是KD效果却减弱了。
>
>**Loss landscape viewpoint** 
>
>随着网络深度提高，它的loss surface 变得更加chaotic , 而导致趋向于local minima，并且梯度也变得更加不稳定。 而Residual connections能防止这个问题。
>
>一个更好的初始化，能一定程度上也能解决这个问题。但是 对于过度chaotic surface， 即使一个好的初始化，可能也不行。



### Interesting Points

**Knowledge  and  Experience of A Network**:

>一个网络=解决一个问题， 它的每一层相当于解决问题的一个步骤，所以一个网络的Knowledge 是 quite structural的。 对于一个更深的网络，它学到的表征是更高层次，所以表达能力更强。 对于更宽的网络，它更容易倾向于记忆。 
>
>对于一个宽的网络加上regularization， 相当于让网络有更多experience 在解决这个问题上（比如data augmentation)，所以experience是unstructual的。 有更多experience的网络，可以更好泛华，学到问题本质而不是记忆。











# 蒸馏方式



## CVPR 2020 : Online Knowledge Distillation via Collaborative Learning



### Overview

同时训练多个子网络，并用他们的集成输出$\mathbf z^T=h(\mathbf z_1,..,\mathbf z_m)$  作为老师指导每个子模型
$$
\mathcal L_{KDCL}=\sum_i\mathcal L_{CE}^i+\mathcal L_{KD}^i(\sigma(\mathbf z_i),\sigma(\mathbf z^T))
$$
所以问题关键h选择：

1. Naive: $\mathbf z^T=\arg \min_{\mathbf z_i} \mathcal L_{CE}^i$
2. Linear:  直接优化 各个子模型的权重， 得到$\min \mathcal L_{CE}(\sigma(\mathbf z^T),\mathbf y),\mathbf z^T=\sum_i \alpha_i \mathbf z_i$
3. min Logit: $\mathbf z'=\mathbf z-z^c$ 表示减去target logit后值，  $\mathbf z^T_j=\min_i \mathbf z_{i,j}’$
4. 







##  CVPR 2020: The Knowledge Within: Methods for Data-Free Model Compression

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





**PS: 实验中的student采用了一个很简单的3层卷积结构。 然后对KD的超参数设计 为0.1和 T=2 也不是大多数论文里的比较优解。** 





## AAAI 2020: Improved knowledge distillation via teacher assistant: Bridging the gap between student and teacher



**Motivation** :  teacher size -学生size必须小于一定程度，学生才能模仿

* Experiments that show surprisingly a student model distilled from a teacher with **more parameters and better accuracy performs worse than the same one distilled from a smaller teacher with a smaller capacity.**



* Teacher assistant 到底是什么？ ： 比如最开始的teacher size 是8， 那么TA size=4， 再小就训练不了，最后再student size 2 去模仿TA
* Observation: 中间TA数量越多越好





**理论分析**

需要理解VC theory [Vapnik, V. Statistical learning theory. 1998, volume 3. Wiley, New York, 1998.] 

和论文

Lopez-Paz, D., Bottou, L., Scholkopf, B., and Vapnik, V. ¨ Unifying distillation and privileged information. arXiv preprint arXiv:1511.03643, 2015.



##  AAAI  2019: Knowledge Distillation with Adversarial Samples Supporting Decision Boundary

### Overview

**Motivation**:  靠近decison boundary 的样本对模型的影响更大。 所以找到这些decision boundary supporting samples BSS 用来帮助训练学生或许有用。



**如何找到 Boundary supporting sample (BSS)**  ： 

* 样本$\mathbf x$ 属于base class b 

* 相对于class k 的一个adversarial 样本$\mathbf x^k_0=\mathbf x$, 经过i 轮迭代 变成$\mathbf x_i^k$

* $f_b$  and $f_k$  ：  classification scores for the base class and the target class k

* adversarial attack 目标： 降低base class评分，提高class k评分
  $$
  \min L_k(\mathbf x)=f_b(\mathbf x)-f_k(\mathbf x)\\
  \Rightarrow \mathbf x_{i+1}^k= \mathbf x_{i}^k-\eta(L_k( \mathbf x_{i}^k)+\epsilon)\frac{\nabla L_k( \mathbf x_{i}^k)}{||\nabla L_k( \mathbf x_{i}^k)||_2}
  $$



**如何使用BSS**

给定一组BSSs $\mathbf x_i^k,i\in\{1,..n\},k\in{1,...C}$  
$$
\mathcal L_{BSS}=\mathcal L_{KD}+\beta \sum_i\sum_k P_n^k\times \mathcal D_{KL}(p^T_{\tau}(k),p^S_{\tau}(k)) \\\\
P_n^k=q^T(k)/(1-\max q^T(k')):\text{probability of class k being selected as the target class}
$$




##  ICCV 2019:  A Comprehensive Overhaul of Feature Distillation

### Overview

在pair-wise feature distillation时候， teacher 和student 的feature通常是通过一个transformer比较的， 这样可以避免过度正则化， 也能解决size不同问题
$$
\min distance(T^T(\mathbf f^T),T^S(\mathbf f^S))
$$


比如  AB-Distillation 就是用一个0 or 1函数表示neuron 激活， AT-distillation用激活图。 这样的缺点是造成信息丢失。 



**贡献**：

* 使用激活函数Relu之前的输出作为知识蒸馏

* 提出了margin ReLU,  $\sigma_m=\max(x,m),m<0$  作为teacher transformer

  

**最终loss**
$$
\mathcal L_{OH}=\mathcal L_{CE}+\alpha d(\sigma_m(\mathbf f^T), reg(\mathbf f^S))
$$






## ICCV 2019：Similarity-Preserving Knowledge Distillation

**Motivation**：





在teacher 里产生similar activation 的样本，在student 里， 也会产生相似的激活。

**Formulation**

* Activation Map of teacher   at layer l : $A_T^l\in \mathbb R^{b\times c\times h\times w}$  

  * b是batch size

*  Student model  activation map在相对应的layer l’ $A_S^{l’}\in \mathbb R^{b\times c’\times h'\times w'}$

   

* $Q_T^l\in \mathbb R^{b\times chw} $ :  把A 展开来

* $\bar{G}_T^l= Q_T^l Q_T^{l ~T}\in \mathbb R^{b\times b}$
  $$
  \overbrace{G_T^l[i:]}^{ith~row}=\bar G_T^l[i:]/||G_T^l[i:]||_2^2
  $$


  表明各个样本的相关系数

Loss Function :
$$
\frac{1}{b^2} \sum_{(l,l')\in I} ||G_T^l-G_S^{L'}||_F^2
$$

##CVPR 2019:  Relational Knowledge Distillation.



* Knowledge type : minibatch 内部的sample 之间关系作为匹配对象

  * distance based relation 
    $$
    \psi(x_i,x_j)=\frac{||\mathbf f_i-\mathbf f_j||_2}{\mu}\\
    \mu=\frac{\sum_{i,j}||\mathbf f_i-\mathbf f_i||_2}{N^2}
    $$

  * Angle based relation: **任意3个样本之间**的夹角
    $$
    \psi(\mathbf f_i,\mathbf f_j,\mathbf  f_k)=cos(\mathbf f_i-\mathbf f_j,\mathbf f_k-\mathbf f_j)
    $$



## CVPR 2017：A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning

### Overview 

用Flow of Solution Procedure Matrix  (FSP) 作为正则项避免过度正则化



**FSP matrix  $G\in \mathbb R^{m\times n}$**

假设 

* 特征图  $F^1\in \mathbb R^{h\times w\times m}$

- 特征图  $F^2\in \mathbb R^{h\times w\times n}$

- 那么Layer1./2的FSP   
  $$
  G_{i,j}(x;\mathbf W)=\sum_{s=1}^h\sum_{t=1}^w \frac{F^1_{s,t,i}(x; \mathbf W)\times F^1_{s,t,j}(x; \mathbf W)}{h\times w}
  $$
  ==解释： G的第i,j个元素，**表示第i个特征图 与第j个特征图**的**元素內积的均值**==



**Loss**

假设student and teacher 有n个FSP， 那么他们之间的loss in pair wise 
$$
L_{FSP}(W_t,W_s)=\frac{1}{N}\sum_{x^i}^N\sum_i^n\lambda_i\times  ||G_i^T-G_i^S||_2^2
$$

**Two-Stage 学习**

1. 训练 FSP Loss
2. 之后再在原始数据上训练 Student 



#  学习方式

## CVPR 2020  Neural Networks Are More Productive Teachers Than Human Raters: Active Mixup for Data-Efficient Knowledge Distillation from a Blackbox Model



### Overview

针对少量无标签样本$X$ 情况。  论文采用mixup 先构造一个样本池， 用teacher 提供原始样本以及mixup的样本的标签。  首先在原始样本上预训练student。  然后每个循环中， 从mixup samples选取出一些 student model is most uncertian，i.e., $\max P(y|x) $ 比较小的。  因为mixup构造中涉及到$x^{i,j}=\lambda x^i+(1-\lambda )x^j$  ,  需要确保$\lambda$ 选取使得student  score $C(x_i,x_j)=\min_{\lambda} \max P(y|x^{i,j})$   来训练student.  经过多次迭代， student 会学习到越来越多的uncertian mixup samples 以此提高performance.



PS: 相比于传统KD， 针对是数据不充足情况， 也并没有比较在样本充足情况下，是否比KD效果好。

##CVPR 2020: Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion

本文提出了一个用一个teacher 模型（考虑到pretrained teacher knowledge 包含了natural image set 的先验知识）来生成数据帮助训练student网络。为什么生成的数据要符合natural image prior : 防止student 在非天然数据是过度拟合。 然后本文首先提出了利用teacher BN 层的统计信息($\mu_l,\sigma_l$)，保证生成的数据尽量符合原始分布 (PS:这里优化目标是初始化为随机噪声的input)
$$
\mathcal R_{feature}(\hat x)=\sum_l ||(\mu_l(\hat x))-\mu_l||_2^2+\sum_l ||\sigma_l^2(\hat x)-\sigma_l^2||\\\hat x=\min L(\hat x,y)+\mathcal R_{feature}(\hat x)
$$
此外为了保证生成的数据多样性（而不是teacher 模型本身见到的那些数据）， 在损失函数中增加了teacher 与student 输出的 JS 散度
$$
\hat x=\min L(\hat x,y)+\mathcal R_{feature}(\hat x)-JS(p^S(\hat x),p^T(\hat x))
$$
![Screenshot 2020-06-18 at 10.31.16 AM](KD.assets/Screenshot%202020-06-18%20at%2010.31.16%20AM.png)















