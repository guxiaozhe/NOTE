#   SOTA

##  One-Stage KD

| 缩写                 | Loss                                                         | 描述                                                         |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| KD 、HKD     NIPS 15 | $$\mathcal L_{KD}=(1-\alpha) H(q,p^S)+\alpha D_{KL}(p^T_{\tau}\\\\$$ |                                                              |
| AT        ICLR 2017  | $$\mathcal L_{AT}=\mathcal L_{KD}+\beta\sum_{j}||A_j^S-A_j^T||_p\\\\$$ | 模仿教师中间层的激活图 attention KD                          |
| BSS     AAAI 2019    | $$\mathcal L_{BSS}=\mathcal L_{KD}+\beta \sum_i\sum_k P_n^k\times \mathcal D_{KL}(p^T_{\tau}(k),p^S_{\tau}(k)) \\\\P_n^k=q^T(k)/(1-\max q^T(k’))\\\\:\text{probability of class k being selected as the target class}$$ | 生成靠近决策边界的样本(Boundary Supporting Samples)来帮助训练 |
| SP       ICCV 2019   | $$\mathcal L_{SP}=\mathcal L_{CE}+1/{n_{batch}^2}\sum_{l,l’}||G^T_l,G^S_{l’}||_F^2\\\\G_l\text{: batch 内不同样本feature vector的內积矩阵作为相似度衡量}$$ | Similarity Preserving :保留两个样本之间相似性                |
| RKD CVPR 2019        | $$\mathcal L_{RKD}=\mathcal L_{CE}+\beta ||rela(\mathbf f_i^T, \mathbf f_j^T))-rela(\mathbf f_i^S,\mathbf f_j^S))||$$ | 用batch 内两个样本feature vector 的距离$d(\mathbf f_i,\mathbf f_j)$， 或者三个样本间的cosine $\cos (\mathbf f_i-\mathbf f_k, \mathbf f_j- \mathbf f_k)$ 衡量相似度 |
| OH  ICCV 2019        | $$\mathcal L_{OH}=\mathcal L_{CE}+\alpha d(\sigma_m(\mathbf f^T), reg(\mathbf f^S))$$ | feature distillation升级版，提出了特殊的激活函数保留了一定量的negative 信息（不想relu所有negative 信息=0），防止了信息的丢失。 |
|                      |                                                              |                                                              |

##Multi-Stage KD

|                       | Stage 1                                                      | Stage 2           | 描述                                                         |
| --------------------- | ------------------------------------------------------------ | ----------------- | ------------------------------------------------------------ |
| FitNet      ICLR 2015 | $$\mathcal L_{hint}=||\mathbf f_{hint}^T(\mathbf x)-reg(\mathbf f_{guide}^S(\mathbf x))||^2$$ | $\mathcal L_{KD}$ | 选一对layer                                                  |
| FSP  CVPR 2017        | $$\mathcal L_{FSP}=\sum_{\mathcal I \in pair} ||G_{\mathcal I}^T-G_{\mathcal I}^T||,\\\\G_{\mathcal I}:\text{ inner product of two feature map} $$ | CE                | 两个layer 內积（flow of solution）作为知识                   |
| AB    AAAI   2019     | $$\mathcal L_{AB}= (\mathbf f^T>0)-\mathbf (f^S>0)\\\\\text{对于卷积网络，则按照每个pixel 对应的feature vector}$$ | CE                | 保证teacher 激活的neuron, student 也激活，但是不考虑激活强度 |
|                       |                                                              |                   |                                                              |

##学习方式 



| 缩写           | Loss                                                         | 描述                                                         |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| MKD  CVPR 2018 | $$\mathcal L_{net1}: \mathcal L_{CE}+\mathcal D_{KL}(p_2||p_1)\\\\\mathcal L_{net2}: \mathcal L_{CE}+\mathcal D_{KL}(p_1||p_2)$$ | 两个网络 相互mutual学习                                      |
| ONE  NIPS 2018 | $$\mathcal L_{one}=\sum_i^m \mathcal L_{CE}(i)+L_{KD}(i)+\mathcal L_{CE}(ensemble)$$ | 构造一个特殊的网络， 共享浅层特征提取，然后分支多个子网络。把子网络的集成作为老师。 同时 不同子网络的权值 是可学习的。 |
| TAKD AAAI 2020 | $$\mathcal L_{neti}=\mathcal L_{KD}(net_{i-1})$$             | 不断缩小老师大小来适应学生                                   |
| KDCL CVPR 2020 | $$\mathcal L_{DKCL}=\sum_i \mathcal L_{CE}(net_i)+\mathcal L_{KD}(i)\\\\ \mathbf z^T=h(\mathbf z^T_1,...\mathbf z^T_m)$$ | 训练m个模型，并用集成logits作为teacher。 集成的方式可以是 比如最小误差的model, 子模型输出的线性组合啊 |

## 结果



<details>
  <summary>数据来源</summary>
2020 Knowledge Distillation Beyond Model Compression 
</details>

**Cifar 10**

<img src="KDsummary.assets/Screenshot%202020-07-13%20at%2010.10.23%20PM.png" style="zoom:50%;" />

**Cifar 100**

<img src="KDsummary.assets/Screenshot%202020-07-13%20at%2010.11.20%20PM-4649574.png" alt="Screenshot 2020-07-13 at 10.11.20 PM" style="zoom:50%;" />













#  简短总结

* ECCV 2020 : Knowledge Distillation Meets Self-Supervision

>类似与之前的constrative learning那篇， 用增光的数据$\tilde x$  创造额外的任务来提高子网络。 首先在teacher最后的feature添加个MLP 输出为$(z_i,\tilde z_i)$。 MLP的训练目标是增大同一个样本的cosine相似度 $\cos(z_i,\tilde z_i)$, 而减小不同样本的输出的cos相似度。训练学生时候，student最后的feature 也添加个mlp来模仿这里的相似度矩阵。
>
>**作用1**： 有个数据增广， 并用teacher 给label. 
>
>**作用2**： $(\mathbf f^T,\tilde {\mathbf f^T})$, 通过teacher MLP-student MLP-传递到student feature $\mathbf f^S$
>
>





*  CVPR 2020:   REVISIT KNOWLEDGE DISTILLATION: A TEACHERFREE FRAMEWORK

> 主要是通过发现 1） student and teacher reverse training可以提高performance 以及2）a poorly trained teacher 可以提高student performancce 得出结论： 1) KD是一种可学习的label smoothing regularization    2 ) LS 提供了一个虚拟老师。 因而完全可以找到一个适合的 virtual teacher 来提高model performance.  但是本质上也就是LS， 没有特别意义

* CVPR 2020  On the Demystification of Knowledge Distillation: A Residual Network Perspective 

>根据 即使很烂的teacher 也能指导student的事实， 说明teacher 其实并不是单纯起着被模仿的作用。因而我们也不知道是什么样才是好老师，什么才是好学生。事实上，学生的好坏 和教师本身无关了。
>
>因而作者提出了个**很有趣的假设: **教师并不是在传到自己的结构性知识之类， 而是在训练中，帮助学生到达一个更加优秀的初始化， 然后student 在一个 a well-behaved  non-chaotic  region of loss-landscape  收敛。**这点其实和NAS里的lucky ticket非常像： 给定初始化，需要一个lucky ticket 结构。  给定一个结构，需要一个lucky ticket initialization**
>
>

* CVPR 2020 : Online Knowledge Distillation via Collaborative Learning

>  同时训练多个子网络，并用他们的集成输出$\mathbf z^T=h(\mathbf z_1,..,\mathbf z_m)$  作为老师指导每个子模型。 这里的h的选择目标 主要就是 让集成输出能有搞好的泛华性。
>



* CVPR 2020: The Knowledge Within: Methods for Data-Free Model Compression

>针对情景是**在不接触原始数据情况下**， 用一个full precision teacher来生成人造样本帮助fine tune/calibrate一个量化的student。 主要是先用teacher 包含的知识来生成synthetic样本来帮助被压缩的模型精力减少performance 损失。 同时提出利用BN层的统计数据在不使用原始数据情况下， 估算数据集之间的相似度。



*   CVPR 2020:  Regularizing Class-wise Predictions via Self-knowledge Distillation

>主要想法就是减少同个label的不同sample之间output distribution差异。 那么不管对于correct or misclassified sample， 它们之间的差异都是相似的。 Regularization loss也简单， sample 两组有**相同label的样本**， 然后想KD一样减少它们**soft output KL散度**。 最后还添加了一项原始样本x与曾广样本x’的soft输出之间的KL散度

* CVPR 2020 ： Highlight Every Step: Knowledge Distillation via Collaborative Teaching

> 采用了两个teacher。 其中一个teacher 是和student 一起step by step 的训练， 并用当前的输出让student 模仿teacher 训练中的每一步状态。另一个teacher类似于attention KD 提供sample的attention map 让学生模仿。
>
> 实验数据表明显示这种协同训练有少量提升，但是发现作为baseline KD 本身的结果非常差。 怀疑是否也只是在当前实验设定下才work。
>

* CVPR 2020:  Few Sample Knowledge Distillation for Efficient Network Compression

> 首先是从一个预训练teacher 模型prune一个student 模型，并保证他们在各个block上的feature map size一致。  然后在student模型对应teacher的block上添加1x1的卷积，来匹配student-teacher 的block输出。最后把这个1x1卷积merge到前面的卷积层里， 因为1x1 conv相当于feature map 的线性组合， 所以也是可以实现的。  最后在block wise 的student /teacher 拟合中， 是逐层进行的，即先用样本 对第一个block输出拟合，然后对第二层，依次类推。

* CVPR 2020:  Search to Distill: Pearls are Everywhere but not the Eyes

>解决了不同结构teacher适合不同结构的student的问题。 通常teacher的结构性知识 是不能传递给student的，而且student 在不同的teacher 下performance是不同的， 而不是最优的teacher一定会训练出最优的student。 一个类比就是curve fitting里 我们需要选择不同的function family来适应不同的data。 因而给定teacher需要来搜索适合的student 结构。 主要的贡献是提出用强化学习来在predefined 结构用KD 训练指导的accuracy作为reward搜索适合的结构。

*  CVPR 2020:  Heterogeneous Knowledge Distillation using Information Flow Modeling


>  主要考虑解决两个问题1） student在训练的不同阶段 需要不同的老师 2）利用中间层的KD通常无法应对不同结构的teacher and student，需要block之间的1-to-1match。 为了解决问题， student and teacher 的某层输出与label之间的互相学习MI(label, Teacher feature)-MI(label, Student feature) 作为蒸馏目标  ，这样的好处是1）不必考虑网络结构的不同 与2）**避免了over-regularization**  

* CVPR 2020  Neural Networks Are More Productive Teachers Than Human Raters: Active Mixup for Data-Efficient Knowledge Distillation from a Blackbox Model

>用teacher模型， 选取mixup 曾广的数据中，不确定性高的数据来训练学生。

* CVPR 2020:   Dreaming to Distill: Data-Free Knowledge Transfer via DeepInversion

>给定随机噪声初始化的输入x， 目标就是优化x让 teacher的中间层输出 更加符合BN层理的统计信息， 并且teacher的输出有较小的损失。 此外为了保证生成的数据多样性（而不是teacher 模型本身见到的那些数据）， 在损失函数中增加了teacher 与student 输出的 JS 散度。



* AAAI 2020: Improved knowledge distillation via teacher assistant: Bridging the gap between student and teacher

>**Motivation** :  teacher size -学生size必须小于一定程度，学生才能模仿. 所以不断缩小teacher size并用前一个teacher来训练。



* AAAI  2019: Knowledge Distillation with Adversarial Samples Supporting Decision Boundary

>靠近decison boundary 的样本对模型的影响更大。 所以找到这些decision boundary supporting samples BSS 用来帮助训练学生或许有用。
>
>**用基于梯度方法找到 Boundary supporting sample (BSS)** 。
>
>

* AAAI   2019  ：  Knowledge Transfer via Distillation of Activation Boundaries Formed by Hidden Neurons

>**STAGE1:** 通过transfer activation boundary 传递 神经元是否激活，而不是激活的强度, 所以强迫老师feature  vector的激活neuron，学生feature vector也激活，老师不激活，学生也不激活
>
>**STAGE2**:  用传统CE LOSS

* ICCV 2019:  A Comprehensive Overhaul of Feature Distillation

>在pair-wise feature distillation时候， teacher 和student 的feature通常是通过一个transformer比较的， 这样可以避免过度正则化， 也能解决size不同问题 $\min distance(T^T(\mathbf f^T),T^S(\mathbf f^S))$ . 比如  AB-Distillation 就是用一个0 or 1函数表示neuron 激活， AT-distillation用激活图。 这样的缺点是造成信息丢失。 本文提出用了一个保利部分negative信息的RELU，作为teacher transformer。



* ICCV 2019：Similarity-Preserving Knowledge Distillation

>两个样本的feature 的內积被作为样本之间相关性衡量标准。 然后要求student / teacher 有相似的相关性矩阵。

* AAAI  2019: Rocket Launching: A Universal and Efficient Framework for Training Well-performing Light Net



> 基本思路是teacher 模型和student 模型共享前几层的网络参数， 然后同时用label训练teacher 和student 并保持teacher 与student之间的logits 比较相似。从实验结果看对WRN， 对比原始KD有一定提升，但是这时候KD本身效果极差， 相信因该也是特殊实验设定的结果。



* CVPR 2019:  Relational Knowledge Distillation.

>用l2距离$d(f_1,f_2)$ 或者$\cos(f_i-f_k,f_j-f_k)$ 作为两个样本之间的相关性的衡量。



*  CVPR 2017：A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning

>用两个block 输出的內积， 作为蒸馏的知识



