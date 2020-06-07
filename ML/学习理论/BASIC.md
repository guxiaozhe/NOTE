# ERM- Empirical Risk Minimization 



## 最小化期望“平方误差 ” = 均值  

以Piecewise最小化平方误差为目标，那么最佳模型就是**条件期望**:   $f(X)=\mathbb E\left[Y|X\right ]$



>期望预测误差EPE**
$$
\text{EPE}(\hat f)=\mathbb E_{p(X,Y)} \left[(Y-\hat f(X))^2\right]\\\\
=\int p(x)\left[\int \left(y-\hat f(x)\right)^2 p(y|x)dy\right]dx \\\\
=\mathbb E_{p(X)}\left[~~\mathbb E_{p(Y|X)} \left[(Y-\hat f(X))^2 |X\right]\right]\\\\
\mbox{minimize  pointwise}\Rightarrow   \min\mathbb E_{p(Y|x)}[(Y-\hat f(x))^2 |x]\\\\
=\min \underbrace{\mathbb E[Y^2|x]}\_{\text {constant}}+\hat f(x)^2-2\hat f(x) \underbrace{\mathbb E[Y|x]}\_{\text {constant}}\\\\
\Rightarrow \hat f(x)=\mathbb E[Y|x]
$$






##  平方误差 Bias  Variance 分解



###  对于一个特定点 $(x,y)$  和模型$\hat y$ 

训练数据集$\mathcal D$,   常数label y,  那么某一个点$(x,y)$误差Bias  Variance 分解。

$$
\mathbb E\left[(y-\hat y)^2|x\right]=\mathbb E\left[y^2+\hat y^2-2y\hat y+\mathbb E[\hat y]^2-\mathbb E[\hat y]^2+2\mathbb E[\hat y]\hat y-2\mathbb E[\hat y]\hat y\right]\\\\
=\mathbb E[(\hat y-\mathbb E[\hat y])^2]+y^2-2y\mathbb E[\hat y]-\mathbb E[\hat y]^2+2\mathbb E[\hat y]^2\\\\
=\underbrace{\mathbb E[(\hat y-\mathbb E[\hat y])^2]}\_{\text{variance}}+\underbrace{(y-\mathbb E[\hat y])^2}\_{\text{bias square}}
$$

###    假设 $Y=f(X)+\epsilon,~\mathbb E[\epsilon]=0, \text{Var}(\epsilon)=\sigma^2$

训练数据集$\mathcal D$,     那么某一个点$ (x,y)$  误差Bias  Variance 分解。

$$
\mathbb E\left[(y-\hat y)^2\right]=\mathbb E\left[(f(x)+\epsilon-\hat y)^2\right]\\\\
=\mathbb E\left[ (f(x)-\hat y)^2+\epsilon^2+2\epsilon(f(x)-\hat y )     \right]\\\\
=\underbrace{\mathbb E[(\hat y-\mathbb E[\hat y])^2]}\_{\text{variance}}+\underbrace{f(x)^2+\mathbb E[\hat y]^2-2f(x)\mathbb E[\hat y]}\_{\text{bias square}}+Var(\epsilon)\\
$$

### Bias-Variance Tradeoff

> **KNN角度的Bias-Variance Tradeoff**

KNN的模型 $\hat y=\hat f_k(x)=\frac{\sum_{k=1}^K f(x_k)+\epsilon}{K}$

KNN的模型Bias-Variance 分解

$$
\mathbb E[(\hat y-y)^2]={\mathbb E[(\hat y-\mathbb E[\hat y])^2]}+({f(x)-\mathbb E[\hat y]})^2+\sigma^2\\\\
=\sigma^2+\mathbb E[(\frac{\sum_{k=1}^K f(x_k)+\epsilon}{K}-\mathbb E[\frac{\sum_{k=1}^K f(x_k)+\epsilon}{K}])^2]+({f(x)-\mathbb E[\frac{\sum_{k=1}^K f(x_k)+\epsilon}{K}]]})^2\\\\
=\sigma^2+\frac{\mathbb E[\epsilon^2]}{K}+({f(x)-\mathbb E[\frac{\sum_{k=1}^K f(x_k)+\epsilon}{K}]]})^2\\\\
=\sigma^2+\underbrace{\frac{\sigma^2}{K}}\_{\text{variance}}+\underbrace{(f(x)-\frac{\sum_{k=1}^K f(x_k)}{K})^2}\_{\text{bias square}}
$$

>分析：当k小的时候，模型复杂度大， variance大，bias 更小，（因为 更近的点 更大概率类似目标值）。









 ## 最小化“期望绝对误差 ”  = 中位数

以最小化绝对值误差为目标，那么最佳模型就是条件中位数

$$
\mbox{EPE}(f)=\mathbb E \left[|Y-f(X)|\right]=\int p(x)\int  |y-f(x)|  p(y|x)dydx\\\\
 \mbox{minimize EPE pointwise }L(x)=\int  |y-\hat y|  p(y|x)dy\Rightarrow   \\\\
 \frac{\partial L}{\partial \hat y}=0=\int \mbox{sign}(y\geq \hat y)   p(y|x)dy=\int^{\hat y} -   p(y|x)dy+\int_{\hat y}   p(y|x)dy=0\\\\
 \Rightarrow  \hat y=\mbox{median, i.e. cdf=0.5}
$$

## 最大似然概率MLE 等价 Least Square Error 

假设 $Y=f(X;\theta)+\epsilon$, where $\epsilon\sim N(0,\sigma^2)$, 那么
$$
\mbox{P}(Y|X;\theta)\sim N(f(X;\theta),\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(Y-f(X;\theta))^2}{2\sigma^2})\\\\
\log \mbox{P}(Y|X;\theta)=-\frac{(Y-f(X;\theta))^2}{2\sigma^2}+\text{constsant}
$$

**MLE:**


$$
P(
\mathbf Y|\mathbf X;\theta)=\prod_{i=1}^N\mbox{P}(y^i|x^i;\theta)\\\\
\log P(\mathbf Y|\mathbf X;\theta)= \sum_{i=1}^N -\frac{(y^i-f(x^i;\theta))^2}{2\sigma^2}+\text{constsant}\\\\
\arg \max_{\theta}P(\mathbf Y|\mathbf X;\theta) =\arg \min_{\theta}  \sum_{i=1}^N(y^i-f(x^i;\theta))^2
$$





## 最大后验概率 MAE 等价正则化

### 高斯先验分布下的MAE对应 L2 正则化

高斯先验
$$
P(\theta)=\frac{1}{(\sqrt{2\pi})^n\det(\Sigma)}\exp(-1/2\theta^T\Sigma^{-1}\theta)\\
$$
假设 $Y=f(X;\theta)+\epsilon$, where $\epsilon\sim N(0,\sigma^2)$, 那么似然概率
$$
\mbox{P}(Y|X;\theta)\sim N(f(X;\theta),\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(Y-f(X;\theta))^2}{2\sigma^2})\\
$$
由贝叶斯公式得后验概率
$$
P(\theta|\mathbf Y,\mathbf X)=\frac{P(\mathbf Y|\theta,\mathbf X)P(\theta)}{P(\mathbf Y|\mathbf X)}\propto P(\mathbf Y|\theta,\mathbf X)P(\theta)\\\\
=\prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}}\exp(-\frac{(y^i-f(x^i;\theta))^2}{2\sigma^2})\times \frac{1}{(\sqrt{2\pi})^n\det(\Sigma)}\exp(-1/2\theta^T\Sigma^{-1}\theta)\\\\
$$
Maximize MAE
$$
\theta_{MAE}=\arg\max_{\theta} P(\theta|\mathbf Y,\mathbf X)=\arg\max_{\theta} \log   P(\mathbf Y|\theta,\mathbf X)P(\theta)\\\\
=\arg \min_{\theta} \sum_{i=1}^N\left (y^i-f(x^i;\theta)\right)^2+ \theta^T \Sigma^{-1}\theta
$$

如果各个参数相互独立（$\Sigma$对角矩阵）的情况下， 就是L2正则化





### 拉普拉斯分布 下的MAE对应 L1 正则化

唯一的区别就是拉普拉斯分布的log形式。

拉普拉斯分布
$$
P(\theta)=\frac{1}{2\lambda}\exp(-1/\lambda|\theta-\mu|)\\
\log P(\theta)=-1/\lambda|\theta-\mu|
$$
如果$\mu$ 是0向量，就是L1 正则化。



# VRM- Vicinal Risk Minimization  

Empirical Risk :
$$
ER(\hat f)=\sum_i^n e(\hat f(x^i),y^i)=\sum_i^n \int e(\hat f(x),y^i)   \underbrace{\delta_{x^i}(x)}\_{考虑单独x^i一个点}
$$
而VRM 对于任意样本，考虑它的领域的分布
$$
VR(\hat f)=\sum_i^n \int e(\hat f(x),y^i)   \underbrace{\delta_{x^i}(x)}\_{考虑x^i邻域上的分布}
$$








# PAC- Probably Approximately Correct Learning 

## 概念

— 需要学习的概念 C

— 训练数据的**i.i.d** 于分布D 

— Data set  $S=(\mathbf X^{m\times d},\mathbf y)$, **m个**样本，d个feature

— 假设空间 hypothesis set H, where $\hat y=h(x)$

—  泛华误差Generalization Error 
$$
R(h)=\mbox{P}_{x\sim D}(h(x)\neq y)=\mathbb E_{x\sim D}[\mathbf  1_(\hat y\neq y)]
$$
— 经验误差Empirical Error
$$
\hat R(h)=\frac{1}{m}\sum_{i=1}^m \mathbf  1_{(h(x^i)\neq y)}\\\\
\mbox{根据中心极限定理 }\\\\

\mathbb E_{x\sim D}[\hat R(h)]=\sum_{i=1}^m \mathbb E_{x^i\sim D}[\mathbf  1_{( h(x)\neq y)}]
\\\\

=R(h)\\
$$


### PAC-Learnable

A concept class C  is said to be **PAC-learnable** if  存在

1.算法A 

2.多项式函数poly(.,.,.,) 

such that for any $\epsilon>0,\delta>0$ 和任意 $m>poly(1/\epsilon,1/\delta,d,\mbox{size of concept C})$ 
$$
\mbox{Pr}_{S\sim D}(R(h_S)\leq \epsilon )\geq 1-\delta
$$





##  有限假设空间 Finite Hypothesis Sets

### Learning Bound for Consistent Case

— 如果假设空间$|\mathcal H|$ 有限， 且

— 算法A返回关于样本S的consistent 的假设 $h_S$， 且经验误差$\hat R(h_S)=0$, 且

— $m\geq \frac{1}{\epsilon}(\log |\mathcal H|+\log \frac{1}{\delta})$,  那么以下成立
$$
\mbox{Pr}_{S\sim D}(R(h_S)\leq \epsilon)\geq 1-\delta\\
\Rightarrow \\
\mbox{Pr}_{S\sim D}\left(R(h_S)\leq \frac{1}{m}(\log |\mathcal H|+\log \frac{1}{\delta})\right)\geq 1-\delta
$$


**证明**： 
$$
\mbox{Pr}[\exists  h \in \mathcal H :\hat  R(h)=0 ∧ R(h) > \epsilon ]\\\\
=\sum_{h\in \mathcal H}\mbox{Pr}[ \hat R(h)=0 ∧ R(h) > \epsilon ]=\sum_{h\in \mathcal H}\mbox{Pr}[ \hat R(h)=0 |R(h) > \epsilon] \mbox{Pr}[R(h) > \epsilon]\\\\
\leq \sum_{h\in \mathcal H}\mbox{Pr}[ \hat R(h)=0| R(h) > \epsilon ]\\\\
\leq  \sum_{h\in \mathcal H}(1-\epsilon)^m=|\mathcal H|(1-\epsilon)^m\\\\
\Rightarrow  1- \mbox{Pr}[\exists  h \in \mathcal H :\hat  R(h)=0 ∧ R(h) > \epsilon ]\geq  1-|\mathcal H|(1-\epsilon)^m\\\\
\Rightarrow   \mbox{Pr}[\forall h \in \mathcal H :\hat  R(h)\neq 0 \vee  R(h) \leq \epsilon ]\geq  1-|\mathcal H|(1-\epsilon)^m\\\\
\Rightarrow \mbox{Pr}_{S\sim D}(R(h_S)\leq \epsilon)\geq 1-|\mathcal H|(1-\epsilon)^m
$$


### Learning Bound for Consistent Case

#### Hoeffding不等式：

给定m个取值[0,1]之间的独立随机变量 $$[x_1, x_2, \ldots ,x_m]$$，对任意 $$\epsilon>0$$有如下等式成立：
$$
\mbox{Pr}(|\frac{1}{m}\sum x_i-\frac{1}{m}\sum \mathbb E(x_i)|\geq \epsilon)\leq 2 e^{-2m\epsilon^2}
$$
假设模型h的 经验误差  $$\hat{\mathbb E}(h)$$ , 泛化误差 $$\mathbb E(h)=\mathbb E[\hat{\mathbb E}(h)]$$ 
$$
\forall h\in \mathcal H:\\\\
\mbox{Pr}(|\hat R(h)-R(h)|\geq \epsilon)\leq 2e^{-2m\epsilon^2}\\\\
\Rightarrow \\\\
\mbox{Pr}(|\hat R(h)-R(h)| \leq  \epsilon)= \left(1-\sum_{h\in\mathcal H}\mbox{Pr}(h:|\hat R(h)-R(h)|\geq \epsilon)\right)\geq 1-|\mathcal H|2e^{-2m\epsilon^2}
$$

#### 结论

- 只要样本数量 ![m](http://www.zhihu.com/equation?tex=m) 足够大或者假设空间的大小 ![\vert\mathcal{H}\vert](http://www.zhihu.com/equation?tex=%5Cvert%5Cmathcal%7BH%7D%5Cvert) 足够小，就能保证学到的假设 ![h^{'}](http://www.zhihu.com/equation?tex=h%5E%7B%27%7D) 的泛化误差 ![E(h^{'})](http://www.zhihu.com/equation?tex=E%28h%5E%7B%27%7D%29) 与经验误差 ![\hat{E}(h^{'})](http://www.zhihu.com/equation?tex=%5Chat%7BE%7D%28h%5E%7B%27%7D%29) 足够接近
- 为什么少量样本用CNN/RNN等复杂模型会导致过拟合？还是看公式（2）。样本数量m太小的话 ![\vert{E(h^{'})-\hat{E}(h^{'})}\vert\le\epsilon](http://www.zhihu.com/equation?tex=%5Cvert%7BE%28h%5E%7B%27%7D%29-%5Chat%7BE%7D%28h%5E%7B%27%7D%29%7D%5Cvert%5Cle%5Cepsilon) 发生的可能性变小。**即学到的 ![h^{'}](http://www.zhihu.com/equation?tex=h%5E%7B%27%7D) 在训练样本上的表现与在真实样本上不一致，这是过拟合直接的表现形式**。





### 什么条件才能满足PAC可学习?

假设
$$
\delta=2|\mathcal H|e^{-2m\epsilon^2}\\
-2m\epsilon^2=\ln(\frac{\delta}{2|\mathcal H|})\Rightarrow m=\frac{\ln(\frac{\delta}{2|\mathcal H|})}{-2\epsilon^2}
$$
所以样本空间确定条件下， 只要样本数量 大于m



## 无限假设空间 Infinite Hypothesis Sets

— 损失函数 $L(h(x),y): (h(x),y)\rightarrow \mathbb R$

— 关联函数$g: (x,y)\rightarrow L(h(x),y)$  直接输出了某个点对应假设h的损失

— $G=\{g: h\in \mathcal H\}$  



### Rademacher Complexity 

— The Rademacher complexity captures the richness of a family of functions by measuring the **degree to which a hypothesis set can fit random noise**.  **拉德马赫尔复杂度通过测量一个函数族拟合随机噪声的能力来反映该函数族的丰富度**

####Empirical Rademacher complexity

— 给定样本 $S=(z_1,…z_m)$

— $\mathbf g_S=(g(z_1),…g(z_m))$

— $\sigma_i$:  50% 概率 取1 or -1, 向量 $\sigma=(\sigma_1,…\sigma_m)$



**Empirical Rademacher complexity 相对于 G和S 定义如下**
$$
\hat{\mathfrak R}_S(G)=\mathbb E_{\sigma}\left[\sup_{g\in G} \frac{1}{m}\sum_i \sigma_i g(z_i) \right]\\
=\mathbb E_{\sigma}\left[\sup_{g\in G} \frac{\mathbf g_S^T \sigma}{m} \right]\\
$$


**理解** ：

— $\mathbf g_S^T \sigma$ 表示 g在 单个函数在样本集S上与随机噪声 $\sigma$ 的相关性(correlation)

— 取上确界的动作，则表示函数族 G 在样本集S上与随机噪声相关性(correlation)

— 最后取期望的动作，则表示函数族G样本集S上与随机噪声相关性的平均水平

G的丰富度——更加丰富和复杂的函数族 G能够生成更多的 $\mathbf  g_S$ ，因此平均水平上能够更好地与随机噪声“契合”

####从  Emperical Rademacher Complexity 到   Rademacher Complexity

— D 样本分布

Rademacher complexity of G is **the expectation of the empirical Rademacher complexity** over all samples of size m drawn according to D
$$
\mathfrak R_m (G)=\mathbb E_{S\sim D}[\hat {\mathfrak R}_S(G)]
$$

####g输出0-1

for any δ > 0, with probability at least 1 − δ, each of the following holds for all g ∈ G:

**略**

#### 拉德马赫尔复杂度边界

$\mathcal H$是输出空间 $\{-1,1\}$ 的函数族， G是0-1损失下与相关$\mathcal H$的损失函数族； 对于任何 $\delta>0$ ，有至少 $1-\delta$的概率，对于从分布D上i.i.d.的容量为 m 的样本集S ，每个 $h\in \mathcal H$都有下列不等式成立：

$$
R(h)\leq \hat R_S(h)+\hat {\mathfrak R }_m(H)+\sqrt\frac{\log 1/\delta}{2m}\\
R(h)\leq \hat R_S(h)+\hat {\mathfrak R }_S(H)+3\sqrt\frac{\log 2/\delta}{2m}\\
$$


上述泛化边界为二元分类问题的可学习性提供了保证。注意，第二个边界$R(h)\leq \hat R_S(h)+\hat {\mathfrak R }_S(H)+3\sqrt\frac{\log 2/\delta}{2m}\\$是**数据依赖的(data-dependent)** 



==当然对于Infinite $\mathcal H $ sup不好求==



###增长函数（growth function）

The growth function $\prod_H: \mathbb N\rightarrow \mathbb N$  |定义
$$
\prod_H(m)=\max_{S\in D^m}|\{h(x_1),....h(x_m):h\in \mathcal H\}|
$$




— 增长函数表示假设空间H对m个示例所能赋予标记的**最大可能结果数**。函数值越大则假设空间H的表示能力越强，复杂度也越高，学习任务的适应能力越强。

— 不过尽管H中可以有**无穷多的假设h，但是增长函数却不是无穷大的：**对于m个示例的数据集，最多只能有$2^m$ 个标记结果，而且很多情况下也达不到$2^m $的情况。

#### 例子

比如说现在数据集有两个数据点，考虑一种二分类的情况，可以将其分类成A或者B，则可能的值有：AA、AB、BA和BB，所以这里增长函数的值为4.

- **对分**： 对于二分类问题来说，H中的假设对D中m个示例赋予标记的每种可能结果称为对D的一种**对分（dichotomy）**。对分也是增长函数的一种上限。

- **打散**：

  打散指的是假设空间H能实现数据集D上全部示例的对分，即**增长函数=![2^m](https://www.zhihu.com/equation?tex=2%5Em)。**但是认识到**不打散**是什么则更加重要——

  有些情况下**H的增长函数不可以达到对应的![2^m](https://www.zhihu.com/equation?tex=2%5Em) 值**，比如说在二维实平面上的线性划分情况中，以下的情况就不可以线性可分（也就是说不能算作赋予标记的结果）：

  ![img](https://pic3.zhimg.com/v2-7492d14da3e2b248e2c4971f1937ad12_b.png)![img](https://pic3.zhimg.com/80/v2-7492d14da3e2b248e2c4971f1937ad12_hd.png)

  或者下图这个

  ![img](https://pic4.zhimg.com/v2-64faf9d2dc907120bbc9d859b35677a3_b.png)![img](https://pic4.zhimg.com/80/v2-64faf9d2dc907120bbc9d859b35677a3_hd.png)

  虽然图画的非常直击灵魂，但是你应该可以体会到这种情况下二维平面的线性分类器是**不可以给上面的情况分类的**（事实上对于任何集合，其![2^4](https://www.zhihu.com/equation?tex=2%5E4)=16种对分中至少有一种不能被线性划分实现 ）



###Rademacher complexity 和  growth function.

#### G 输出是-1，+1情况下

$$
\mathfrak R_m(G)\leq \sqrt{\frac{2\log \prod_G(m)}{m}}
$$

#### 增长函数的泛化边界(Growth function generalization bound)

— $\mathcal H$ be a family of functions taking values in {−1, +1}

— with Probability $1-\delta$, 
$$
R(h)\leq \hat R(h)+\sqrt\frac{2\log \prod_H(m)}{m}+\sqrt\frac{\log 1/\delta}{2m}
$$




### Vapink-Chervonenkis Dimension

####  Shatter

**一个假说集合能够打散m个样例**，是指该假说集合最多能够标记出所有由m个样例构成的对分，也就是
$$
\prod_H(m)=2^m
$$


### VC维 定义

一个假说集合 $\mathcal H $的VC维是其能够打散的最大样例数量：
$$
VC(\mathcal H)=\max_{\prod_H(m)=2^m} m
$$




对于一个假设空间H，如果存在m个数据样本能够被假设空间H中的函数按所有可能的![2^m](https://www.zhihu.com/equation?tex=2%5Eh) 种形式分开 ，则称假设空间H能够把m个数据样本打散（shatter）。假设空间H的VC维就是能打散的最大数据样本数目m。若对任意数目的数据样本都有函数能将它们shatter，则假设空间H的VC维为无穷大。

