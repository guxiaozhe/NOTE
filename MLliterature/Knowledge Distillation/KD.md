#  学习方法



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





### Virtual Teacher

$$
p^d(k)=\begin{cases}\alpha\mbox{ if }k=c\\\\
(1-\alpha)/(K-1) \alpha\mbox{ otherwise }
\\ \end{cases}\\\\
LS: q'(k)=(1-\alpha)q(k)+\alpha/K\\\\
VKD: q'(k)=\begin{cases}(1-\alpha)q(k)+\alpha^2 \mbox{ if }k=c\\\\
(1-\alpha)q(k)+(1-\alpha)\alpha/(K-1) \alpha\mbox{ otherwise }
\\ \end{cases}\\\\
$$





# 知识种类



