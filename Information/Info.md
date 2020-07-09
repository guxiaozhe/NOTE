# 熵

## 熵 (香农熵)

* 对整个概率分布中的不确定性总量进行量化 
* 最优编码长度



### 最大化熵

在已知部分知识的前提下，关于未知分布最合理的推断就是符合已知知识最不确定或最随机的推断，其原则是承认已知事物（知识），且对未知事物不做任何假设，没有任何偏见。

$$
H(X)=\mathbb E_{P(X)}\left[-\log P(X)\right]\\\\
=-\int P(x)\log P(x)d_x
$$





##  条件熵  H(Y|X）

联合分布和条件分布的差异

条件概率熵在条件变量上的期望


$$
H(Y|X)=\mathbb E_{P(X)}[H(Y|x)]=\int_x \int_y  -P(y|x)\log P(y|x)  P(x)\\\\
=\int_x \int_y  -P(y,x)\log \frac{P(y,x)}{P(x)}
=\int_x \int_y  -P(y,x)\log P(y|x)\\\\
=H(X,Y)- \int_x \log P(x) \int_y P(x,y)\\
=H(X,Y)-H(X)
$$




## Mutual Information 互信息

变量X,Y之间的独立性
$$
I(X,Y)=\mathbb E_{P(X,Y)} \log\frac{P(X,Y)}{P(X)P(Y)}\\\\
=\int P(x,y) \log P(y|x)P(x)-P(x,y)\log P(x)P(y)\\\\
=\int P(x,y) \log P(y|x)-P(x,y)\log P(y)=H(Y) \int P(x) -H(Y|X)\int P(x)\\\\
=H(Y)-H(Y|X)\\\\
=H(Y)+H(X)-H(X,Y)
$$







##  KL 散度(Kullback-Leibler (KL) divergence)  - 相对熵

— 额外编码
$$
KL(P||Q)=\mathbb E_{ P(X)}\left[\log\frac{P(x)}{Q(x)} \right]=\mathbb E_{x\sim P}\left[\log P(x)-\log Q(x) \right]\\\\
=H(P,Q)-H(P)
$$

用q分布的最佳编码，来发送p分布的消息的额外编码

## 交叉熵

用q分布的最佳编码，来发送p分布的消息的编码
$$
H(P,Q)=\mathbb E_{P(X)}\left[-\log Q(x) \right]=H(P)+KL(P|Q)
$$

##Jensen-Shannon 散度

**JS散度**度量了两个概率分布的相似度，基于KL散度的变体，解决了KL**散度非对称**的问题。一般地，JS散度是对称的，其取值是0到1之间。定义如下：
$$
JS(P||Q)=0.5KL(P||(P+Q)/2)+0.5KL(Q||(P+Q)/2)
$$
