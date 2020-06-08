# 熵

##熵 (香农熵)

* 对整个概率分布中的不确定性总量进行量化 
* 最优编码长度


$$
H(X)=\mathbb E_{P(X)}\left[-\log P(X)\right]\\\\
=-\int P(x)\log P(x)d_x
$$



## KL 散度(Kullback-Leibler (KL) divergence)

— 额外编码
$$
D_{KL}(P||Q)=\mathbb E_{x\sim P}\left[\log\frac{P(x)}{Q(x)} \right]=\mathbb E_{x\sim P}\left[\log P(x)-\log Q(x) \right]\\
=H(P,Q)-H(P)
$$

用q分布的最佳编码，来发送p分布的消息的额外编码

## 交叉熵

用q分布的最佳编码，来发送p分布的消息的编码
$$
H(P,Q)=\mathbb E_{x\sim P}\left[-\log Q(x) \right]=H(P)+D_{KL}(P||Q)
$$



## why 最大化熵

在已知部分知识的前提下，关于未知分布最合理的推断就是符合已知知识最不确定或最随机的推断，其原则是承认已知事物（知识），且对未知事物不做任何假设，没有任何偏见。