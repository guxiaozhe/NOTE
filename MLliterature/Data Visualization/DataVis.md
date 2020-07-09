# SNE :  Stochastic Neighbor Embedding  

把**high-dimensition** 样本之间的**欧氏距离**， 转化为条件概率 $p_{i|j}$ 来表示样本之间的相似性, 这里 $p_{j|i}$ 表示：   样本$x_i$ 会挑选$x_j$ 作为邻居的条件概率
$$
p_{j|i}=\frac{\exp (-||x_i-x_j||^2/2\sigma_i^2)}{\sum_{k\neq i} \exp (-||x_i-x_k||^2/2\sigma_i^2)}\\\\

$$






假设$y_i,y_j$ 是转换后的low-dimension 数据， $q_{j|i}$ 表示 样本$y_i$ 会挑选$y_j$ 作为邻居的条件概率
$$
q_{j|i}=\frac{\exp (-||y_i-y_j||^2)}{\sum_{k\neq i} \exp (-||y_i-y_k||^2)}\\\\
\frac{\partial q_{j|i}}{\partial y_i}=q_{j|i}(-2)(y_i-y_j)-q_{j|i} \sum_k q_{k|i}(-2)(y_i-y_k)\\=2q_{j|i}y_j+2q_{j|i}\sum_k q_{k|i}y_k\\\\
\frac{\partial q_{j|i}}{\partial y_j}=q_{j|i}2(y_i-y_j)-q_{j|i}^22(y_i-y_j)=2q_{j|i}(y_i-y_j)(1-q_{j|i})\\\\
\\\\
\frac{\partial q_{j|i}}{\partial y_k}=-2q_{j|i}q_{k|i}(y_i-y_k)
$$


**SNE 目标**

> 就是找到映射， 使得$q,p$  的分布更加相似，  用KL 散度作为优化目标则我们有

$$
C=\sum_i D_{KL}(p||q)=\sum_i \sum_{j\neq i} p_{j|i}\log \frac{p_{j|i}}{q_{j|i}}\\\\
=\sum_{j\neq *} p_{j|*} \log\frac{p_{j|*}}{q_{j|*}}+\sum_{i\neq *} p_{*|i}\log \frac{p_{*|i}}{q_{*|i}}+\sum_{i\neq *,j\neq *}p_{j|i}\log \frac{p_{j|i}}{q_{j|i}}\\\\
=\sum_{j\neq *}-2p_{j|*}y_j+\sum_{j\neq *}-2p_{j|*}\sum_kq_{k|*}y_k
$$



> 由于KL散度是不对称的, i.e.,  $p>q$ 时候（high dimension 样本接近，但是low dimension样本原理）， cost比较大， 反之则cost 比较小。所以， **它倾向于**  保留high dimension里的局部信息（只有邻近的点的信息得到保留，而远的点则被忽略）。



**损失函数梯度**

>$$
>
>$$












$$

$$
