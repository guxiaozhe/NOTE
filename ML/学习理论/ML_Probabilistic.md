



## Chapter 4 Gaussian models 



####Basic

* MVG 
  $$
  \mathcal N(\mathbf x|\mathbf \mu,\Sigma) \triangleq \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\exp\left( -\frac{1}{2}(\mathbf x -\mathbf \mu)^T \Sigma^{-1} (\mathbf x -\mathbf \mu)\right)
  $$

* **Mahalanobis distance ** $(\mathbf x -\mathbf \mu)^T \Sigma^{-1} (\mathbf x -\mathbf \mu)$

* Eigendecomposition  :  
  $$
  \Sigma=\mathbf U \Lambda \mathbf U^T\\
  \Sigma^{-1}= \mathbf U \Lambda^{-1} \mathbf U^T=[\frac{u_1}{\lambda_1},...][\begin{array} ~u_1^T\\u_2^T\\..\\u_D^T\end{array}]=\sum_{i=1}^D \frac{u_i}{\lambda_i}u_i^T
  $$

  $$
  (\mathbf x -\mathbf \mu)^T \Sigma^{-1} (\mathbf x -\mathbf \mu)=(\mathbf x -\mathbf \mu)^T (\sum_{i=1}^D \frac{u_i}{\lambda_i}u_i^T) (\mathbf x -\mathbf \mu)\\
  =\sum_{i=1}^D \frac{1}{\lambda_i} (\mathbf x -\mathbf \mu)^T ( u_i u_i^T) (\mathbf x -\mathbf \mu)\\
  =\sum_{i=1}^D  \frac{y_i^2}{\lambda_i}\\
  \mbox{where, } y_i= (\mathbf x -\mathbf \mu)^T u_i
  $$

  
  **发现**：Mahalanobis distance 是在正坐标系上的欧式距离

  

  #### Gaussian Identities 

  
  $$
  \mathbf z=\left(\begin{array}
  ~\mathbf x\\
  \mathbf y
  \end{array}\right)\sim\mathcal N
  \left(~\begin{array}
  ~\mu_{\mathbf x}\\
  \mu_{\mathbf y}
  \end{array},
  \left[\begin{array}
  ~A&C\\
  C^T&  B
  \end{array}\right]
  \right)=\mathcal N
  \left(~\begin{array}
  ~\mu_{\mathbf x}\\
  \mu_{\mathbf y}
  \end{array},
  \left[\begin{array}
  ~\tilde{A}&\tilde{C}\\
  \tilde{C}^T& \tilde{ B}
  \end{array}\right]^{-1}
  \right)\\
  \mathbf x\sim \mathcal N(\mu_{\mathbf x},A)
  $$

  $$
  p(\mathbf x,\mathbf y)=(2\pi)^{-D/2}|\Sigma|^{1/2}\exp\left( -\frac{1}{2}(\mathbf z -\mathbf \mu)^T \Sigma^{-1} (\mathbf z -\mathbf \mu)\right)\\
  p(\mathbf y)=(2\pi)^{-|D_y|/2}|B|^{1/2}\exp\left( -\frac{1}{2}(\mathbf y -\mathbf \mu_y)^T B^{-1} (\mathbf y -\mathbf \mu_y)\right)\\
  p(\mathbf x|\mathbf y)=\frac{p(\mathbf x,\mathbf y)}{p(\mathbf y)}\\
  =\mathcal N(\mu_x+CB^{-1}(\mathbf y-\mu_y), A-CB^{-1}C^T)\\
  \bar {\mathbf x}=\mu_x+CB^{-1}(\mathbf y-\mu_y)
  $$

  ##### The product of two Gaussians gives another (un-normalized) Gaussian 

  $$
  \mathcal N(x|a,A)\mathcal N(x|b,B)=Z^{-1}\mathcal N(x|c,C)\\
  C=(A^{-1}+B^{-1})^{-1}\\
  c=C(A^{-1}a+B^{-1}b)\\
  Z^{-1}=(2\pi)^{D/2}|A+B|^{1/2}\exp [-1/2(a-b)^T(A+B)^{-1}(a-b)]
  $$

  

  ####MLE for an MVN

  

  > (MLE for a Gaussian). If we have N iid samples xi ∼ N (μ, Σ), then the MLE for the parameters is given by 
  > $$
  > \mathbf \mu_{mle}=\sum_{i=1}^N \frac{{\mathbf x}_i}{N}\triangleq \bar {\mathbf x}\\
  > \hat \Sigma_{mle} =\frac{1}{N}\sum_{i=1}^N (\mathbf x_i-\bar{\mathbf x})(\mathbf x_i-\bar{\mathbf x})^T=\frac{1}{N}\sum_{i=1}^N \mathbf x_i\mathbf x_i^T-\bar{\mathbf x}\bar{\mathbf x}^T
  > $$
  > 

​    

​      **Proof**:
$$
\log p (\mathcal D|\mu,\Sigma)=\log \prod_{i=1}^N \left (\frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}}\exp\left( -\frac{1}{2}(\mathbf x_i -\mathbf \mu)^T \Sigma^{-1} (\mathbf x_i -\mathbf \mu)\right)\right)\\
=-\sum_i \left( \frac{1}{2}(\mathbf x -\mathbf \mu)^T \Sigma^{-1} (\mathbf x -\mathbf \mu)\right)+ N/2 \log |\Sigma|^{-1}+constant
$$



$$
\partial (\mathbf a^ T \mathbf A \mathbf a)/\partial \mathbf a =(\mathbf A+\mathbf A^T)\mathbf a \Rightarrow \partial \sum_i \left( \frac{1}{2}(\mathbf x -\mathbf \mu)^T \Sigma^{-1} (\mathbf x -\mathbf \mu)\right)/\partial \mu\\

=1/2 \sum_i (\Sigma^{-1}+\Sigma^{-T} ) (\mathbf x_i-\mu)(-1)\\
\partial \mathcal e(\mu,\Sigma)/\partial \mu=1/2 \sum_i (\Sigma^{-1}+\Sigma^{-T} ) (\mathbf x_i-\mu)=\sum_i \Sigma^{-1} (\mathbf x_i-\mu)=0\\
\Rightarrow \mu =\frac{\sum_i x_i}{N}
$$

$$
\log p (\mathcal D|\mu,\Sigma)
=-\sum_i \left( \frac{1}{2}(\mathbf x -\mathbf \mu)^T \Sigma^{-1} (\mathbf x -\mathbf \mu)\right)+ N/2 \log |\Sigma^{-1}|\\

=-\sum_i  \mbox{tr}((\mathbf x -\mathbf \mu)(\mathbf x -\mathbf \mu)^T\Sigma^{-1})+ N/2 \log |\Sigma^{-1}|\\
=-1/2\mbox{tr}(\mathbf S_{\mu}\Sigma^{-1})+N/2 \log |\Sigma^{-1}|\\
\partial e/\partial \Sigma^{-1}=-1/2\mathbf S_{\mu}^T+N/2 \Sigma^T=0\\
\Rightarrow \Sigma^T=\mathbf S_{\mu}^T/N
$$

#### Maximum entropy derivation of the Gaussian  

Multivariate Gaussian is the distribution with maximum entropy subject to having a specified mean and covariance 



To simplify notation, we will assume the mean is zero. The pdf has the form  $p(x)=\frac{1}{Z}(\mathbf x-\mu)^T \Sigma^{-1}(\mathbf x-\mu)$
$$
h(\mathcal N(\mu,\Sigma)=-\int p(x)\ln p(x) d_x=\int p(x) \left( \frac{1}{2}(\mathbf x -\mathbf \mu)^T \Sigma^{-1} (\mathbf x -\mathbf \mu)\right)d_x+\int \frac{D}{2}p(x)\ln(2\pi |\Sigma|)d_x\\
=\int p(x) \left( \frac{1}{2}(\mathbf x -\mathbf \mu)^T \Sigma^{-1} (\mathbf x -\mathbf \mu)\right)d_x=1/2\\
+D\ln (2\pi|\Sigma|)/2=\ln ([2\pi|\Sigma|e]^D)/2
$$
**Theorem **  Let   $f_{ij}(x) = x_ix_j$ and $λ_{ij} = 1/2(\Sigma^{−1})_{ij}$, for$ i,j ∈ {1,...,D}$, let $	q(x)$ satisfying corvariance $\int q(\mathbf x)x_ix_j d_{\mathbf x}=\Sigma_{i,j}$, and $p\sim \mathcal N(\mu=0,\Sigma)$, then $h(q)\leq h(p)$

**Proof** :

> $$
> 0 \leq \mathbb{KL}(q||p)=\int q(x)\log \frac{q(x)}{p(x)}d_x=-h(q)-\int q(x)\log p(x)\\
> =-h(q)-\int p(x)\log p(x)=-h(q)+h(p)\\
> \int q(x)\log p(x)=\int q(x) \left( \frac{1}{2}(\mathbf x -\mathbf \mu)^T \Sigma^{-1} (\mathbf x -\mathbf \mu)\right)d_x
> +D\ln (2\pi|\Sigma|)/2\\= \ln([2\pi|\Sigma|^D])/2=h(p)
> $$
>

**Lemma**:  拥有相同的均值和协方差矩阵的两个分布的 **相对熵=单独熵**



#### Gaussian discriminant analysis 


$$
p(\mathcal x|y=c,\theta)=\mathcal N(\mu_c,\Sigma_c)=(2\pi |\Sigma_c|)^{-D/2}\exp\left (-1/2(\mathbf x-\mu)^T(\Sigma_c^{-1})   (\mathbf x-\mu)\right)\\
\\
\Rightarrow \hat y=\arg \max_c  \log ( p(x|\theta).p(y|\vec \pi))
$$

$$
\hat y(x)=\arg \min_c\log p(\mathcal x|y=c,\theta)\propto \arg \max_c (\mathbf x-\mu)^T(\Sigma_c^{-1})   (\mathbf x-\mu)
$$

##### Quadratic discriminant analysis (QDA) 

$$
p(y|x,\theta)=\frac{p(y,x|\theta)}{\sum_{c'} p(y=c',x|\theta)}=\frac{p(x|c,\theta).p(c|\vec \pi)}{\sum_{c'} p(x|c',\theta).p(c'|\vec \pi)}\\
=\frac{\pi_c (2\pi |\Sigma_c|)^{-D/2}\exp\left (-1/2(\mathbf x-\mu)^T(\Sigma_c^{-1})   (\mathbf x-\mu)\right) }{\sum_{c'} p(x|c',\theta).p(c'|\vec \pi)}\\
$$

##### Linear discriminant analysis (LDA)  

Suppose $\Sigma_c=\Sigma$
$$
p(y|x,\theta)\propto \pi_c \exp\left (-1/2(\mathbf x-\mu)^T(\Sigma^{-1})   (\mathbf x-\mu)\right)\\
=\exp\left (\mu_c^T\Sigma^{-1}\mathbf x-\frac{1}{2}\mu_c^T\Sigma^{-1}\mu_c +\log \pi_c  \right)\exp \left (-\frac{1}{2}\mathbf x^T\Sigma^{-1}\mathbf x\right)
$$
Define  

* $\gamma_c=-\frac{1}{2}\mu_c^T\Sigma^{-1}\mu_c +\log \pi_c $
* $\beta_c=\Sigma^{-1}\mu_c$


$$
p(y|x,\theta)= \frac{\exp (\beta_c^Tx+\gamma_c)}{\sum_{c'}\exp (\beta_{c'}^Tx+\gamma_{c'})}
$$


**Decision Boundary**
$$
p(y=c|x,\theta)=p(y=c'|x,\theta)\\
\beta_c^Tx+\gamma_c=\beta_{c'}^Tx+\gamma_{c'}\Rightarrow \gamma_c-\gamma_{c'}=(\beta_{c'}^T-\beta_c^T)x
$$


#####MLE for discriminant analysis 


$$
\hat {\mu_c}=\frac{1}{N_c}\sum_{i:y_i=c} x_i\\
\hat{\Sigma_c}=\frac{1}{N_c}\sum_{i:y_i=c}(x_i-\mu_c)(x_i-\mu_c)^T
$$


#####Sec 4.3-4.8













## Chapter 23 Monte Carlo inference 



###Sampling from standard distributions 

Let  $F(x)=\mbox{Pr}(X\leq x)$ be a cdf of some distribution we want to sample from, and let $Z=F^{−1 }(U)$be its inverse,  then  if $U\sim U(0,1)$,   $F^{−1 }(U) \sim F \Rightarrow \mbox{Pr}(F^{−1 }(U)\leq x)=F(x)$

> Proof:
> $$
> \mbox{Pr}(Z\leq z)=\mbox{Pr}(U\leq u)=u=F(z)
> $$
>



Then we can sample u from $U(0,1)$ and get $z=F^{-1}(u)$



##### Box-Muller method. 





### Rejection sampling 

When the inverse cdf method cannot be used, one simple alternative is to use rejection sam- pling, 





## Chapter 24 Markov chain Monte Carlo (MCMC) 



#### Gibbs sampling 

