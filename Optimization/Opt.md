# Reinforce  Gradient Estimator

$$
\frac{\partial }{\partial\phi}\mathbb E_{p_\phi(\mathbf z)}[\mathcal L(\mathbf z)]= \frac{\partial }{\partial\phi}  \int \mathcal L(\mathbf z) p_\phi(\mathbf z) =\int \frac{\partial p_\phi(\mathbf z)}{\partial\phi } \mathcal L(\mathbf z) \\\\
=\int  \mathcal L(\mathbf z)  p_\phi(\mathbf z)\frac{\partial \log p_{\phi}(\mathbf z)}{\partial \phi}\\
=\mathbb E_{p_\phi(\mathbf z)} [\mathcal L(\mathbf z)\frac{\partial \log p_{\phi}(\mathbf z)}{\partial \phi}]
$$

然后通过对z采样就可以估计出梯度。

