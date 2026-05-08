#scsi
Extension of [[2026-05-01 Convergence of Gaussian + AWGN in Lifted SCSI]] 
___
Let's look at the convergence of Gaussian but on a linear projection AWGN channel
- $\mathcal F(X) = A X + \sigma W$, where $A \in \mathbb R^{m \times n}$. What are the dynamics when $m < n$ ($A$ moves $X$ to a lower dimensional space)
	- Obviously $m < n$ makes the problem not injective, but what if $A$ is random for each iteration $k$? You should be recover it, but perhaps interesting thing happen?
- In the CryoEM case $A$ will be random, perhaps we can select a couple nice random matricies that will be nice to compute
	1. Orthogonal sampled from Haar Measure
	2. Bernoulli Masking (this is image in-painting)
	We'll need to look at the Lyapunov exponent of Radom Matrix Products
____
Prior $X^{(k)} \sim \mathcal N(\mu_k, \Sigma_k)$ and observation $Y = A X^{(k)} + \sigma W \sim \mathcal N( A \mu_k , A \Sigma_k A^T + \sigma^2 \mathbb I)$. So the posterior is
$$
\begin{align}
X^{(k+1)}:= X^{(k)} | Y & = y \sim \mathcal N(\mu_{X|y}^{(k)}, \Sigma_{X|y}^{(k)})\\
\mu_{X|y}^{(k)} & = \mu_k + \Sigma_k A^T (A \Sigma_k A^T + \sigma^2 \mathbb I) (y - A \mu_k)\\
\Sigma_{X|y}^{(k)}& = \Sigma_k - \Sigma_k A^T (A \Sigma_k A^T + \sigma^2 \mathbb I)^{-1}
\end{align}
$$
