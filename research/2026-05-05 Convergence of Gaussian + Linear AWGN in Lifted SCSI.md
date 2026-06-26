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

# Setup
- True data $X \sim \mathcal N(\mu, \Sigma)$
- Observed data: $Y = A X +\sigma W \sim \mathcal N(A \mu, A \Sigma A^T + \sigma^2 \mathbb I)$
- Current iteration: $X^{(k)} \sim \mathcal N(\mu_k , \Sigma_k)$
- Lifted observation: $Y^{(k)} = \mathcal F(X^{(k)}) \sim \mathcal N(A \mu_k , A \Sigma_k A^T +\sigma^2 \mathbb I)$

$$
\begin{align}
X_1 | y \sim X^{(k)} | Y^{(k)} = y
\end{align}
$$
$$
\begin{align} 
\mathbb{E}[x_1 | y] &= \mu_k + K_k (y - A\mu_k) \\ 
\text{Var}(x_1 | y) &= \Sigma_k - K_k A \Sigma_k 
\end{align}
$$
where $K_k = \Sigma_k A^T (A \Sigma_k A^T + \sigma^2 \mathbb I)^{-1}$

$$
\begin{align} \mu_{k+1} &= \mathbb{E}_Y[\mathbb{E}[x_1 | Y]] \\ 
&= \mathbb{E}_Y[\mu_k + K_k (Y - A\mu_k)] \\ 
\Aboxed{\mu_{k+1} &= \mu_k + K_k A (\mu - \mu_k)} \end{align}
$$
$$
\begin{align} 
\Sigma_{k+1} &= \mathbb{E}_Y[\text{Var}(x_1 | Y)] + \text{Var}_Y(\mathbb{E}[x_1 | Y]) \\ 
&= (\Sigma_k - K_k A \Sigma_k) + \text{Var}_Y(\mu_k + K_k(Y - A\mu_k)) \\ 
&= \Sigma_k - K_k A \Sigma_k + K_k \text{Var}(Y) K_k^\top \\
&= \Sigma_k - K_k A \Sigma_k + K_k (A\Sigma A^\top + \sigma^2 \mathbb{I}) K_k^\top \\
\Sigma_{k+1} &= \Sigma_k + K_k A (\Sigma - \Sigma_k) A^\top K_k^\top
\end{align}
$$

# Case: $A$ is fixed
