#rmt #math 

So in [[2025-01-13 Autocorrelation of HMCs]] I have this weird object

$$
B = \begin{pmatrix}
 |  & ... & |\\
 \vec a_1 - \frac{1}{N/2}\sum_{i=1}^{N/2} \vec a_i & ... & \vec a_{N/2} - \frac{1}{N/2}\sum_{i=1}^{N/2} \vec a_i\\
 | & ... & |
\end{pmatrix} 
$$
where $\vec a_i \in \mathbb R^d$ meaning $B \in \mathbb R^{d \times N/2}$. I also make $\vec a_i \sim^{iid} \mathcal N(0, \mathbb \sigma^2 I_d)$. Basically I'm trying to compute $\mathbb E[\cos(\sqrt{BB^T} t)]$... If it was a vector this is barely computable...


# Wishart Distribution
**Definition**
	Let $X \in \mathbb R^{d \times n}$ be a data matrix where the $n$ columns are iid according to $\mathcal N(0, \Sigma$). Then the matrix $S = XX^T$ is distributed according to the Wishart Distribution $W_d(n, \Sigma)$. 

We defined them this way because we want to put a distribution over estimating covariance matricies. That is $\mathbb E[S] = \Sigma$.






I have no problem accepting that $BB^T \sim \sigma^2 \, W_d(\frac{N}{2} - 1, \mathbb I_d)$, where $W_d(n, \Sigma)$ is a Wishart distribution, where the sampled matrix is $d \times d$ dimensional, and $n$ is the degrees of freedom.

**Fact:**
Let $M \sim W_d(n, \Sigma)$, where $\Sigma$ is isotropic. For an even matrix function $f$ (that is $f(M) = f(-M)$), you can show
$$
\boxed{\mathbb E[f(M)] = \frac{1}{d} \sum_i \mathbb E [f(\lambda_i)] \mathbb I_d}
$$
where $\lambda_i$ is the $i$'th eigenvector of $M$.

