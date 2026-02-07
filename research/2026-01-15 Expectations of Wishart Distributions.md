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
**Definition** Let $X \in \mathbb R^{d \times n}$ be a data matrix where the $n$ columns are iid according to $\mathcal N(0, \Sigma$). Then the matrix $S = XX^T$ is distributed according to the Wishart Distribution $W_d(n, \Sigma)$. 

We defined them this way because we want to put a distribution over estimating covariance matricies. That is $\mathbb E[S] = \Sigma$.

I have no problem accepting that $BB^T \sim \sigma^2 \, W_d(\frac{N}{2} - 1, \mathbb I_d)$, where $W_d(n, \Sigma)$ is a Wishart distribution, where the sampled matrix is $d \times d$ dimensional, and $n$ is the degrees of freedom.

# Marchenko-Pastur Distribution
Let $S \sim W_d(n, \sigma^2 \mathbb I)$, the associated eigenvalues $\lambda_1, ..., \lambda_d$ have the following distribution when $d,n \to \infty$ (holding $d/n = \gamma$ constant).

$$
\begin{align}
d\mu(x) & = 
\begin{cases} 
d\nu(x) & \gamma \in [0,1]\\
(1 - \tfrac{1}{\gamma}) \delta(x)\, dx + d\nu(x ) & \gamma \in (1, \infty)
\end{cases}\\
d\nu(x) & = \frac{1}{2 \pi \sigma^2} \frac{\sqrt{(\lambda_+ - x)(x - \lambda_-)}}{\gamma x} \, dx
\end{align}
$$

where $\lambda_{\pm} = \sigma^2(1 \pm \sqrt{\gamma})^2$


**Fact:**
Let $M \sim W_d(n, \Sigma)$, where $\Sigma$ is isotropic. For an matrix function $f$, you can show

$$
\boxed{\mathbb E[f(M)] = \frac{1}{d} \sum_i \mathbb E [f(\lambda_i)] \mathbb I_d} = \mathbb E[f(\lambda)] \mathbb I_d
$$

where $\lambda_i$ is the $i$'th eigenvector of $M$.

**Proof:**
Consider the eigenvalue decomposition of the matrix $M = U \Lambda U^T$. This implies for any matrix function $f(M) = U f(\Lambda) U^T$. If $M \sim W_d(n, \Sigma)$, $U \sim \mu_H$ where $\mu_H$ is the Haar measure of $O(d)$ ($d \times d$ orthogonal matricies) and it's independent from $\Lambda$.

We can attempt to evaluate the Haar measure integral first. Let's look a generic problem

$$
\begin{align}
\int U M U^T \, d\mu(U) & = V \left( \int U M U^T d\mu(U) \right)V^T 
\end{align}
$$

We can exploit the invariance under elements of the group (on the Haar measure). That is for all $V \in O(d)$, you can replace $U \mapsto V U$.  From here this implies the integral $I$ satisfies $I V = VI$ for all $V \in O(d)$, which implies it has to be proportional to the identity matrix. 

What is that coefficient of proportionality?

$$
\text{Tr}\left (\int U M U^T d\mu(U) \right) = \int \text{Tr}(U M U^T) d\mu(U) = \text{Tr}(M)
$$

But since we know it's proportional to the identity

$$
\begin{align}
\int UMU^T d\mu(U) = c \, \mathbb I_d\\
\text{Tr} \left( \int UMU^T d\mu(U)\right) = c \, d
\end{align}
$$

Therefore

$$
\mathbb E_{U \sim \mu_H}[U M U^T] = \frac{\text{Tr}(M)}{d} \mathbb I_d
$$

Applying this to our problem
$$
\mathbb E[f(M)] = \mathbb E[U f(\Lambda) U^T] = \frac{1}{d} \mathbb E[\text{Tr}(f(\Lambda))] \mathbb I_d = \frac{1}{d} \sum_i \mathbb E[f(\lambda_i)] \mathbb I_d
$$
where the expectation is taken over the distribution of eigenvalues. QED.