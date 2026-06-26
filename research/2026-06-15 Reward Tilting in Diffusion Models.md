**Theorem (Doob h-transform)**
Let $$
\begin{align}
h_t(x) = \mathbb E[\exp(\lambda r(X_1)) | X_t = x]
\end{align}
$$
The SDE
$$
\begin{align}
dX_t = (b_t(X_t) + \frac{1}{2} \sigma_t \sigma_t^T \nabla \log h_t(X_t)) dt + \sigma_t dB_t
\end{align}
$$
has teh same marginals as $X_t = I_t(X_0, X_1)$ with $X_0 \sim \rho$ and $X_1 \sim \pi^{\lambda, r}$ provided $X_0 \perp X_1$. 

**Memoryless schedule** Independence is guaranteed by
$$
\begin{align}
\sigma_t = \sqrt{\frac{2(1-t)}{t}} \mathbb I_d
\end{align}
$$
Everything now boils down to computing $\nabla \log h_t$.

Rewriting the $h$ function as an integral
$$
\begin{align}
h_t(x) = \mathbb E[\exp (\lambda r(X_t)) | X_t =x]= \int \exp(\lambda r(x)) p_{1|t}(x_1 | x) \, dx_1
\end{align}
$$

**Example** DPS style plug in
Replace posterior with Dirac at its mean
$$
\begin{align}
p_{1|t}(\cdot | x) = \delta_{x}
\end{align}
$$
Consequence
$$
\begin{align}
h_t(x) \approx \exp (\lambda r (\mathbb E[X_1 | X_t = x]))
\end{align}
$$


**Example**
Consider
$$
\begin{align}
\pi = \mathcal N(\mu, \Sigma), \qaud r(x) = - \|x - a\|_2^2
\end{align}
$$
Then
$$
\begin{align}
p_t = \mathcal N(t \mu, (1-t)^2 \mathbb I_d + t^2 \Sigma)
\end{align}
$$
