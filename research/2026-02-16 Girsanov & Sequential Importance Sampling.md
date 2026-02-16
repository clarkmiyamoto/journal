#sampling #sequentialmontecarlo #math #stochasticprocess 

Reference
- https://alexxthiery.github.io/notes/girsanov/girsanov.html

Continuing my exposition on change of measures in stochastic process (see [[2026-02-15 Change of Variables, Cameron–Martin Formula, and Girsanov Theorem]]), I talk about a nice application of Girsanov's theorem & importance sampling.

Consider an application of Girsanov's theorem applied to $\mathbb P = \mathcal N(\mu, \Gamma)$ and $\mathbb Q = \mathcal N(\mu + \Gamma^{1/2} u, \Gamma)$ 
$$
\frac{d \mathbb Q}{d\mathbb P}(x) = \exp \left(\langle u, \Gamma^{-1/2}(x-\mu) - \frac{1}{2}\|u\|^2 \right)
$$
Now consider this applied between two SDES
$$
\begin{align}
dX_t & = b_t(X_t) dt + \sigma(X_t)\, dW_t\\
dY_t & = [b_t(X_t) + \sigma(X_t)u_t(X_t)] dt + \sigma(X_t) dW_t
\end{align}
$$
where $\mathbb P$ is the measure in path space over $X_t$, and $\mathbb Q$ is the corresponding for $Y_t$.
These have the Euler Maruyama discretization
$$
\begin{align}
X_{t+h} & = X_t + h \, b_t(X_t) + \sqrt{h}\, \sigma(X_t) \, z\\
Y_{t+h} & = Y_t + h [b_t(X_t) + \sigma(X_t)u_t(X_t)] + \sqrt{h} \sigma(X_t) \, z
\end{align}
$$
where $\text{Law}(X_0 ) = \text{Law}(Y_0) = \nu$
If you write the joint probability over $X_{t_i}$ (where $i = 0, 1, ..., K$ with boundary conditions $t_0 = 0$ and $t_K = T$) becomes
$$
\begin{align}
p(X_{[0:T]}) & = \nu(X_0) \prod_{i=1}^K p(X_{t_{i}} | X_{t_{i-1}})\\
& \propto p(X_0) \prod_{i=1}^K \exp \Big(- \frac{1}{2h} \| x_{t_i} - [x_{t_{i-1}} + h \, b_{t_{i-1}}] \|_{\Gamma_{t_{i-1}}^{-1}} \Big)\\
q(Y_{[0:T]}) & \propto \nu(Y_0) \,  \prod_{i=1}^K \exp \left(- \frac{1}{2h}\|y_{t_i} - [y_{t_{i-1}} + h [b_{t_{i-1}}(y_{t_{i-1}}) + \sigma(y_{t_{i-1}})u_{t_{i-1}}(y_{t_{i-1}})]]\|_{\Gamma_{t_{i-1}}^{-1}} \right)\\
& = 
\end{align}
$$
where we've introduced the shorthand $b_{t_{i}} \equiv b_{t_i}(x_{t_i})$ and $\Gamma_{t_i} = \sigma(x_{t_i}) \sigma(x_{t_i})^{T}$. We can now calculate the Radon Nikodym derivative
$$
\frac{d\mathbb P}{d\mathbb Q} = \prod_{i=1}^K \exp \left( \right)
$$