#stochasticinterpolants 

Papers from Yifan Chen
- [Lipschitz-guided design of interpolation schedules in generative models](https://arxiv.org/abs/2509.01629)
- [Scale-adaptive generative flows for multiscale scientific data](https://arxiv.org/abs/2509.02971)
- [Variational Optimality of Föllmer Processes in Generative Diffusions](https://arxiv.org/abs/2602.10989)

Consider the stochastic interpolant
$$
\begin{align}
I_t = \alpha_t z + \beta_t x_1
\end{align}
$$
where $z \sim \mathcal N(0, \mathbb I)$, and $x_1 \sim \mu^*$

Let $b_t(x) = \mathbb E[\dot I_t | I_t = x]$, then solutions to the ODE
$$
\begin{align}
dX_t = b_t(X_t) \, dt, \quad X_0 \sim \mathcal N(0, \mathbb I)
\end{align}
$$
satisfy $\text{Law}(X_t) = \text{Law}(I_t)$ and $X_1 \sim \mu^*$.

Additionally, for any $\epsilon_t \geq 0$, solutions to the SDE
$$
\begin{align}
dX_t = (b_t(X_t) + \epsilon_t \, \nabla \log p_t(X_t)) \, dt + \sqrt{2 \epsilon_t} \, dW_t, \quad X_0 \sim \mathcal N(0, \mathbb I)
\end{align}
$$
where
$$
\begin{align}
\nabla \log p_t(x) = - \frac{1}{\alpha_t} \mathbb E[z | I_t = x]
\end{align}
$$
You then can write
$$
\begin{align}
b_t(x) = \frac{\dot \beta_t}{\beta_t } x + \alpha^2_t ( \frac{\dot \beta_t}{\beta_t} - \frac{\dot \alpha_t}{\alpha_t}) \nabla \log p_t(x)
\end{align}
$$
Usually we will estimate $b_t$ or $\nabla \log p_t$ via a neural network. Let's assume we'll only estimate the score. Given the flexibility of choosing $\epsilon_t$, it is natural to ask, what $\epsilon_t$ is is optimal. Consider the KL divergence between path measure $\mathbb P_X$ and $P_{\hat X}$ (estimated from the neural network). Via Girsanov's theorem
$$
\begin{align}
\textsf{KL}(\mathbb P_X  \| \mathbb P_{\hat X}) \int \frac{(\alpha_t^2 (\dot \beta_t / \beta_t + \dot \alpha_t / \alpha_t) + \epsilon_t)^2}{2 \epsilon_t} \, \| \nabla \log p_t(x) - \hat s_t(x) \|_2^2 \, p_t(x) \, dt \, dx
\end{align}
$$
The minimizer is then
$$
\begin{align}
\epsilon_t = \alpha_t ( \frac{\dot \beta_t}{\beta_t} - \frac{\dot \alpha_t}{ \alpha_t})
\end{align}
$$
Remarkably the KL divergence is invariant under choice of $\alpha_t, ]beta_t$.
