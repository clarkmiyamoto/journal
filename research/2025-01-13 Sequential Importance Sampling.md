#sampling #sequentialmontecarlo 

In sequential importance sampling, you basically do [[2025-01-13 Importance Sampling]]importance but with a bunch of intermediate distributions.

Your goal is to estimate
$$
\mathbb E_\pi [f(x)] = \int f(x) \, \pi(x) \, dx
$$
where you have oracle access to $\pi(x)$, but it's difficult to compute this directly. You do the usual trick of introducing an auxiliary distribution $q(x)$ where you have sample access to
$$
\begin{align}
\int f(x) \pi(x) \, dx & = \int f(x) \frac{\pi(x)}{q(x)} \, dx\\
& = \int f(x) \, \frac{\pi(x)}{q_{t_N}(x)} \frac{q_{t_N}(x)}{q_{t_{N-1}}(x)} \, ... \,  \frac{q_{t_2}(x)}{q_{t_1}(x)} \, q_{t_N}(x)
\end{align}
$$
but you continue it, and do it for a bunch of intermediate distirbution
$$

$$

