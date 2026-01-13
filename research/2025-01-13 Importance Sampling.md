#sampling 

# Problem Setup
Your goal is to estimate 
$$\mu =\mathbb E_{\pi}[f(x)] = \int f(x) \pi(x) \, dx$$
where you have oracle access to $\pi(x)$. But for whatever reason (too high of a dimension so you can't integrate via quadrature, or difficult to sample), it's difficult to compute/estimate $\int ... \pi(x) \, dx$. So you employ a (fundamental) trick
$$
\mathbb E_\pi [f(x)] = \mathbb E_{q}\left[f(x) \frac{\pi(x)}{q(x)} \right]
$$
where $q(x)$ is a distribution that you can easily sample. For notation, we'll denote $w(x) \equiv \pi(x)/q(x)$. Meaning if you sample $x_1, ..., x_N \sim^{iid} q$, you can construct the estimator $\hat \mu$ for $\mu$.
$$
\mathbb E_{q}\left[f(x) w(x) \right] \approx \frac{1}{N} \sum_{i=1}^N f(x_i) w(x_i) \equiv \hat \mu
$$
### Properties of Estimator
Inspect the mean of the estimator, we see it is unbiased.
$$
\begin{align}
\mathbb E_q[\hat \mu] & = \frac{1}{N} \sum_{i=1}^N \mathbb E_q \left[f(x_i) \frac{\pi(x_i)}{q(x_i)} \right] = \mu
\end{align}
$$
