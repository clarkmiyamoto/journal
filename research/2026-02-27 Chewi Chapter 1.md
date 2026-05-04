# Markov Semigroup
**Definition** For a time-homogeous Markov process $(X_t)_t$, its associated **Markov semigroup** $(P_t)_t$ is the familiy of operators acting on functions via
$$
\begin{align}
P_t f(x) := \mathbb E[f(X_t) | X_0 = x]
\end{align}
$$
It has the properties of a semigroup
- $P_0 = \text{id}$.
  *Proof:* $P_0f(x) = \mathbb E[f(X_0) | X_0 = x] = f(x)$
- $P_s P_t = P_t P_s = P_{s+t}$
  *Proof:* $P_s P_t f(x) = P_s \mathbb E[f(X_t) | X_0 = x] = \mathbb E[ \mathbb E[f(X_t) | X_0 = X_s] | X_0 = x ] = \mathbb E[f(X_{t+s}) | X_0 = x]$  

**Definition** The **infinitesimal generator** $\mathcal L$ associated with the Markov semigroup $(P_t)_t$ is the operator defined by
$$
\begin{align}
\mathcal Lf = \lim_{t \downarrow 0} \frac{P_t f - f}{t}
\end{align}
$$
The natural space of functions for which this is defined is $L^2(\pi)$ where $\pi$ is the stationary distribution of diffusion.

*Example:* Let's compute the generator associated w/ Langevin dynamics where $\nabla \log \pi(x) = - \nabla V(x)$ 
$$
\begin{align}
dX_t & = -\nabla V(x) + \sqrt 2 \, dW_t\\
X_t & = X_0 - \int_0^t -\nabla V(X_s) \, ds + \sqrt 2 W_t
\end{align}
$$
Now we can compute how $P_t$ acts
$$
\begin{align}
\mathbb E[f(X_t)]
\end{align}
$$
