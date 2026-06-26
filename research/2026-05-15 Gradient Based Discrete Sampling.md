#talk #sampling 

Speaker: Ruqi Zhang
https://proceedings.mlr.press/v162/zhang22t/zhang22t.pdf
# Setup
Sample from unnormalized distribution $\pi(\theta) \propto \exp(U(\theta))$

**How do you make gradients in discrete spaces?**
So the un-normalized log prob is $U: \Omega \to \mathbb R$, but just continue it to $U : \mathbb R^d \to \mathbb R$ and evaluate along discrete domains.

**What is Langevin dynamics in discrete domains?**

# Proposed Method
Consider the Langevin dynamics proposal
$$
\begin{align}
	q(\theta' | \theta) = \frac{\exp( -\|\theta' - \theta - \alpha \nabla U(\theta)/2\|^2 / 2\alpha)}{Z_\Theta(\theta)}
\end{align}
$$
When $\Theta = \mathbb R^d$, this is a Gaussian proposal, however when $\Theta$ is discrete, obtain a gradient-based discrete proposal.

Do a coordinate factorization $q(\theta' | \theta) = \prod_{i=1}^d q_i(\theta'_i | \theta)$, s.t.
$$
\begin{align}
q_i(\theta'_i | \theta) = \text{Cat} \circ \text{Softamx} \circ \left( \frac 1 2 \nabla U(\theta)_i (\theta'_i - \theta_i) - \frac{(\theta'_i - \theta_i)^2}{2\alpha} \right)
\end{align}
$$
- Parallelize over $i$.

**Theorem** The asymptotic bias of this proposal's stationary distribution is zero for log-quadratic distributions and is small for distributions that are close to being log-quadratic.
> The convergence rate is not well known








