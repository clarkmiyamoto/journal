
- You can use Gromwall / Picard to bound $| x_t - y_t|$
- Berstein Wasserstein Space, good for dealing w/ distances between Gaussians
- You could also inspect the strength of the coupling (between corruption space & data space). For example you could write a KL divergence between the coupling and product measure (independent).
- Another idea is to write $\Sigma_{k+1} = f_\alpha(\Sigma_k)$. And if $f_\alpha = \nabla \varphi_\alpha$, you can inspect the Hessian $\nabla^2 \varphi$ and look at the PSD ordering in $\alpha$, and that'll tell you wether it gets stronger convergence or not as a function of $\alpha$.
- You could write down all steps (e.g $x = \xi$, then do $\text{DO}$) as one big operator, and then bound this operator. This has come from implicit vs explicit Euler.

Inspecting the
$$
\begin{align}
b_{k+1} = b_k + \Phi(\Sigma_k, \alpha) (b - b_k)
\end{align}
$$
Evan had the intuition that when $\Phi = \mathbb I$ it's the best because you can view $(b-b_k) = \nabla \|b - \hat b\|^2/2$ (so it's like a gradient descent on an isotropic thing).

### Book recommendations on Optimization
Online SGD
- Online Convex Optimization, by Hazan
	- Learning in an online setting when loss function is changing.

- Learning from First Principles, by Frances Bach
	- Learning in Baron space, Kernel Learning.
	- So you play more continuous time SGDs
	- Joan is much more in Frances Bach

- Statistical query
	- Theodore M. does this
	- Maybe not that good for Sampling.

- Jonathan Niles Weed. Statistical Optimal Transport 

- Non-smooth optimization, by Steven Wright (UW)

### Classes
- Graduate level Probability 1 & 2

# Other classes
- Numerical analysis
- Analysis 2 (Functional Analysis / Harmonic Analysis)
- PDE 1 & 2 (2 is more interesting)
- Probability 1 (Measure Theory) & 2 (Applied / Stochastic Analysis)

# Panos Rec
- Analysis 1