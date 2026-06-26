#paper #stochasticinterpolants 
https://arxiv.org/abs/2504.15736


Consider a target distribution $\pi$ defined on a manifold $M \subset \mathbb R^d$. We want to a generative model to sample $\pi$.
- A naive idea is to just run the diffusion model in the ambient space $\mathbb R^d$, and at a final time, the points will lie closely to the manifold. 
	- While they can do that, empirically, you'll find they struggle with this task. E.g. diffusion models on discrete data (when values are embedded in $\mathbb R^d$) struggle to learn.

 Assume $M$ is compact, that way we know it has a limiting / uniform distribution.
### Diffusion on a Manifold
The forward process is defined the generalization of the Euclidian Fokker Planck
$$
\begin{align}
\partial_t p_t = \frac{1}{2} \Delta p_t
\end{align}
$$
to the covariant Fokker Planck defined on $(M,g)$.

$$
\begin{align}
\partial_t p_t = \frac{1}{2} \Delta_g p_t
\end{align}
$$
where $\Delta_g$ is the Laplacian with metric $g$. 
$$
\begin{align}
\Delta_g \, f = \frac{1}{\sqrt{g}} \partial_i (\sqrt{g} g^{ij} \partial_j f)
\end{align}
$$
When the manifold is compact, this defines a process which equilibrates (as $t \to \infty$) to the uniform measure over the manifold.

This is a equation which conserves probability w.r.t. the volume integral on the manifold. Consider the volume measure on $M$, denote as $dV_g$.
$$
\begin{align}
\partial_t \int p_t \, dV_g & = \frac{1}{2} \int \Delta_g \, p_t \, dV_g = 0
\end{align}
$$

**Example: (Brownian Motion on a Sphere w/ radius $1$)**
Let $x = (x^1, x^2) = (\theta, \phi)$ be the coordinates. The metric is given by
$$
\begin{align}
ds^2 & = r^2 (d\theta^2 + \sin^2\theta d\phi^2) \\
g_{ij} & = \text{diag}(r^2 , r^2 \sin^2 \theta)
\end{align}
$$
When $r=1$, the Laplacian becomes
$$
\begin{align}
\sqrt{g} & = \sqrt{\det g} = \sin\theta\\
g^{\theta\theta} & r^2 = 1 \\
g^{\phi \phi} & = \frac{1}{\sin^2 \theta}\\
\Delta_g f & = \frac{1}{\sqrt g} \left( \partial_\theta (\sqrt{g}\, g^{\theta \theta} \partial_\theta f) +  \sqrt{g} \, g^{\phi \phi}\, \partial_\phi^2 f \right)\\
&= \frac{1}{\sqrt g} \partial_\theta(\sqrt g g^{\theta \theta}) \partial_\theta f + g^{\theta \theta} \partial_\theta^2 f +  g^{\phi \phi} \partial^2_\phi f\\
& = \cot(\theta)\, \partial_\theta f + \partial_\theta^2 f + \frac{1}{\sin^2 \theta} \, \partial^2_\phi f
\end{align}
$$

### Solution
But this is a process on $t \in [0, \infty)$. It'd be nice to have an interpolant / flow matching model which works on $t \in [0,1]$.

Consider the interpolant
$$
\begin{align}
I_t = I(t, X_0, X_1)
\end{align}
$$
I want an interpolant such that $I_t \in M \ (\forall t)$. 

**Reimannian Geodesic Interpolant**. Given two random variables $X_0 , X_1 : \Omega \to M$, the geodesic interpolant $I(t, X_0, X_1)$ is defined as
$$
\begin{align}
I(t, X_0, X_1) = \exp_{X_0} (t \, \log_{x_0}(x_1))
\end{align}
$$
where $\exp_{a}$ and $\log_a$ denote the exponential & logarithmic map at point $a$ respectively.

The $\text{Law}(I_t) = p_t$ has the time evolution
$$
\begin{align}
\partial_t p_t = - \text{div}_g(b_t\, p_t), \quad b_t(x) = \mathbb E[\dot I_t | I_t = x]
\end{align}
$$
