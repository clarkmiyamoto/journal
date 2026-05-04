#notes #bayesianinference 

# Notes
First let's talk about equality in distribution. Consider $X \sim p$ and $Y \sim q$ are said to be equal, if their CDF are equal.
$$
\begin{align}
\mathbb P(X \leq z) = \mathbb P(Y \leq z)
\end{align}
$$
Equivalently,
$$
\begin{align}
X \overset{(d)}{=} Y \iff  \mathbb E[f(X)] = \mathbb E[f(Y)] \ \forall \text{ continuous bounded } f
\end{align}
$$

Now consider a distance between probability
$$
\begin{align}
d_{\mathcal F}(q,p) = \sup_{f \in \mathcal F} |\mathbb E_q[f] - \mathbb E_p[f]|
\end{align}
$$
- Different choices of $\mathcal F$ give different distances (TV, Wasserstein,etc.
	- TV is functions bounded between -1 and 1
	- Wasserstein is all 1-Lipschitz functions

We define this distance to construct a loss function, $\arg \min_\theta  d_{\mathcal F}(q_\theta, p)$. So hpoefully we can learn to sample it well. However the problem is we are interested in cases when we don't know how to compute $\mathbb E_p$ so this isn't a great method here

## Stein's Method
To get around this, pick a function class where $\mathbb E_p[f] = 0$ for all $f$ in the class. Then this distance becomes
$$
\begin{align}
d_{\mathcal F}(q,p) = \sup_f |\mathbb E_q [f]|
\end{align}
$$
So the way to do this is to pick $\mathcal F_p=  \left\{ T_p g : g \in \mathcal G \right\}$ (e.g. $G$ is all continuous functions), then construct an operator $T_p$ which transforms the functions to have $\mathbb E_p [f] = 0$.

Some questions becomes
1. Does this operator always exist for our $p$?
2. Is it expressive enough to characterize convergence?

Bad Example: Pick the operator s.t. $T_p g = 0$, this is stupid because it's not expressive enough to find convergence.

Some old math results
- A markov process $X_t$ has a stationary distribution $p$
- Consider the generator $\mathcal L\, f(x) = \lim_{t \to 0} \frac{\mathbb E[f(X_t) | X_0 = x] - \mathbb E[f(X_0)]}{t}$ 
- Consider the adjoint $\mathcal L^*$, this defines time-evolution on the density. $\partial_t p_t = \mathcal L^* \, p_t$ 

Example: Langevin $dX_t= \nabla \log p(X_t) dt + \sqrt 2 dW_t$, the Generator is $\mathcal L g(x) = \nabla \log p(x)^T g(x) + \nabla \cdot g(x)$. 
If I set $T_p g = \mathcal L g$ it is zero under $p$. See
$$
\begin{align}
\mathbb E_p [\nabla \log p \cdot g + \nabla \cdot g] & = \int p \nabla \log p \cdot g + p(\nabla \cdot g) \, dx\\
& = \int \nabla p \cdot g + p (\nabla \cdot g) \, dx\\
& = \int \nabla \cdot (p g) \,d x\\
& = \oint p \, g\cdot d \vec S\\
& = 0
\end{align}
$$

So it seems that the infinitesimal generator could make for a nice $T_p$ operator.


Aside: When $\mathcal F$ is a unit ball of the RKHS w/ kernel $k$
$$
\begin{align}
d(p,q)^2 = \mathbb E_{x \sim p, x' \sim p}[k(x,x')] + \mathbb E_{y \sim q, y' \sim q'}[k(y,y')]  - 2 \mathbb E_{x \sim p, y\sim q}[k(x,y)]
\end{align}
$$


Kernel Stein Discrepancy
Now pick $\mathcal G$ to be unit ball of RKHS w/ kernel $k$. Let $s_p(x) = \nabla_x \log p(x)$ so then
$$
\begin{align}
S(q,p) & = \sup_{g \in \mathcal G}|\mathbb E_q[T_pg]|\\
S_k(q,p)^2 &: = \mathbb E_{x \sim q, x' \sim q}[u_p(x,x')]
\end{align}
$$
where $u_p(x,x') = s_p(x)^T k(x,x') x_p(x') + s_p(x)^T \nabla_{x'} k(x,x') + \nabla_x k(x,x')^T s_P(x') = \sum_i \partial_{x_i} \partial_{x'_i} k(x,x')$  holy mol

So as a recap, we wanted a distance $d(q,p)$ without reliance on $\mathbb E_p[\cdot]$, used Stein operators which we used using the generator. We only need the score, and using the RKHS function class gave a closed form solution of the $\sup$.

## Applications:
### Neural Sampler / Stein GAN
Train $G_\theta$ such that if $z \sim \mathcal N(0, \mathbb I)$ and $x = G_\theta(z)$ satisfies $x \sim q \approx p$. We'll do the optimization using the kernel stein discrepancy. No discriminator needed, like in GANs.

### Test Statistics
You already generated $\left\{ x_i \right\}$ with MCMC, and you want to ask how well do they match $p$? 