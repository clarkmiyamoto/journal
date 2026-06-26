#talk #sampling 

Speaker: Aimee Maurais
# Notes
**Objective**: sample distribution $\pi$ on $\mathbb R^d$.
In particular we're working in the Bayesian inference setting, where $\pi$ is a posterior (meaning you have unnormalized access).

Sampling via transport.
- Given a reference density $\eta$
- Find a transport map $S : \mathbb R^d \to \mathbb R^d$, s.t. $S_\# \eta = \pi$
The nice part of this is sampling $\pi$ is as easy as sampling $\eta$.

But we've moved the the problem to finding the transport map $S$.

**Dynamic Measure Transport**
Consider a family distributions $\left( p_t \right)_{t \in [0,1]}$ s.t. you iteratively flow according to the family of distributions. In continuous time, you can think of the transport map $S$ becoming the dynamics
$$
\begin{align}
dX_t = v_t(X_t)dt + \sigma_t(X_t) dW_t, \quad X_0\sim \eta
\end{align}
$$
which ensures $X_t \sim p_t$

She's enabled DMT for desnity-driven sampling by developing
1. Interacting particle system for Bayesian sampling: Kernel Fisher-Rao Flow
2. Principled design of DMT
3. Structure exploitation for scalable DMT

### Kernel Fisher-Rao Flow
Consider the geometric mixture $\mu_t = \eta^{1-t} \pi^t \propto \eta(\pi/\eta)^t$ and find an ODE
$$
\begin{align}
dX_t = v_t(X_t)\, dt
\end{align}
$$
By Fokker Planck, you must satisfy continuity equation. Since we've ansatz the flow, you can derive
$$
\begin{align}
- \nabla \cdot (\mu v) = \mu \left( \log \frac{\pi}{\eta} - \mathbb E_\mu [\log \pi / \eta] \right) \equiv \partial_t \mu
\end{align}
$$
The middle term $\log ... - \mathbb  \log...$  is independent of normalizing constant. One more time, take $\mu = \nabla u$, and find $u$. This becomes a Poisson equation, and now you can find a "weak solution" using RKHS

Take ansatz
1. Take RKHS ansatz: $u(\cdot , t) = \int k(\cdot, x) f(x, t) \mu(x, t) dx$, where $k$ is a kernel
2. Compute $\nabla u(\cdot , t) = \int \nabla_1 k(\cdot, m) M^{-1} K (\log ... - \mathbb E \log ...)(x) \mu(x,t) dx$
3. This corresponds to a mean-field ODE for sampling $\dot X_t = \mathbb E_{p_t} [\nabla_1 k(X_t, X') M K (\log ... - \mathbb E \log ...)...]$.
This ODE just is an interacting particle system.

- This requires samples for $\eta$ and evaluations of $\log \pi / \eta$.
	- Density of $\pi$ is known, set $\eta$ to tractable reference. In the bayesian setting set $\pi / \eta = \ell$ (choose $\eta$ to be the actual prior, which makes this become the likelihood)
- It's gradient free & in closed-form

**Can we use other paths $(p_t)_t$?**
The geometric path sometimes has teleportation behavior.

Mate and Fleuret (2013), perturb the geometric mixture
$$
\begin{align}
\log \mu_t(x) = (1-t) \log \eta + t \log \pi + t(1-t) f_t(x) - \log Z_f(t)
\end{align}
$$
And you learn the velocity by using a PINN loss. This is similar to the NETS algorithm.

**PDE-Constrained optimization for Paths**
Let's seek tilted paths $\log p_t(x) = \log \mu_t(x) + g_t(x) - \log Z_t$. Instead of using a PINN loss (which relies on implicit regularization due to the ADAM optimizer, that is your hoping ADAM can solve it) the corresponding velocity is found by solving
$$
\begin{align}
\inf \|v \|^2 + \lambda_g \| g \|^2 \text{ s.t. } \begin{cases}  \text{PDE } \\ \text{boundary condition}\end{cases}
\end{align}
$$


