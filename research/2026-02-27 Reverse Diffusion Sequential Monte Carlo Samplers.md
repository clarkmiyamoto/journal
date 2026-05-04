#talk
We continue the CBS meetings (https://docs.google.com/document/u/0/d/1TB3IauL3Ns9Tx5PmLZ5oV-fAGucSlcqZcc-ztG6Ehu8/mobilebasic), today Luhuan is presenting [Reverse Diffusion Sequential Monte Carlo Samplers](https://arxiv.org/abs/2508.05926) 

```table-of-contents
```
# TLDR
You construct a sequence of distributions via the path of diffusion models. From there you can use MCMC to estimate the score and generate samples using the reverse SDE. We are very lucky because we can correct for the estimation of the score using 

# Week 1
## Setup
We have oracle access to $\pi(x) \propto \exp \left( -f(x) \right)$, where $f(x)$ is called the *energy function*. Our objective is to get samples $x \sim \pi(x)$ and also evaluate the normalization $Z = \int \exp \left( -f(x) \right) dx$.
### Annealing
The classical technique we'll take inspiration from is annealing. The idea is to embed the target distribution $\pi$ in a family of probability distributions $\pi_t$ s.t. at some final time $\pi_{t=1} = \pi$.

*Example:* Geometric annealing. Consider the annealing scheme
$$
\begin{align}
f_t(x) & =  (1-t) f(x) + t \, f_{base}(x)\\
\pi_t & \propto \pi^{1-t} \pi_{base}^t
\end{align}
$$
where $f_{base}$ is the simple base distribution. So $t=1$ is base, and $t=0$ is the target. Luhuan takes the notation used in diffusion.

While simple (and seemingly nice), this sometimes behaves poorly. 
- Consider $\pi_{base} = \mathcal N(0, 1)$ and $\pi = \mathcal N(\mu, 1)$. Performing this scheme will cause "mass teleportation", in the sense the modes will not move left to right, but rather grow and shrink.
- Another example is $\pi(x) \propto w_1 \mathcal N(m, 1) + w_2 \text{Unif}[0,1]$ and $\pi_{base} = \mathcal N(0,1)$. Performing this scheme will cause a "spurious mode", which is where a new mode appears between the target & base distribution's modes.

This motivates us to to construct a new annealing path, inspired by diffusion models.

## Diffusion Models
Recall, diffusion models iteratively perform $x_t = x_0 + \epsilon$ with increasing strength of noise over time, s.t.
$$
\begin{align}
\pi_{t=1} = \pi_{base} = \mathcal N(x_t ; \alpha_t x_0, \sigma_t^2 \mathbb I)
\end{align}
$$
So your time-dependent marginal distribution becomes
$$
\begin{align}
\pi_t(x_t) = \int \mathcal N(x_t ; \alpha_t x_0 , \sigma_t^2 \mathbb I) \pi(x_0)\, dx_0
\end{align}
$$
Let's inspect the probability density functions w.r.t. time. Recall Louis' said that diffusion models learn "tunnels" in path space (given sufficient intermediary steps), so there is no spurious modes and there is no mass teleportation.

### SDE Formulation
Consider the forward process
$$
\begin{align}
dX_t = f_t \, X_t \, dt + g_t \, dB_t\\
\end{align}
$$
which corresponds to the reverse process
$$
\begin{align}
dX_t = [f_t \, X_t - g^2_t \nabla \log \pi_t(X_t)]  \, dt + g_t \, dB_t
\end{align}
$$
Since $\log \pi_t(X_t)$ is a convolution of a Gaussian. The score has a closed form solution given by Tweedie's formula
$$
\nabla \log \pi_t(x_t) = \mathbb E_{x_0 \sim\pi(x_0 | x_t)} \left[ \frac{\alpha_t X_0 - x_t}{\sigma_t^2}  \right]
$$
where $\pi(x_0 | x_t) \propto \pi(x_0) \mathcal N(x_0 ; \alpha_t x_0, \sigma_t^2 \mathbb I)$.

### Monte Carlo Score Estimation
Since the score can be written as an expectation value. We can construct an estimation of it by running MCMC on it. Here's a couple ideas
- Do importance sampling, where you use a proposal as a Gaussian which scales according to your noise schedulers from diffusion.
- Or really do anything you desire...
But compared to Louis' proposal last time, we are running this estimate on the dimension of the target distribution, previous Louis was doing the sampling on the entire path space (so you got killed by high dimensions)

## Algorithm
1. Draw $x_1 \sim \mathcal N(0,1)$
2. For $t_i = t_N , ..., t_1$
	1. Run MC method targeting $\pi(x_0 | x_t) \propto \pi(x_0) \mathcal N(x_0 ; \alpha_t x_0, \sigma_t^2 \mathbb I)$ 
	2. Estimate $\mathbb E[x_0 | x_t]$ and by Tweedie's formula, this gives an estimator for the score $\hat s(x_{t_i})$.

# Week 2:
## Recap
- Our goal is to sample $\pi(x) \propto \exp(- E(x))$ where you have oracle access to it. 
- We wanted to create a new annealing path by using the path in probability space $p_t$ which diffusion models use
	- A classical annealing path is the geometric mean $\pi_t(x) \propto \pi_0(x)^{1-t} \pi_1(x)^t$

Monte Carlo Estimate of Score