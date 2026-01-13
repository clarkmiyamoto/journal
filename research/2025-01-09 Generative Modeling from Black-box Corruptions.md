#paper #generative #bayesianinference #flowmatching

**Title:** Generative Modeling from Black-box Corruptions via Self-Consistent Stochastic Interpolants
**Authors:** Chirag Modi, Jiequn Han, Eric Vanden-Eijnden, Joan Bruna
**Link:** https://arxiv.org/pdf/2512.10857

# Technical Notes

## Motivation
Assume the pov of a scientist.

Consider an experiment which produce $y_1, ..., y_n \sim^{iid} \mu$, each arising from an unobserved signal $x_1, ..., x_n \sim^{iid} \pi$. You have a simulator $\mathcal F$, that is $y = \mathcal F(x)$, which can be viewed as sampling from $p(y|x)$.  Your goal is to infer is to infer the signal $x$ from a specific experiment $y$, that is to sample the posterior $p(x | y)$. This requires knowledge of a prior

Denoting $\mathcal K$ as the integral operator (or the pushforward through the simulator)
$$
\begin{align}
\mathcal K & : \mathcal P(\mathcal X) \to \mathcal P(\mathcal Y)\\
(\mathcal K \pi)(\cdot) & = \int p(\cdot | x) \, d\pi(x)
\end{align}
$$
Our desired solution is such that $\mathcal K \pi = \mu$

## Problem Setup
Consider a probability distribution $\pi \in \mathcal P(\mathcal X)$ and a forward model $\mathcal F: \mathcal X \to \mathcal X$ (which can be stochastic). At the level of the samples $y = \mathcal F(x)$, the function is non-invertible. However at the level of the distribution $\mathcal P(X)$, that is associate an integral mapping $\mathcal K$ with $\mathcal F$, as soon as $\mathcal K$ is injective ($\mathcal K \pi = \mathcal K \tilde \pi \iff \pi = \tilde \pi$) you can hope to recover $\mu$ from $\pi$ by inverting $\mathcal K$.

Assuming we have sampling access to $\mu$ (experiment), our goal is to find a transport map $\hat \Phi$ that pushes $\mu$ (experiment) to a distribution $\hat \pi = \hat \Phi \mu$ (unobserved signal) which satisfies
$$ \mathcal K \hat \pi = \mathcal K \hat \Phi_\# \mu = \mu \tag{1}$$
In plain English:
- Left: Using the stochastic interpolant's prediction of unobserved signals $\hat \pi$, put it through the simulator $\mathcal K$.
- Middle: , put all of that through the simulator $\mathcal K$.
- Right: This is the real observed experimental distribution $\mu$.
should all be equal to each other!

## Proposed Solution: Self-Consistent Stochastic Interpolants
### Review of Stochastic Interpolants
Let $\pi$ be the clean data distribution, $\mu$ be the corrupted  data distribution that is available to us.

The stochastic interpolant between $p_{t=0} = \pi$ and $p_{t=1} = \mu$ is defined as
$$
I_t  = \alpha_t x_0 + \beta_t x_1 + \gamma_t z, \ t \in [0,1] \tag{2}
$$
where $(x_0, x_1) \sim \nu(x_0, x_1)$ (a coupling distribution) that marginalizes to $\int \nu \, dx_0 = \mu$ and $\int \nu \, dx_1 = \pi$. $z$ is independent Gaussian noise. The schedules satisfy the boundary conditions $\alpha_0 = \beta_1 = 1, \alpha_1 = \beta_0 = \gamma_0 = \gamma_1 = 0$. 

In the Fokker Planck sense, you can write a drift $b^{target}$ & diffusion $\sigma^{target}$ field s.t. $\text{Law}(I_t) = p_t$
$$\partial_t p_t = - \nabla \cdot (( b^{target} + \epsilon_t \gamma^{-1}_t g^{target}  ) \,  p_t)  +  \epsilon_t \nabla^2 (p_t)$$
where
$$
\begin{align}
b^{target}_t(x) & = \mathbb E[\dot I_t | I_t = x]\\
g^{target}_t(x) & = \mathbb E[z | I_t = x]
\end{align}
$$
We define the PDE on $p_t$ as a pushforward on the densities $\pi_\Theta := (\Phi_\Theta)_{\#} \mu$ (denote the model generically as $\Theta$).  We can then use the ODE/SDE formulation to transport samples. 
$$
\begin{align}
dX_t^B = 
\Big(b_t(X_t^B) + \epsilon_t \gamma^{-1}_t \, g_t(X_t^B) \Big) \, dt + \sqrt{2 \epsilon_t} \, dW_t^B
\end{align}

$$
Let $\Phi_\Theta$ be the backwards transport map induced by $\Theta$; that is $\Phi_\Theta(X_1) = X_0$. 

### Self-Consistency Modification
The interpolant in (Eq 2) defines a map between $\pi$ (signal, target) and $\mu$ (experiment). However, we don't have access to $\pi$.

We can instead modify the interpolant using the simulator $\mathcal F$
$$
I_t = \alpha_t x + \beta_t \mathcal F(x) + \gamma_t z, \ t \in[0,1]
$$
where $x \sim \pi$. Note this still defines a valid interpolation between $\pi$ and $\mu$.

Note the backwards transport map $\Phi^*$ computes $\Phi^*_\# \mu = \pi$. We can also note that we can use this to define the self-consistency condition (Eq 1)

We can attempt to iteratively enforce this by constructing
$$
I_t^{(k+1)} = \alpha_t \Phi_{\Theta^{(k)}}(y)+ \beta_t \mathcal F(\Phi_{\Theta^{(k)}}(y)) + \gamma_t z, \ y \sim \mu
$$
where $

We get the following Algorithm
1. Initialize $\Theta \leftarrow \Theta^{(0)}$
2. for $k \in 1,..., K$ do
	1. for $i \in 1,..., T_{tr}$ do
		1. Sample $I_t$ with $\Theta^{(k)}$
		2. SGD update $\Theta$ using regular losses
	2. Update transport $\Theta^{(k)} \leftarrow \Theta$
3. Return $\Theta^{(K)}$

## Theoretical Notes on the Algorithm
Some notation $\| b\|_{\pi[0,1]}^2 \equiv \mathbb E_{t \sim \text{Unif}[0,1]} \mathbb E_{x \sim \pi_t} [\|b \|^2]$  

 Some facts
 - Convergence in Wasserstein $$W_2^2(\pi, \pi^{(k)}) \leq R^2 W_2^2 (\pi, \pi^{(k)})$$
	 where $\pi^{(k+1)} = \Phi^{(k)}_\# \mu$  