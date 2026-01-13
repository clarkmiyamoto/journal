#hemcee 

# Leapfrog Integrators
- Regular leapfrog: $(\tilde x, \tilde p) = \text{Leapfrog}_{t, M}(x, p)$ are the proposed steps using the leapfrog integrator for a total time $t$ with a mass matrix $M$. Where $x \in \mathbb R^d$ and $p \in \mathbb R^d$.
$$
\frac{dx}{dt} = M^{-1} \, p , \ \frac{dp}{dt}= -\nabla V(x )
$$
- Walk-move leapfrog: $(\tilde x, \tilde p) = \text{Leapfrog}_{t, B} (x, p)$, where $x \in \mathbb R^d$ and $p \in \mathbb R^{N/2}$.
	$$
	\frac{d x}{dt} = B \, p, \ \frac{dp}{dt} = - B^T \nabla V(x) 
	$$
	where $B \in \mathbb R^{d \times N/2}$ is the centered positions of the component ensemble. Note that when the ensemble reaches equilibrium, $\mathbb E[B B^T] = \Sigma$ . 

# Note on Autocorrelation
Given a sequence of random variables $X_1, X_2, ...$ the lag-k autocorrelation $\rho_k$ is given by
$$
\rho_k = \frac{\mathbb E[(X_t - \mu)(X_{t+k} -\mu)]}{\mathbb E[(X_{t} - \mu)^2]} = \frac{\text{Cov}(X_{t+k}, X_t)}{\text{Var}(X_t)}
$$where $\mu$ is the stationary mean. Note the random variables can be transformations $X_i = f(\theta_i)$ of the outputs of the MCMC algorithm.

Properties
- First is unity: $\rho_0 = 1$
- Bounded: $\rho_k \in [-1, +1]$.

The [[integrated autocorrelation time]] is given by "integrating" (summing) up all the lag-k autocorrelations.
$$
\rho = \sum_{k=-\infty}^\infty \rho_k = 1 + 2 \sum_{k=1}^\infty \rho_k
$$
# Analysis of Lag-1 Autocorreation
In the [original ChEES paper](https://proceedings.mlr.press/v130/hoffman21a/hoffman21a-supp.pdf), they analyze the lag-1 autocorrelation for a $d$-dimensional Gaussian $\mathcal N(0, \Sigma)$ where the covariance is diagonal $[\Sigma]_{ii} = \sigma^2_i$. They assume the discretization of the leapfrog integrator is negligible, allowing the analysis to be done on exact Hamiltonian dynamics (which I list above).

They are allowed to assume the Gaussian has zero mean, and diagonal components on covariance, because the ChEES criterion is invariant under shifts and rotations. 

We can still assume the Gaussian has zero mean, and only diagonal components. Because we just want to highlight the affine invariance, which can be effectively described by it's condition number $\lambda_{max}(\Sigma) / \lambda_{\min}(\Sigma)$ where $\lambda(\cdot)$ is the eigenvalue of the input.

**Outline of analysis**
We are analyzing the distribution
$$
p = \mathcal N(0, \Sigma), \ [\Sigma]_{ii} = \sigma_i^2
$$
Meaning the potential $V(x)$ (aka negative log-likelihood without normalization constant) is given by
$$
V(x) = \frac{1}{2}x^T \Sigma^{-1} x \implies \nabla V(x) = \Sigma^{-1} x
$$
In the lag-1 autocorrelation, let $X_{t+1}$ to be the output after using leapfrog dynamics for total time $T = \epsilon L$ (in the limit where the discretization error $\epsilon \to 0$ holding $\epsilon L$ constant, allowing us to use exact Hamiltonian dynamics).
1. Calculate the equations of motion. That is $(\tilde x, \tilde p) = \text{Leapfrog}(x, p)$
2. You can directly calculate the lag-1 autocorrelation.
## Regular Leapfrog
The equations of motion are
$$
\frac{dx}{dt} = M^{-1} \, p , \ \frac{dp}{dt}=  \Sigma^{-1}x
$$

Let's do something simple where $M = \text{diag}(m_1, m_2)$ and $\Sigma = \text{diag}(\sigma_1^2, \sigma_2^2)$. The solution becomes $$
\begin{align}
\tilde x & = x_i(t) = x_i(0) \cos(\omega_i t) + \frac{p_i(0)}{m_i \omega_i} \sin(\omega_i  t)\\
\tilde p & = p_i(t) = p_i(0) \cos(\omega_i t) - x_i(0) \, m_i \omega_i  \, \sin(\omega_i t)
\end{align}
$$where $\omega_i = 1/(\sigma_i \sqrt{m_i})$. 
### First Moment's Lag-1 Autocorrelation
Consider the first moment's autocorrelation
$$\rho^{(1)}= \frac{\mathbb E[(x - \mathbb E[x])(\tilde x -\mathbb E[x])]}{\mathbb E[(x - \mathbb E[x])^2]}$$
Assuming the chain is already at stationarity, $\mathbb E[x_i] = 0$ and $\mathbb E[x_i x_j] = \sigma_i^2 \delta_{ij}$. 

The only weird thing to calculate is $\mathbb E[\tilde x_i x_i]$ $$
\begin{align}
 \mathbb E[\tilde x_i x_i] & = \mathbb E[x_i x_i \cos(\omega_i t)] & x_i(0),p_i(0) \text{ are independent random variables}\\
 & = \sigma_i^2 \cos(\omega_i \, t)
\end{align}
$$Therefore the first moment's lag-1 autocorrelation 
$$
\rho_1 = \cos(\omega_i t)
$$

### Second Moment's Lag-1 Autocorrelation
Take $X = f(x)$, where $f(x) = x^2$.
$$
\rho^{(2)}= \frac{\mathbb E[(x^2 - \mathbb E[x^2])(\tilde x^2 -\mathbb E[x^2])]}{\mathbb E[(x^2 - \mathbb E[x^2])^2]}
$$
Assuming the chain is already at stationarity, $\mathbb E[x_i^2] = \sigma_i^2$ and $\mathbb E[x_i^4] = 6 \sigma_i^4$.

The denominator becomes $$
\begin{align}
\mathbb E[(x_i^2 - \mathbb E[x_i^2])^2] & = \mathbb E[x_i^4] - \mathbb E[x_i^2]^2\\
& = 5 \sigma_i^4
\end{align}
$$
The numerator 


## Walk-Move Leapfrog
The equations of motion
$$
	\frac{d x}{dt} = B \, p, \ \frac{dp}{dt} = - B^T \Sigma^{-1} x
$$
where $B \in \mathbb R^{d \times N/2}$ 

