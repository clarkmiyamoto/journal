- Let $p(\theta | \mathcal D)$ to be a posterior
- Let $p(\mathcal D | \theta)$ be a likelihood
- Let $p(\theta)$ be a prior

Your goal is to sample
$$
\begin{align}
p(\theta | \mathcal D) = \frac{ p(\mathcal D |\theta) p(\theta)}{Z}
\end{align}
$$
But in Stan, you only have oracle $\log p(\theta | \mathcal D$)
$$
\begin{align}
\log p(\theta | \mathcal D) = \log p(\mathcal D |\theta) + \log p(\theta) -\log Z
\end{align}
$$
```python
def log_posterior(theta: np.ndarray) -> np.ndarray:
	'''
	Args
		theta: Shape (batch, dim)
	
	Returns
		evaluation (batch)
	'''
	return log_likelihood(theta) + log_prior(theta)
```

```python
def log prior(theta: np.ndarray) -> np.ndarray:
	bound = lambda x: if x > observational_bound_upper then x = -np.infinity
	bound2 = lambda x: if x < observational_bound_lower then x = -np.infinity
	prior_bound = bound2(bound(theta))
	
	prior_mean = ...
	prior_var = ...
	
	return prior_bound + prior_mean + prior_var

```


# Specifying Likelihood

Objective is to fit a gaussian to the observed positions $y = (r_1, r_2)$ of a star. The experimentalist told us that there was measurement error on, $(\sigma_1, \sigma_2)$ The fitted parameter will be called $(\theta_1, \theta_2)$

I "believe"
$$
\begin{align}
y = (\theta_1, \theta_2) + \epsilon
\end{align}
$$
$\epsilon \sim \mathcal N(0, \text{diag}(\sigma_1^2, \sigma_2^2))$ 

# asdf
Given a column (in the sky).
You have a bunch of positions of stars $\mathcal D= \{\phi_1^{(i)}\}_{i=1}^{N_{tot}}$. You believe that $\phi_1^{(i)} \sim^{iid} \mathcal N(\mu, \sigma^2)$. 
Your objective is to figure out $\mu,\sigma^2$.

You are doing Kernel Density Estimation.

Solutions
1. EM Algorithm
2. Another is minimize the KL divergence (which is equivalent to a maximum likelihood inference)
   $$
   \text{KL}(p \|q) = \mathbb E_{x \sim p} \left[ \frac{p(x)}{q(x)} \right]
   $$
   Properties: $\text{KL} \geq 0$, and $=0$ if and only if $p=q$.
   
   In your case. $\theta = (\mu, \sigma^2)$
   $$
   \mathcal L(\theta) = - \sum_{i=1}^{N_{tot}}\frac{(\phi_1^{(i)} - \mu)^2}{\sigma^2} + \log\sqrt{(2\pi)^d} {\sigma^d}
   $$
1. Posterior sampling above..


# Model Specification
What you currently have:

Let $x_i$ denote the position of the star in the bin $i$.
$$
\begin{align}
p(x_i) = (1-\alpha) \mathcal N(x_i; \mu, \sigma^2) + \alpha \, \text{Poisson}(x_i; \lambda)
\end{align}
$$Where $\alpha \in \mathbb R$ is the strength of the background.

To do maximum likelihood inference here. You have a dataset of of observed star positions in the $i$'th bin $\mathcal D = \{x^{(j)}_i\}_j$  and you believe that $x^{(j)}_i \sim^{iid} p(x_i)$ .

You have a joint over observed star positions
(1-\alpha) \mathcal N(x_i; \mu, \sigma^2) + \alpha \, \text{Poisson}(x_i; \lambda)
$$
\begin{align}
p(x_i^{(1)}, ..., x_i^{(N_{tot})})  & = \prod_j p(x_i^j)\\
\mathcal L(\theta) = \log (...)& =  \sum_j [(1-\alpha) \mathcal N(x_i^j; \mu, \sigma^2) + \alpha \, \text{Poisson}(x_i^j; \lambda)]
\end{align}
$$
______
You want to find relative sizes of $p(x_i)$... What I mean is 
$$
\begin{align}
p(x_1, ..., x_n) \neq \prod_i p(x_i)
\end{align}
$$
$$
 p(x_1, ..., x_n) = \sum_i \alpha_i \, p(x_i)
$$s.t $\sum \alpha = 1$
