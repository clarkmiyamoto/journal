#talk #generative #sampling 
Talk by Mark Goldstein
Paper: https://arxiv.org/abs/2402.06121
# Setup
- You want to sample $$\mu(x) \propto \exp(- E(x))$$
  and have oracle access to $E(x)$
- Want to compute $\mathbb E_\mu [f]$ for all test functions $f$

# Notes
Here's your diffusion model setup. Set $p_{t=0} = \mu$, and the forward process is given by OU
$$
\begin{align}
dX_t = - \alpha_t X_t dt + g_t dW_t
\end{align}
$$

- $\alpha_t = 0$, gives the Heat-semigroup. Which corresponds to the Variance-Exploding process
- $\alpha_t = g_t^2 /2$ finite gives the OU-semigroup. Which corresponds to the Variance-Preserving process.
- Linear SDE implies that $X_t | X_0$ is conditionally Gaussian

By Fokker Planck
$$
\begin{align}
\partial_t p_t = - \nabla \cdot (-\alpha_t x \, p_t) + \frac{g_t^2}{2} \Delta p_t
\end{align}
$$
where $\text{Law}(X_t) = p_t$.

From there you can write the Reverse SDE
$$
\begin{align}
dX_t = [-\alpha_t X_t - g^2_t \nabla \log p_t(X_t)] dt + g_t dW_t
\end{align}
$$
This is useful because
$$
\begin{align}
\nabla \log p_t(x) = \mathbb E_{p(X_0 | X_t)}[\nabla \log p_t(X_t = x | X_0)]
\end{align}
$$
- The integral is not known
- But the integrand is known
- We can use the best function to predict random variables in mean-squared error, we can rewrite this expectation value as a minimization.
$$
\begin{align}
= \min_\theta \mathbb E_{X_0, X_t} [\| s_\theta(X_t, t) - \frac{X_0 - X_t}{\sigma_t^2} \|^2] 
\end{align}
$$
But if you could compute $\mathbb E_{X_0}$ then you would have been done already (as that was our original task). So the paper gives an estimator for this expectation value.

So let's derive it. *Assume the variance exploding process*
$$
\begin{align}
p_t & = p_0 * N(x_0, \sigma_t^2)\\
\nabla \log p_t(x) & = \nabla p_t(x) / p(x)\\
& = \frac{\nabla(p_0 * \mathcal N(0, \sigma_t^2))(x)}{p_t(x)}\\
& = \frac{\mathbb E_{z \sim \mathcal N(x, \sigma^2_t)}[\nabla p_0(z)]}{\mathbb E_{z \sim \mathcal N(x, \sigma_t^2)}[p_0(z)]}\\
& \boxed{= \frac{\mathbb E_{z \sim \mathcal N(x, \sigma^2_t)}[\nabla \exp(-E(z))]}{\mathbb E_{z \sim \mathcal N(x, \sigma_t^2)}[\exp(- E(z))]}}
\end{align}
$$
*Proof (of the 2nd line):*
$$
\begin{align}
p_t(x_t) & = \int p_0(x) \mathcal N(x_t ; x,\sigma_t^2) dx\\
\nabla_{x_t} p_t(x_t)& = \nabla_{x_t}\int p_0(x_t - y) \mathcal N(y) \, dy \\
& = \mathbb E_{y \sim \mathcal N(x_t,\sigma_t^2)}  [\nabla p_0(y)]
\end{align}
$$
where I made the substitution $y_t = x_t - x_0$ (on the integral's measure). The denominator follows by our definition of the variance exploding process.

You can now estimate this by sampling $z \sim N(x, \sigma_t^2)$, and then you can write an empirical average
$$
\begin{align}
\nabla \log p_t(x) = \nabla_{x} \log \sum_i \exp(- E(x_i))
\end{align}
$$
where $x_i = x + \sigma_t z$.

*Aside (reprameterization trick)*
If you have $\nabla_\mu \mathbb E_x[f(x)]$ where $x \sim \mathcal N(\mu, 1)$. You can always compute this by just rewriting the expectation value to expose the $\mu$. $\nabla_\mu \mathbb E[f(z + \mu)]$ where $z \sim \mathcal N(0, 1)$. 

This is a consistent estimator!


