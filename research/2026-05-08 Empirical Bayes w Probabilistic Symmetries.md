#talk #bayesianinference 

Talk by Bohan Wu
# Notes

## Empirical Bayes: G-Modeling
Consider 
$$
\begin{align}
x_i \sim p(x_i | z_i)
\end{align}
$$
You want to make an estimate of $(z_i)_i$. 

In empirical bayes, the procedure is
1.  Construct a prior $z_i \sim g$
2. Estimate $g$ from data. 
   $\hat g = \arg \max_g p(x_1, ..., x_n | g)$
3. Compute posterior for $z_i$
   $z_i  \sim p(z_i | x_i, \hat g) \propto g(z_i) p(x_i | z_i)$

We take a Bayesian approach to assume a prior. But in E

## F-Modeling
In diffusion $\hat z_i = \mathbb E[z_i | x_i] = x_i + \sigma^2 \nabla \log p(x_i)$
1. Estimate the score
2. Apply Tweedie's formula


**Ergodic Decomposition**
There exists a $\Phi$-invariant true prior $p(z)$ such that the marginal distribution
$$
\begin{align}
p(x) = \int_{\mathcal X} p(z) \prod_{\omega \in \Omega} p(x_\omega | z_\omega) dz
\end{align}
$$
is the true data-generating distribution for $x$.

**Ergodic Decomposition Theorem** say that a true prior $p(z)$ can be written as
$$
\begin{align}
p(z) = \int_{\mathcal G} p(z | g) d\mu (g)
\end{align}
$$
- Basically a $\Phi$-invariant prior admits an ergodic decomposition
- 