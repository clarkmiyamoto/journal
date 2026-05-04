#math #learningtheory 

Lecture notes from Misiakie

# Notes
Consider the supervised learning problem. You are given
- Data: $\{(y_i, x_i)\}_{i \leq n}$, where $y \in \mathbb R$ and $x \in \mathbb R^{d}$ and they're drawn iid from $\mathbb P$
- Loss function: $\ell : \mathbb R \times \mathbb R \to \mathbb R$
Your goal is to fit a predictor $\hat f: \mathbb R^d \to \mathbb R$ that has small **Expected/Population/Test Risk** 
$$
\begin{align}
R(\hat f) = \mathbb E_{(y,x) \sim \mathbb P}[\ell(y, \hat f(x))]
\end{align}
$$
In practice you can't train directly on the test-risk, you have access to the **empirical risk**
$$
\begin{align}
\hat R_n(f) := \frac{1}{n} \sum_{i\leq n} \ell(y_i, f(x_i))
\end{align}
$$
By doing **empirical risk minimization (ERM)**, you want to find the best model $\hat f$ which minimizes the ER within a model class $\mathcal F$
$$
\begin{align}
\hat f_n = \arg \min_f \{\hat R_n(f) : f \in \mathcal F\}
\end{align}
$$
## Uniform Convergence
An intuitive idea is if $\epsilon_n(F) = \sup_{f \in F} |\hat R_n (f) - R(f)|$ is small, then $R(\hat f_n) \approx \inf_{f \in F} R(f)$. In plain English, if the possible generalization error (given the model class $F$) is small, then empirical risk minimization will get you close to the true solution.

*Proof:* Let $f_* = \arg \min_{f \in F} R(f)$. Decompose the test-risk (evaluated on the ERM)
$$
\begin{align}
R(\hat f_n) = R(f_*) + [R(\hat f_n) - \hat R_n(\hat f_n)] + [\hat R_n(\hat f_n) - \hat R_n(f_*)] + [\hat R_n(f_*) - R(f_*)]
\end{align}
$$
Notice $\hat R_n (\hat f_n)$ is a minimized quantity, which means $\hat R_n(\hat f_n) - \hat R_n(f_*) \leq 0$, applying this to the decompositions yields
$$
\begin{align}
R(\hat f_n) & \leq R(f_*) + [R(\hat f_n) - \hat R_n(\hat f_n)] + [\hat R_n(f_*) - R(f_*)]\\
& \leq R(f_*) + |R(\hat f_n) - \hat R_n(\hat f_n)| + |\hat R_n(f_*) - R(f_*)|\\
& \leq R(f_*) + \sup_f |R(f ) - \hat R_n(f)| + \sup_f |\hat R_n(f) - R(f)| & \text{Def of} \sup\\
& = R(f_*)  + 2 \sup_f |\hat R_n(f) - R(f ) |
\end{align}
$$
$\blacksquare$   

Remark: $$
\begin{align}
\hat R_n(f) - R(f) = \frac{1}{n} \sum_{i=1}^n \ell(y_i, f(x_i)) - \mathbb E[\ell(y, f(x))] \to^d \mathcal N\left(0, \frac{C}{n} \right)
\end{align}
$$

## Tools to Bound $\epsilon_n(F)$

**Definition (Rademacher Complexity)** Let $(Z,P)$ be a probability space. Let $F$ be a class of functions over $Z$. The Rademacher complexity 
$$
\begin{align}
\text{Rd}_n(G) := \mathbb E_\sigma\left[ \sup_{g \in G} \Big |\frac{1}{n} \sum_{i=1}^n \sigma_i \, g(z_i) \Big | \right]
\end{align}
$$where $z_1, ..., z_n \sim_{iid} P$ and $\sigma_1,..., \sigma_n \sim_{iid} \text{Unif}(\{\pm 1\})$.

In our case, 
$$
\begin{align}
z_i & = (y_i, x_i)\\
g(z_i) & = \ell(y_i, f(x_i))\\
G & = \ell \circ F
\end{align}
$$
**Lemma**
$$
\begin{align}
\frac 1 2 \text{Rd}_n(G) \leq \mathbb E \left[ \sup_{g \in G} \left| \frac{1}{n} \sum_{i=1}^n g(z_i) - \mathbb E \, g \right| \right] \leq 2 \, \text{Rd}_n(G)
\end{align}
$$
*Proof:* 
$$
\begin{align}
\mathbb E_z \left[ \sup_{g \in G} \left| \frac{1}{n} \sum_{i=1}^n g(z_i) - \mathbb E \, g \right| \right] & = \mathbb E_z \left[ \sup_{g \in G} \left| \mathbb E_{z'} \left (\frac{1}{n} \sum_{i=1}^n g(z_i) -   g(z'_i) \right)\right| \right]\\
&\leq \mathbb E_{z,z'}\left[ \sup_{g \in G} \left|  \frac{1}{n} \sum_{i=1}^n g(z_i) -   g(z'_i)\right| \right] & \text{Jensen's}\\
& 
\end{align}
$$
Because $z,z'$ are drawn from the same distribution iid, $D := g(z_i) - g(z'_i) =^d  g(z'_i) - g(z_i)$. Therefore $(D + -D)/2 =^d D$, so we can write as an expectation value over Rademacher variables $D= \mathbb E_\sigma [\sigma D]$.
$$
\begin{align}
& = \mathbb E_{z,z'} \left[ \sup_{g \in G} \left|  \frac{1}{n} \mathbb E_\sigma \sum_{i=1}^n (g(z_i) -   g(z'_i)) \sigma\right| \right]\\
& \leq \mathbb E_{z,z', \sigma} \left[ \sup_{g \in G} \left|  \frac{1}{n} \mathbb  \sum_{i=1}^n (g(z_i) -   g(z'_i)) \sigma_i\right| \right] & \text{Jensen} \\
& = \mathbb E_{z,z', \sigma} \sup_g \left| \frac{1}{n} \sum_{i=1}^n g(z_i) \sigma_i - \frac{1}{n} \sum_{i=1}^n g(z'_i) \sigma_i \right|\\
& \leq 2 \,\text{Rd}_n(G) & \text{Triangle}
\end{align}
$$
$\blacksquare$

**Remark:** From the uniform convergence bound, this implies
$$
\begin{align}
\mathbb E[R(\hat f_n)] \leq \inf_{f \in F} R(f) + 2 \, \text{Rd}_n(\ell \circ F)
\end{align}
$$
**Lemma (Connection Inequality)** Let $\phi_1, ..., \phi_n: \mathbb R \to \mathbb R$ be $\lambda$-Lipschitz functions, such that $\phi_i(0) = 0$
$$
\begin{align}
\mathbb E_{z, \delta} \left[ \sup_{f\in F} \left| \frac 1 n \sum_{i=1}^n \phi_i(f(z_i)) \sigma_i \right| \right] \leq \lambda \,  \mathbb E_{z,\sigma} \left[ \sup_{f\in F} \left | \frac{1}{n} \sum_{i=1}^n f(z_i) \sigma_i   \right | \right]
\end{align}
$$
*Proof:* By Lipschitz-ness
$$
\begin{align}
|\phi_i(f(z_i)) - \phi_i(f(z'_i))| & \leq  \lambda \, |f(z_i) - f(z'_i)|\\
\mathbb E_{z,z'}[|\phi_i(f(z_i)) - \phi_i(f(z'_i))|] & \leq \lambda  \mathbb E_{z,z'} |f(z_i) - f(z'_i)|
\end{align}
$$
Then you can do the symmetrization trick
$$
\begin{align}
\mathbb E_{z,\sigma} |\phi_i(f(z_i)) \sigma_i| \leq \lambda \mathbb E_{z,\sigma} |f(z_i)|
\end{align}
$$
This then generalizes for the sum of multiple functions. Let $\phi = \sum_i \phi_i$
$$
\begin{align}
\left|\frac{1}{n}\sum_{i=1}^n \phi_i(f(z_i)) - \phi_i(f(z'_i))\right| 
\end{align}
$$
$\blacksquare$

We can use this to convert the $\text{Rd}_n(\ell \circ F)$ to $\text{Rd}(F)$. 

**Fact**
$$
\begin{align}
\mathbb E[R(\hat f_n)] \leq \inf_{f \in F} R(f) + \frac{C}{\sqrt n} \mathbb E[\ell(y, 0)^2]^{n/2} + C L \cdot \text{Rd}_n(F)
\end{align}
$$


## Example: $F$ is 2 layer neural networks
$$
\begin{align}
f(x;\theta) = \frac{1}{M} \sum_{j=1}^M a_j \sigma(\langle w_j, x\rangle)
\end{align}
$$
where $x \in \mathbb R^d$ and $\theta =\{(a_j, w_j)\}_{j \leq m}$. Assume $x \in B_2^d(C_x\sqrt{d})$ which is a ball of radius $C_x \sqrt d$.

**Result**
$$
\begin{align}
\text{Rd}_n(\ell \circ F) \leq \frac{C}{\sqrt n} + L \cdot L_\sigma \cdot \pi_0 C_w C_x \sqrt{\frac{d}{n}}
\end{align}
$$
Some remarks
- The bound is independent of $M$. Which implies neural networks can generalize even with $M \to \infty$, as long as $\| a \|_p \ll \sqrt{n/d}$.
- Independent of $p$