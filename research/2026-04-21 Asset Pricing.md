
You observe market prices $q_t$. Assume you know the distribution of the payoff random variable, that is $\text{Law}(x_{t+1})$.   

Our objective is to construct a pricing functional $P_t : L^2 \to \mathbb R$ such that $p_t(x_{i, t+1}) = q_{i,t}$  (where $i$ is the $i$'th stock).

Some axioms
- $p_t$ should be linear. The price of two individual stocks, is equal to the price of that stock twice. 
- No arbitrage. Let $x_{t+1} > 0$ such that $\mathbb P(X_{t+1} \geq 0) = 1$ and $\mathbb P(X_{t+1} > 0 ) > 0$. Pay off is always non-negative and pay off is true w/ the prob
- Boundedness. $p_t$ is bounded.

An equivalent way is to not look at prices but look moments.

Let $M_{t+1}$ be the stochastic discount factor. We call it a SDF if $\forall i$ (assets) 
$$
\begin{align}
q_{i,t} = \mathbb E_t[M_{t+1} X_{i, t+1} | \mathcal F_t]
\end{align}
$$

**Theorem** If $p_t$ prices all assets $i$ and satisfies no-arbitrage, then there exists $M_{t+1}$ s.t. the equation above holds.
*Proof* you can prove this using Riesz representation theorem

**Question** what is the stochastic discount factor? Consider the total payoff
$$
\begin{align}
U(C) = \mathbb E_0[u(C_0) + \beta u(C_1)]
\end{align}
$$
where $\beta < 1$ and $u$ is concave. How should we allocate wealth? This comes down to a Lagrangian optimizaiton,
$$
\begin{align}
\implies \mathbb E_0 [\beta \frac{W_0 u'(C_1)}{u'(C_0)} X_{i, t + 1}] = 1, \forall i \in \left\{ 1,2 \right\}
\end{align}
$$
 If you identify $\beta W_0 u'(C_1) / u'(C_0)$ in the SDF equation, it seems that that is $M_t$!

Now can we predict returns? Consider excess returns $R_{t, t+1}^e := R_{i, t+1} - R_t^f$. My pricing function $p_t(X_{i, t+1}) = q_{i,t}$. I know that $p_t(R_{i, t+1} ) = 1$ and thus $p_t(R^e_{i, t+1}) = 0$ (due to linearity). This gives us $\mathbb E_t [M_{t+1} R_{i,t+1}^e] = 0$

Now you can do the decomposition
$$
\begin{align}
\mathbb E_t[R^e_{i,t+1}] = -\frac{\text{Cov}(M_{t+1}, R^e_{i, t+1})}{\mathbb E_t[M_{t+1}]}
\end{align}
$$
Wow so you can know the returns if this $M_t$ is time varying!

**Example**
Consider the factor structure
$$
\begin{align}
p(r_{t+1} | r^T) = \int p(r_{t+1} | f_{t+1}) \int p(f_{t+1} | f_t) p(f_t | r^T) \, df_t df_{t+1}
\end{align}
$$
notice the first term acts like a decoder, middle term is a prior, and the right turn is an encoder. So if you assume a VAE structure
$$
\begin{align}
f_{t+1} | f_t & \sim \mathcal N(\mu_\phi(f_t), \Sigma_\phi(f_t))\\
r_{t+1} | f_{t+1, z_t} & \sim \mathcal N(\beta_\theta(z_t) f_{t+1} , \Sigma_{\theta}(f_{t+1}, z_t))
\end{align}
$$
So now you can compute the expected return
$$
\begin{align}
\mathbb  E[r_{t+1}] & = \mathbb E[\mathbb E[r_{t+1} | f_{t+1} , \mathcal F_t] | \mathcal F_t]\\
& = \beta_\theta(z_t) \mu_\phi(f_t)
\end{align}
$$
But this must be consistent with our 
$$
\begin{align}
\mathbb E[r_{t+1}] = - \frac{\text{Cov}(m_{t+1} r_{t+1})}{\mathbb E[m_{t+1}]}
\end{align}
$$
So the question is what is the $m_{t+1}$ structure which is consistent with this? Make hte ansatz $m_{t+1} = 1- \lambda'_t u_{t+1}$ where $u_{t+1}$ is the noise in the reparameterization trick in $f_{t+1} | f_t$. Asantz the ansewr as
$$
\begin{align}
\mathbb E[r_{t+1}] := \beta_t \, F_t
\end{align}
$$
where $\beta_t = \text{Cov}(r, u) \text{Var}(u)^{-1}$ and $F_t = \text{Var}(u) \lambda_t$
