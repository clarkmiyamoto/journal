#generative #architecture 

# Setup
## Traditional
Consider the stochastic interpolant
$$
\begin{align}
I_t = \alpha_t \, x + \beta_t F(x)
\end{align}
$$
where $F(x) = x + w$, where $x \sim \pi_{\text{data}}$ (e.g. MNIST data), and $w \sim \mathcal N(0,1)$. All objects live in $\mathcal X$

The optimal drift field is given by
$$
\begin{align}
b_t^{\textsf{opt}}(x) = \mathbb E[\dot I_t | I_t = x]
\end{align}
$$
Which can be found by minimizing
$$
\begin{align}
\mathcal L(\hat b) = \int_0^1 \mathbb E \Big[ \left( \hat b_t(I_t) - \dot I_t \right)^2 \Big]
\end{align}
$$

## Lifting
We can also get this generation by learning a guided model
$$
\begin{align}
I_t | x = \alpha_t \, z + \beta_t \, F(x)
\end{align}
$$
where $z \sim \mathcal N(0, 1)$. That is I want to learn the transport map, to sample $F(x) | x$. In this case, the optimal drift field is given by
$$
\begin{align}
\mathcal L(\hat b) = \int_0^1 \mathbb E \Big[ \left( \hat b_t(I_t | x) - \dot I_t \right)^2 \Big]
\end{align}
$$
where $x,y \in \mathcal X$. The expectation value is taken over $z,x$ which are sampled IID.

# Guidance (Conditioning)
When we learn conditional (but the generative model field prefers guided) drift fields $b_t(x|y)$ (where $x \in \mathcal X$ and $y \in \mathcal Y$), we have to think of an architecture which actually supports this behavior.

In our case, I want to learn how to sample $F(x) | x$, and luckily $F: \mathcal X \to \mathcal X$, so our guidance simplifies to $\mathcal Y = \mathcal X$.

$$
\begin{align}
b_t(x | y)
\end{align}
$$


# Architectures

