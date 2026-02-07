#paper 

# Setup
Your data lives in a discrete set.

You want to construct a bijective map between discrete sets & continuous fields. This will allow the model to operate in a continuous space, where it's easier to learn / inference /etc.

**Encoding** to continuous space
Given a set $S = \{(\mathbf r_i, \mathbf x_i)\}_{i=1}^N$ of positions $\mathbf r_i \in \Omega$ and features $\mathbf x_i \in \mathbb R^d$. Let $K : \Omega \times \Omega \to \mathbb R_{\geq 0}$  be a continuous positive kernel s.t. $\alpha = \int_\Omega K(\mathbf r ; \mathbf s) \, d \mathbf r$. Our continuous fields are
$$
\begin{align}
\rho(\mathbf r) & = \frac 1 \alpha \sum_{i=1}^N K(\mathbf r, \mathbf r_i)\\
\mathbf h(\mathbf r) & =\frac 1 \alpha \sum_{i=1}^N \mathbf x_i \, K(\mathbf r, \mathbf r_i)
\end{align}
$$
1. The integration over $\rho$ gives you a counting number $$N = \int_\Omega \rho(\mathbf r) \, d \mathbf r$$

**Decoding** to discrete space
Surpringly, the encoder is bijective. However solving the inverse requires some optimization. I'll give it briefly
2. You can infer the original positions $\mathbf r_i$ via optimization. Consider the loss function $$ \mathcal L(\mathbf r_1, ..., \mathbf r_N) = \mathbb E_{r\sim \Omega} \left[ \Big( \rho(\mathbf r) - \frac{1}{\alpha} \sum_{i=1}^N K(\mathbf r ; \mathbf r_i) \Big)^2\right]$$