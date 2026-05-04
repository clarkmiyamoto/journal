#talk #optimization 

Talk by Gauthier Gidel (Universite de Montreal)
At CCM Flatiron

Paper: https://arxiv.org/pdf/2505.16098
# Notes
By scaling he's talking about reducing empirical risk given compute budget...

## Setup
Consider minimizing
$$
\begin{align}
\min_\theta \mathcal L(\theta)
\end{align}
$$
where $\mathcal L(\theta) = \mathbb E_z [\ell(\theta ; z)]$. With a large amount of features $d$ and datapoints $n$. In SGD choose a data point $z_k$ (at iteration step $k$) randomly, and then do gradient descent.

Since the risk depends on the number of total iterations $t$ and dimension $d$. The compute optimal  is given by
$$
\begin{align}
d^*(f) = \arg \min_d \mathcal R(d , f/d)
\end{align}
$$
where $f$ is the fixed compute budget, and $\mathcal R = \mathbb E_z[ \ell]$. People have empirically observed Chinchilla scaling laws, which is $d,t \sim \sqrt f$.

## Example: Linear Model
Consider a linear model
$$
\begin{align}
\mathcal R(\theta)= \mathbb E_x[(x^T W \theta - x^T b)^2]
\end{align}
$$
So $x$ is the data, $\theta,b$ is the model parameters. Assume $W_{ij} \sim \mathcal N(0, 1/d)$, and $x_j \sim \mathcal N(0, j^{-2\alpha})$ and $b_j = j^{-\beta}$.

