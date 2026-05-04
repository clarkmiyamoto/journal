#talk #generative 
Talk by Sara
Papers
- Categorical Flow Maps
- One Step Language Modeling via Continuous Denoising
# Notes
## Recap of current techniques
Let $x = (x_1,..., x_n)$
- Autoregressively, you sample $p(x) = p(x_1) \prod_{i} p(x_{i+1} | p(x_i))$
- Maksed discrete diffusion, you sample an unmasking sequence $[i] \sim p(I)$  and then autoregressively sample in order of the sequence $p(x) = p(x_{[1]}) \prod_{[i] \in I} p(x_i | x_{\partial [i]})$ 
- Uniform diffusion, you directly sample the joint $p(x)$ 

## Continuous Time Flow
This time, we'll do the the generation in continuous space. Most current methods use discrete diffusion.

## Review of Stochastic Interpolant
Consider $x_0 \sim p_0 = \mathcal N(0, \mathbb I)$ and $x_1 \sim p_1$ (data). Use the linear interpolant
$$
\begin{align}
I_t = (1-t) x_0 + t x_1
\end{align}
$$
let $\text{Law}(I_t) = p_t$. For ODE inference
$$
\begin{align}
\dot x_t & = b_t(x_t), \qquad x_0 \sim p_0\\
b_t(x) & = \mathbb E[\dot I_t | I_t = x]
\end{align}
$$
Because it's a conditional expectation
$$
\begin{align}
\mathcal L(\hat b) = \int_0^1 \mathbb E[|b_t(x) - \hat b_t(x)] \,dt
\end{align}
$$

## Reparameterization on Simplex
Instead of learning $b_t$ learn the expected endpoints
$$
\begin{align}
D_t(x) = \mathbb E[x_1 | I_t = x], \qquad b_t(x) = \frac{D_t(x) - x}{1 - t}
\end{align}
$$
This is a linearization of the flow path, so it will sample a biased distribution at the end.

Once again it's a conditional expectation, so the loss function is a MSE. 

What's nice is that since $D_t(x)$ is the end-point location, it lies in the probability simplex, so you can use softmax on it, forcing it to behave like a discrete process.
## Flow Map
To take care of the bias, consider a flow map
$$
\begin{align}
X_{s,t}(x_s) = x_t
\end{align}
$$
You can parameterize it in this way
$$
\begin{align}
X_{s,t}(x) = x + (t-s) v_{s,t}(x)
\end{align}
$$
In loss function you decompose it into two terms
- Off diagonal term $X_{s,t}$
- and the on-diagonal $\lim_{s \to t} X_{s,t} = b_t$

![[2026-03-31 Denoising drift path.png]]
