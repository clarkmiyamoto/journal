#talk #deeplearning

Presenter: Nikhil Ghosh
# Notes
Let $n$ be the scaling dimension (e.g. model width) The scaled HPs are defined as 
$$
\begin{align}
\mathcal H_n(\nu,\gamma) = (\nu_1 n^{-\gamma_1}, ..., \nu_h n^{-\gamma_h})
\end{align}
$$
We refer to $\nu$ as the HP's, which are now scale-invariant

As the scale $n \to \infty$, we desire
1. Converges to a well-defined limit
2. Such convergence is "sufficiently fast" (s.t. we can infer the properties of large neural networks from smaller neural networks)

**Questions:** When and why do hyperparameters converge sufficiently fast w/ scale?
q
