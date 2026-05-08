#sampling #mcmc

Adam told me about this paper about having two particles undergoing Langevin but at separate temperatures, it seems this could make for a more interesting

https://journals.aps.org/pre/pdf/10.1103/PhysRevE.92.032118

To be precise, consider the coupled Langevin SDE
$$
\begin{align}
dX^A_t & = - \nabla \log p (X_t^A, X_t^B)  + \sigma^A dW^A_t\\
dX^B_t & = - \nabla \log p (X_t^A, X_t^B)  + \sigma^B dW^B_t\\
\end{align}
$$
You choose $p$ s.t. you hope the marginal for $X_t^B$ will sample the target distribution $\pi$.

Another method is to look at $R_t = X_t^A - X_t^B$, and apparently this will converge to the same distribution but at the "mean" temperature $\bar T(\sigma^A, \sigma^B)$...

