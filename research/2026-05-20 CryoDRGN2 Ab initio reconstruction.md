#scsi 

- Single volume $V : \mathbb R^3 \to \mathbb R$
- Result of projection $\hat X : \mathbb R^2 \to \mathbb R$ in fourier domain
	- $\hat X(k_x, k_y) = \hat g S(t) \hat V(R^T(k_x, k_y, 0)^T) + \epsilon$
	- where $\hat V: \mathbb R^3 \to \mathbb R$ is electron scattering potential, $R \in \text{SO}(3)$, $S(t)$ is phase shift operator (where $t \in R^2$ is a shift on the image), and $\hat g$ is the contrast transfer function

Probability of observing an image $\hat X$ with pose $\phi = (R,t)$ from volume $\hat V$ is a Gaussian
$$
\begin{align}
p(\hat X | R, t, \hat V) \propto \exp \left( \sum_\ell \frac{-1}{2\sigma_\ell^2}  \left | \hat g_\ell A_\ell(R) \hat V -  S_\ell(t) \hat X_\ell \right |\right)
\end{align}
$$
where $A$ is the linear slice operator.