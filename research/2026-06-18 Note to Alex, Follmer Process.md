#stochasticinterpolants 

In the original Follmer Process paper, Yifan uses the interpolant
$$
\begin{align}
I_t = \alpha_t X_0 + \beta_t X_1 + \sigma_t \, W_t
\end{align}
$$
Then solving $dX_t = b_t(X_t | x_0) \, dt$ (with boundary condition $X_0 = x_0$) s.t. the drift field is found from
$$
\begin{align}
\mathcal L[\hat b] & = \int_0^1 \mathbb E[|\hat b_t(I_t, x_0) - R_t|^2] \, dt\\
\text{where } R_t & = \dot \alpha_t X_0 + \dot \beta_t X_1 + \dot \sigma_t W_t
\end{align}
$$
Causes the solutions to the ODE to have $\text{Law}(X_t ) = \text{Law}(I_t | X_0)$. In particular $X_{t=1} \sim \rho_c(\cdot |x_0)$ 

In your case $X_0, X_1$ are positions of some robot, and $t$ denotes both the generative model's time & physical time. So a interpolation between the end points is insufficient, there is plenty of inbetween data from your robot's trajectory.

Imagine between time $t \in [0,1]$, you robot actually traces out $\tilde X_t$ (trajectory in position space). So you more likely want to use the interpolant
$$
\begin{align}
I_t = \tilde X_t + \sigma_t W_t
\end{align}
$$
where the drift field is
$$
\begin{align}
\mathcal L[\hat b] & = \int_0^1 \mathbb E[|\hat b(I_t, x_0) - R_t|^2] dt\\
\text{where } R_t & = \partial_t \tilde X_t + \dot \sigma_t W_t
\end{align}
$$
