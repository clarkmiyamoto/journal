#math #talk 

# Gradient Flows
Consider the task of 
$$
\begin{align}
\min_x \mathcal F(\mu_x), \  \mu_x =\frac{1}{m} \sum_{j=1}^m \delta_{x_j}
\end{align}
$$
You can solve this using gradient flow, assume the particles follow the ODE
$$
dx/dt = \mathcal F(x)
$$
By Fokker Planck, you can show there's a PDE on the measure 
$$
\partial_t \mu_t = \nabla \cdot (\mu_t \nabla \mathcal F'[\mu_t])
$$
Let's take $\mathcal F[\mu_t] = \text{KL}(\mu_t \| \nu_t)$, you can start to think about connections betewen Wasserstien gradient flow & Reimannian geometry. Note this is a bit naive because the empirical measure is not well defined on the KL divergence.

We'll use **Otto Calculus** to mathematically inspect this.

# Min-Max Optimization
As motivation, long ago we have GANs
$$
\begin{align}
\min_{\text{Generator}} \max_{\text{Discriminator}} \mathbb E_{\hat x \sim \mathcal G} \mathbb E_{x_{ref}} \mathcal D(\hat x , x_{ref})
\end{align}
$$
Now'a'days another cool problem is distributionally robust training.

We'll work on this setting, bilinear objectives (where it is convex-concave)
$$
\begin{align}
\min_{p \in \Delta}  \max_{q \in \Delta } \sum_i \sum_j p_i q_j M_{ij}  = \min_p \max_q F(p,q)
\end{align}
$$
Now you can run the optimization using gradient flow. What's interesting is that the discretized-time vs continuous-time optimization dynamics might yield different solutions. For example consider
$$
F(p,q) = pq
$$
The equations of motion become a Harmonic oscillator. The continuous time gradient flow satisfies energy-conservation, so the phase space diagram is a circle. But if you discretize you might get blow-up or fall-in.

