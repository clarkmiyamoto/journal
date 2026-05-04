#stochasticinterpolants #paper #scsi

Paper: https://arxiv.org/pdf/2406.07507

# Notes
Running generative models requires a lot of compute at inference time. Here's a principled framework to reduce the time required.

First assert your stochastic interpolant
$$
\begin{align}
I_t = \alpha_t x_0 + \beta_t x_1 + \gamma_t z
\end{align}
$$
Usually you perform inference by integrating
$$
\begin{align}
dX_t = b_t(X_t) dt, \qquad b_t(x) = \mathbb E[\dot I_t | I_t =x]
\end{align}
$$
Which usually requires ~64 (* 3, if you use RK3 integrator) queries of $b_t(x)$

Instead construct the flow map, defined by
$$
\begin{align}
X_{s,t}(x_s) = x_t
\end{align}
$$
where $x_s$ is the solution to the ODE above. Since it's a solution, $\partial_s x_s = b_s(x_s)$, you can derive a time evolution on $X_{s,t}$
$$
\begin{align}
\partial_t X_{s,t}(x) & = b_t(X_{s,t}(x)) & \text{Lagrangian}\\
X_{t,\tau}(X_{s,t}(x)) & = X_{s,\tau}(x) & \text{Semigroup}\\
\frac{d}{ds} X_{s,t} (x_s)& =  \partial_s X_{s,t}(x_s) + b_s (x_s) & \text{Eulerian}
\end{align}
$$



# Consistency Model on SCSI
```pseudo
\begin{algorithm}
\caption{Lifted SCSI EM with Conditional Flow Map E-step}
\begin{algorithmic}
\Require Corruption operator $F$, drift models $\theta_{\mathrm{em}}, \theta_{\mathrm{or}}$, flow map model $\phi_{\mathrm{cm}}$, outer iterations $K$, inner iterations $N$, CM steps $M$, teacher batch size $B'$, batch size $B$
\State $\phi_0 \leftarrow \theta_{\mathrm{em}}$ \Comment{Initialize frozen generator}
\For{$k = 1, \dots, K$}
    \State \textit{\# Phase 0: train the conditional flow map surrogate}
    \State Sample $\{z^{(i)}\}_{i=1}^{B'} \sim \mathcal{N}(0, I)$, $\{y^{(i)}\}_{i=1}^{B'} \sim p_y$
    \State $x_{\mathrm{teach}}^{(i)} \leftarrow \Phi_{\phi_{k-1}}(z^{(i)}, y^{(i)})$ \Comment{64-step RK3, no grad}
    \For{$m = 1, \dots, M$}
        \State Sample $s, t \sim \mathcal{U}[0,1]$ with $|t - s| \in [\delta_{\min}, \delta_{\max}]$
        \State $I_t^{(i)} \leftarrow (1 - t)\, z^{(i)} + t\, x_{\mathrm{teach}}^{(i)}$
        \State $\hat{x}^{(i)} \leftarrow X_{\phi_{\mathrm{cm}}, t \to s}(I_t^{(i)}, y^{(i)})$ \Comment{backward leg}
        \State $\tilde{x}^{(i)}, \tfrac{\partial}{\partial t}\tilde{x}^{(i)} \leftarrow \mathrm{JVP}_t[X_{\phi_{\mathrm{cm}}, s \to t}(\hat{x}^{(i)}, y^{(i)})]$ \Comment{forward leg + derivative}
        \State $\mathcal{L}_{\mathrm{cm}} \leftarrow \|\tfrac{\partial}{\partial t}\tilde{x} - (x_{\mathrm{teach}} - z)\|^2 + \|\tilde{x} - I_t\|^2$
        \State $\phi_{\mathrm{cm}} \leftarrow \phi_{\mathrm{cm}} - \eta_{\mathrm{cm}}\,\nabla_{\phi_{\mathrm{cm}}}\mathcal{L}_{\mathrm{cm}}$
    \EndFor
    \State \textit{\# Phase 1: inner EM loop}
    \For{$n = 1, \dots, N$}
        \State Sample $z, z' \sim \mathcal{N}(0, I)$, $t \sim \mathcal{U}[0, 1]$
        \State \textit{\# Oracle (supervised)}
        \State Sample $(x_{\mathrm{clean}}, y_{\mathrm{or}}) \sim p_{x,y}$
        \State $\mathcal{L}_{\mathrm{or}} \leftarrow \|\hat{b}^{\theta_{\mathrm{or}}}_t((1{-}t)z' + t\, x_{\mathrm{clean}}, y_{\mathrm{or}}) - (x_{\mathrm{clean}} - z')\|^2$
        \State $\theta_{\mathrm{or}} \leftarrow \theta_{\mathrm{or}} - \eta\,\nabla_{\theta_{\mathrm{or}}}\mathcal{L}_{\mathrm{or}}$
        \State \textit{\# EM (unsupervised)}
        \State Sample $y_{\mathrm{real}} \sim p_y$
        \State $x_{\mathrm{em}} \leftarrow X_{\phi_{\mathrm{cm}}, 0 \to 1}(z, y_{\mathrm{real}})$ \Comment{single forward pass}
        \State $y_{\mathrm{fake}} \leftarrow F(x_{\mathrm{em}})$
        \State $y_{\mathrm{cond}} \leftarrow [y_{\mathrm{fake}}\ (\lfloor\rho B\rfloor),\ y_{\mathrm{real}}\ ((1-\rho)B)]$ \Comment{$\rho =$ \texttt{y fake  ratio}}
        \State $\mathcal{L}_{\mathrm{em}} \leftarrow \|\hat{b}^{\theta_{\mathrm{em}}}_t((1{-}t)z' + t\, x_{\mathrm{em}}, y_{\mathrm{cond}}) - (x_{\mathrm{em}} - z')\|^2$
        \State $\theta_{\mathrm{em}} \leftarrow \theta_{\mathrm{em}} - \eta\,\nabla_{\theta_{\mathrm{em}}}\mathcal{L}_{\mathrm{em}}$
    \EndFor
    \State $\phi_k \leftarrow \theta_{\mathrm{em}}$ \Comment{Update frozen generator}
\EndFor
\end{algorithmic}
\end{algorithm}
```
