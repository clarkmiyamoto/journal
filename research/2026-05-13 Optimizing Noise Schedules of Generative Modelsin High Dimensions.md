#paper #scsi 

Link: https://arxiv.org/pdf/2501.00988

# Notation
- $f(d) = \Theta_d(g(d))$ if $\lim_{d \to \infty} f(d) / g(d) \in (0, \infty)$

# Setup
Conisder the stochastic interpolant
$$
\begin{align}
I_\tau = c \alpha_\tau z + \beta_\tau a
\end{align}
$$
where $c > 0$, $z \sim \mathcal N(0, \mathbb I_d)$, and $a \sim \mu$ (data distribution). Assume $\alpha_\tau, \beta_\tau$ are independent of $d$. This has a corresponding ODE
$$
\begin{align}
\dot X_\tau = b_\tau (X_\tau), \quad b_\tau(x) = \mathbb E[\dot I_\tau | I_\tau = x]
\end{align}
$$
If $X_0 \sim \mathcal N(0, c^2 \mathbb I_d)$, then $\text{Law}(X_\tau) = I_\tau$ s.t. $X_{\tau = 1} \sim \mu$.


# Time Dilated Interpolants
Consider the Dilated VP Interpolant and Dilated VE Interpolant
$$
\begin{align}
I_t^P & := I^P_{\tau = \tau_t} = (1-\tau_t) z + \tau_t a\\
I_t^E & := I_{\tau = \tau_t}^E = \sqrt{d} (1-\tau_t) z + \tau_t a
\end{align}
$$
