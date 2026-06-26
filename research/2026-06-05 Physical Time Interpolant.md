
**Objective:**
Want to generate actions $A \in \mathbb R^6$ (configuration of joints for arms). At every moment $t$, the robot makes an observation of the world $O_t$, and you have a current action $A_t$, and we want to generate the next action $A_{t+\Delta t}$.


### Action to Action Interpolant
Similar to https://arxiv.org/pdf/2602.07322v1

Consider a continuous trajectory of the robot's action $\tilde A_t(\tau | O_t)$ (s.t. $A_t(\tau) = A_{t+\tau}$, meaning $\tilde A_t(\tau=0) = A_t$ and $\tilde A_t(\tau = 1) = A_{t+\Delta t}$).  The action-to-action interpolant can be
$$
\begin{align}
I_\tau = \tilde A_t(\tau | O_t) + \gamma_\tau Z
\end{align}
$$
The interpolant's time $\tau$ now corresponds to increments in physical time $t$. The drift you learn is conditioned on the current position of the robot $a_t$ and observation $o_t$
$$
\begin{align}
b_\tau(x | a_t, o_t) = \mathbb E[\dot I_\tau | I_\tau = x , \tilde A_t(\tau=0) = a_t, O_t = o_t]
\end{align}
$$
where the dot denotes the derivative w.r.t. $\tau$. 
- Note you have to take a derivative of robot's actions w.r.t. increment time $\tau$, but you only have discrete observations of the robot's actions $A_t$. To get over this, you could fit a spline, and then differentiate it.

**Questions I have**
- How do you keep the predicted action coherent over multiple inferences?
