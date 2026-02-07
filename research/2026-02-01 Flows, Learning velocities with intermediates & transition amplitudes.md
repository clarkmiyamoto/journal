#paper #robotics #generative 

Was chatting w/ Alex today and realized it'd be nice to learn more about diffusion for policy (in the context of robotics).

Two papers came up
- Streaming Flow Policy: https://arxiv.org/pdf/2505.21851
- Transition Matching: https://arxiv.org/pdf/2506.23589

# Paper 1: Streaming Flow Policy
You want to tell your robot how it should move $a(t): [0,1] \to \mathcal A$ given it's current position $\xi(0)$. The dataset is a set of trajectories $\xi^{(i)}(t) : [0,1] \to \mathbb R^d$ which you want to emulate.

Given a trajectory $\xi$, we want to construct a drift field $v_t(a)$ that forces samples $a_t$ to closely follow $\xi$.  A simple example of this would be to have a Gaussian centered along the trajectory.
$$
p(a | \xi(t)) = \mathcal N(a; \xi(t), \sigma_t^2)
$$
Now one has to ask, what is the ideal drift $v^{target}$ which has this behavior?

Consider the transport equation:
$$
\begin{align}
\partial_t p_t(a) = - \nabla \cdot (v_t^{target}(a) \, p_t(a)) = -\nabla\cdot v_t^{target}(a) \, p_t(a) - v_t^{target}(a) \cdot \nabla p_t(a)
\end{align}
$$
Substitute in the distribution directly
$$
\begin{align}
	\partial_t p_t(a) & = p_t(a) \left[ - \frac{\dot \sigma_t}{\sigma_t}  + \frac{(a-\xi_t) \xi_t}{\sigma_t^2}  + \frac{(a - \xi_t)^2 \dot \sigma_t}{\sigma_t^3} \right]\\
	\nabla p_t(a) & = - p_t(a) \frac{a-  \xi_t}{\sigma_t^2}
\end{align}
$$
Pluggin in
$$
\begin{align}
- \frac{\dot \sigma_t}{\sigma_t}  + \frac{(a-\xi_t) \xi_t}{\sigma_t^2}  + \frac{(a - \xi_t)^2 \dot \sigma_t}{\sigma_t^3} & = - \nabla \cdot v_t^{target}(a) - v_t^{target}(a) \frac{a- \xi_t}{\sigma_t^2} \\
v_t^{target}(a) & = \dot \xi_t+ \frac{\dot \sigma_t}{\sigma_t}(a - \xi_t)
\end{align}
$$
Now you can use the flow matching loss to find the marginal drift field. 

# Paper 2: Transition Matching
So instead of learning some distribution, you are learning a transition kernel (in the Markov Chain sense).

Given a set of iid samples from a target distribution $p_T$ and a base distribution $p_0$. The goal is to learn the Markov process $p(x_{t+1} | x_t)$, allowing us to flow from $p_0$ to $p_T$.
