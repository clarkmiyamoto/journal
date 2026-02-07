- https://arxiv.org/pdf/2406.07507v1
- https://arxiv.org/pdf/2505.18825

Summary: They introduce flow map matching – an algorithm that learns the two-time flow map of an underlying ordinary differential equation. The approach leads to an efficient few-step generative model whose step count can be chosen a-posteriori to smoothly trade off accuracy for computational expense. They build & describe this in the context of stochastic interpolants.

# Setup
Consider an ODE defined by a stochastic interpolant.
$$
\dot x_t = b_t(x_t), \ x_{t=0} = x_0 \sim p_0
$$
By the transport equation, the $\text{Law}(X_t) = p_t$ s.t.
$$
\partial_t p_t = - \nabla \cdot (b_t \, p_t) , \ p_{t=0} = p_0
$$

**Definition:** Flow Map
The flow map $X_{s,t}$ is s.t.
$$
X_{s,t}(x_s) = x_t
$$
where $x_t$ is the solution to the ODE on the level of the samples. Meaning $x_t \sim p_t$.
1. We can calculate the time evolution on the flow map $X_{s,t}$ over the $t$ parameter 
$$
\begin{align}
X_{s,t}(x_s) & = x_t\\
\partial_t X_{s,t}(x_s) & = \partial_t x_t\\
& = b_t(x_t) & \dot x_t = b_t(x_t)\\
& = b_t(X_{s,t}(x_s)) & X_{s,t}(x_s)  = x_t
\end{align}
$$
	So the flow map's time evolution satisfies the Lagrangian equation
	$$
	\partial_t X_{s,t}(x) = b_t(X_{s,t}(x)), \ X_{s,s}(x) =x
	$$
2. Semi group property. Consider the composition of two flow maps $X_{t,s}(X_{u,t}(x_u)) = X_{t,s}(x_t) = x_s$. This means it acts like
	$$
	X_{t,s}(X_{u,t}(\cdot)) = X_{u,s}(\cdot)
	$$
3. We can also now calculate the time evolution over the $s$ parameter, for $X_{s,t}$. 
	Using the semi-group property on the boundary condition of the Lagrangian equation
	$$
	\begin{align}
	x & = X_{s,s}(x) & \text{B.C. of Lagrangian Eq.}\\
	& = X_{a,s}(X_{s,a}(x)) & \text{Semigroup}\\
	\frac{d}{ds} (x) & = \frac{d}{ds}X_{a,s}(X_{s,a}(x))\\
	0  & = \partial_s X_{a,s}(X_{s,a}(x)) + \nabla X_{a,s}(X_{s,a}(x)) \cdot \partial_s X_{s,a}(x)\\
	& = \partial_s X_{a,s}(X_{s,a}(x)) + \nabla X_{a,s}(X_{s,a}(x)) \cdot b_s(X_{s,a}(x)) &  \partial_s X_{s,a}(x) =  b_s(X_{s,a}(x))\\
	\implies 0 & = \partial_s X_{s,t}(x) + \nabla X_{s,t}(x) \cdot b_s(x)
	\end{align}
	$$
	This is PDE is called the Eulerian equation.

The idea is that you can construct a flow map, and apply it compositionally using the semi-group property, to transport samples in time. Effectively increasing the step size on the ODE discretization.

# Learning the Flow Map
So there are many different ways we can learn the flow map. For example you could treat the Lagrangian / Eulerian equations as PINN losses
$$
\begin{align}
\mathcal L_L(\hat X) & = \mathbb E_{t,s \sim \text{Unif}[0,1]^2}  \mathbb E_{x \sim p_s} \Big[w_{s,t} |  \partial_t X_{s,t}(x) - b_t(X_{s,t}(x))|^2 \Big]\\
\mathcal L_E(\hat E) & = \mathbb E_{t,s \sim \text{Unif}[0,1]^2} \mathbb E_{x \sim p_s} \Big[ w_{s,t} | \partial_s X_{s,t}(x) + \nabla X_{s,t}(x) \cdot b_s(x) |^2 \Big]
\end{align}
$$
The losses are $> 0$ when the PDEs are not satisfied, and go to zero when exactly satisfied. The weights $w_{s,t}>0$ play a similar role to the weighted expectation in the score matching loss.

This assume that you have already learned $b_t(\cdot)$-- that is you had a pretrained stochastic interpolant. What if you don't have that, you can instead make the flow map directly learn to transport samples s.t. $\text{Law}(x_t) = \text{Law}(I_t)$.a

Small aside: Recall that in stochastic interpolants $b_t(x) = \mathbb E[\dot I_t | I_t = x]$, where the ML-engineer specifies their choice for $I_t = I(t, x_0, x_1) + \gamma_t z$ s.t. it's easy to sample. The $\mathbb E$ is over the coupling $\nu(x_0, x_1)$ and the auxiliary noise $z \sim \mathbb N(0, \mathbb I)$.








