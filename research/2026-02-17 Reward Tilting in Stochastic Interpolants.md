#paper #stochasticinterpolants 

Paper: https://arxiv.org/pdf/2512.21829

# Notes
Say we have access to sample $p_1(x)$, but instead we want to sample $p_{1,a} = p_1 e^{ar}$ where $r(\cdot)$ is the reward function.

Define the tilted interpolant
$$
I_t^a = \alpha_t x_0 + \beta_t x_1^a
$$
where the coupling is independent $(x_0, x_1^a) \sim p_0(x_0) p_{1,a}(x_1^a)$. Define $p_{t,a} = \text{Law}(I_t^a)$.

We could write the Fokker Planck and say we'll learn $b_{t,a}(x ) = \mathbb E[\dot I_t | I_t^a = x]$, but this training objective assumes we have access to samples $x_1^a \sim p_{1,a}(x_1^a)$.

**Proposition 1** (Esscher Transform): Let $I_t^a = \alpha_t x_0 + \beta_t x_1^a$ be the interpolant constructed from $x_1^a \sim p_{1,a}$. Then the augmented velocity field $b_{t,a}(x)$ is given by
$$
b_{t,a}(x) = \frac{\mathbb E[\dot I_t^0 e^{ar(x_1)} | I_t^0 = x]}{\mathbb E[e^{ar(x_1)} | I_t^0 = x]}
$$
Furthermore, any shift from $a$ to $a+h$ updates the velocity
$$
b_{t,a+h}(x) = \frac{\mathbb E[\dot I_t^a e^{hr(x_1^a)} | I_t^a = x]}{\mathbb E[e^{hr(x_1^a)} | I_t^a = x]}
$$
*Proof:*  
The PDF of the tilted interpolant $I_t^a$ is given by 
$$
p_{t,a}(x) = \frac{1}{Z_a} \mathbb E[\delta(x - I_t^0) e^{a r(x_1^0)}]
$$