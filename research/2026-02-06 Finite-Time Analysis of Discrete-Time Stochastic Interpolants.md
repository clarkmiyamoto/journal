#paper #stochasticinterpolants #generative 
Paper: https://arxiv.org/pdf/2502.09130

The paper looks at the discretized version of the SDE for Stochastic Interpolants, and suggests a better noise scheduler.

Consider the forward SDE
$$
dX_t = b_t(X_t) \, dt + \sqrt{2 \epsilon_t} dW_t
$$
In discrete time this becomes
$$
X_{t + \Delta t} = b_t(X_t) \Delta t + \sqrt{2 \epsilon_t \, \Delta t} \, z
$$
where $z \sim \mathcal N(0, \mathbb I)$. 