#bayesianinference 

Consider an exponential family distribution tilted by a prior $h(x)$.
$$
p_\theta(x) = \frac{1}{Z(\theta)} e^{\langle \theta, \phi(x)\rangle} h(x)
$$
We'll see the posterior mean is related to the score $\nabla_x \log p(x)$.

$$
\nabla_x\log p(x) = \sum_i \theta_i \nabla_x \phi(x) \, 
$$