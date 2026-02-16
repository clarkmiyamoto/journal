#stochasticinterpolants #robotics 

Had an interesting chat w/ [[Alex Jiang]] about using generative models where the physical actions operate in the generative model's inference time. 

He had a couple questions
1. Your final distribution is a Gaussian mixture, where each Gaussian has a certain variance. You don't know this but this is, your data is just samples. Instead you want sample a Gaussian with reduced variance. 
2. Consider the Push-T dataset / problem. Historically people have used diffusion models, which go from white noise to actions, to solve this. The problem is that you have to wait for the diffusion model to finish before you can take your action. What if you could initialize your robots position in your base distribution, and the sample's trajectory would be the actual path your robot should take. People have done this in the flow matching context (see paper attached), but I'd like to see if this is possible in stochastic interpolants.

To explain that jargon. In diffusion models you move from white noise to an action. But what if you could put in an action and then have an output over the conditional. I think you can do this using couplings.

# Solution to (1)
Use Self Consistent Stochastic Interpolants where your forward model is $\mathcal F(x) = x + \sigma z$. And you tune $\sigma$ to be the amount of variance reduction you want.

SCSI solves the problem of you have noisy observations $y$ but you actually want to sample $x$ (a corresponding signal). If you're able to specify a forward model, you can sample $x$.

# Solution to (2)
So in the [streaming flow policy](https://siddancha.github.io/streaming-flow-policy/) (SPE) you can implement this directly in stochastic interpolants. Choose the interpolant
$$
x_t = \xi(t) + \sigma_t z
$$
where $\xi(t)$ is a trajectory the robot hand in practice.

But what's nice is you can also the SDE, something which was not easily found in the flow matching pape.r








# Interesting Realization
Consider the stochastic interpolant
$$
x_t = I(t, x_0, x_1) + \gamma_t z
$$
where $x_0,x_1$ are drawn from the coupling $\nu(x_0, x_1)$. This has the condition $\int \nu(\cdot, x_1) \, dx_1 = p_0$ and $\int \nu(x_0, \cdot) dx_0 = p_1$. Traditionally, we just take $\nu(\cdot, \star) = p_0 \times p_1$ to just be independent, but if you couple them you can get nicer interpolation paths.

For example. In rectified flows, they train the algorithm to perform the transport, and after one run, they retrain using the coupling endowed by the transport, you then get perfectly straight lines.
![[2026-02-12 interpolant path.png]]
(a) You use the linear interpolant, this is your training data. (b) The learned flow prevent flow paths from intersecting. I'll denote the transport at $\Phi$ and it acts on the states as $\Phi(x_0) = x_1$. So if $x_0$ comes from the top left gaussian, $\Phi(x_0)$ will only end up at the top right gaussian. (c) You relearn the flow using the coupling $\nu(x_0, \Phi(x_0))$.