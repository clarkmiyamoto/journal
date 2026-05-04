#talk #sampling 
Talk by Louis

This coves two papers
- Diffusion Gibbs Sampling
- Faster Sampling via Stochastic Gradient Proximal Sampler

# Notes
# Part 1: Iterative Posterior Sampling

Consider annealing using the convolution over a Gaussian (the path defined by diffusion models). There are 3 phases of the samples
- Gaussian: At times close to Gaussian, the distribution looks very Gaussian.
- Commit: At an intermediate time $\tau \sim \lambda_{max}(\Sigma)$ (where $\Sigma$ is the covariance of the target distribution), particles commit to a mode. Notice in the generative problem, the covariance is accessible to you; in the sampling problem you don't have this!
- Memorization: At a final time, the particle commits to a sample in the dataset. But in the 

To implement this, you should use the reverse-SDE dynamics
$$
\begin{align}
\nabla \log p_t(x_t) = - \frac{x_t - \mathbb E_{p_{0|t}}[X_0]}{\sigma_t^2}
\end{align}
$$
and then they computed the expectation value using importance sampling
$$
\begin{align}
\mathbb E_{p_{0|t}}[X_0] = \mathbb E_{q_{0|t}}[\frac{p_{0|t}}{q_{0|t}} X_0] 
\end{align}
$$
where $q = \mathcal N(x_t / \alpha_t , \sigma_t^2 / \alpha^2_t)$. The problem is due to importance sampling, if you have multiple modes (where some modes have lower mass), you'll never travel to them. So you can only do this for times greater than the speciation time $t_*$ (closer to Gaussianaity).

So perhaps instead of wasting compute, and having samples commit too early high-mass modes. You can initialize your samples at the speciation time. Let's use Langevin at that time
$$
\begin{align}
dX_t = \nabla \log p_{t_*}(X_t) dt + \sqrt{2} \, dW_t
\end{align}
$$
where I'm estimating the score using the importance sampling above.
____
**Algorithm**
Input:
- Speciation time: $t_*$ 
Procedure
1. Estimate the score $\hat s \approx \nabla \log p_{t_*}$ using importance sampling (talked about above)
2. Run Langevin on $\hat s$ (which samples $p_{t_*}$)
3. Using the generated samples, now run diffusion to get samples from $\pi = p_0$
_____
Just remember finding the speciation time is hard (because it corresponds to the covariance of the target distribution, which you don't know a priori).
# Part 2: New Method (Diffusion Gibbs Sampling)

In this method, you iterative noise and denoise to some intermediate time. You save the samples at the target distribution and the intermediate time. This will define a joint.