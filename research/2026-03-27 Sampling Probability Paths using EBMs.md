#talk #sequentialmontecarlo #generative 

# Notes
## Setup
Let $p_t$ be a family of probability distributions, which defines a probability path. We use the diffusion time notation: $t=1$ denote the target distribution, and $t=0$ denotes base distribution (Gaussian).

There are many choices for paths, some examples
- Annealed MCMC
- Sequential Monte Carlo
- Simulated Annealing
- Telescoping Desnity Ratio Estimation
- Annealed MCMC
- Multi-level denoising score-matching
- Energy-based diffusion models
But what is a good path?

## Sampling
**Goal:** You have query access to $\pi$, and you want to generate samples $x\sim \pi$.

Let's consider the Langevin algorithm for sampling, and a path-guided Langevin
- No path, requires exponential iteration
- If you use $\nabla \log p_t(x)$, where you used the geometric mean $p_t$. You need exponential iteration
- But if you use $\nabla \log p_t(x) \sim \text{DO}_\pi$ (diffusion path), then you only need polynomial iterations (at least for low dimensions).

