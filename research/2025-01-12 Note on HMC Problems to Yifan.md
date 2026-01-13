#bayesianinference #sampling #paper #note

I'm trying to answer [[Yifan Chen]]'s question on "What problems did Hogg recommend for our HMC algorithm?". There are two Hogg recommended
1. Posterior of radial velocity measurements with multiple planets (see [TheJoker](https://thejoker.readthedocs.io/en/latest/)).
2. Modeling what galaxies look pixel space.
Here are the problems written in more detail.
# TLDR
- Option (1): Posterior sampling
	- Easy to implement
	- Can scale to very high dimensions (just change number of planets)
	- Posterior changes number of modes dependent on how well the data constrains it
		- When data causes posterior is unimodal, [we can use MCMC](https://thejoker.readthedocs.io/en/latest/examples/4-Continue-sampling-mcmc.html) and it has a direct comparison to `emcee`.
	- Scientific goal is misaligned with our method. In the multi-planet case, astronomers would be interested in a method which compares which model is true (how many planets does this star have?). Which our method doesn't directly answer.
- Option (2) 
	- May be more difficult to implement...
	- Can scale to very high dimensions
	- I am unsure of the scientific goal here, so it's not clear (to me) if this contribution will be appreciated by astronomers.

If you ask Erik Petigura, I predict he'll find (1) more interesting as his background is in exoplanets. 
## 1)  RV Data
### Problem Setup:
Consider a planet orbiting a star. That sun has some velocity $\vec v(t)$ which is observed to us on Earth--well I'm kinda lying because we only see the velocity projected parallel to our line of sight $\hat n(t)$, this is the **radial velocity (RV)** $v_{RV}(t) = \vec v(t) \cdot \hat n(t)$. Astronomers can get sporadic measurements $\mathcal D  = \{(t_i, v^{RV}_{t_i}, \sigma_{t_i})\}_{i=1}^N$ using red-shift arguments (which I won't get into...) (where $\sigma_t$ is the measurement error at time $t$).

![[2025-01-12 rv.png]]

![[2025-01-12 raw rv data.png]]
*Example: Set of measurements $\mathcal D$*

Why talk about this? Well physics 101 says the velocity $\vec v(t)$ is determined by the mass of planets, distance from each other, stuff like that. I'll generically notate these parameters as $\theta$. So basically physics gives you a model $p(\vec v(t) | \theta)$. Astronomers want $p(\theta | \vec v(t))$, meaning we'll sample a posterior!

For notation sake, going forward I'll notate $v(t) \equiv v_{RV}(t)$.

Anyways, if you assume a Newtonian two body model, you can calculate the RV using this equation (see [Murray & Correia](https://www.semanticscholar.org/reader/b996a8ccd2bff4c869e0a1c03e1232ee69ae2651))
$$
\begin{align}
v(t; \theta) & = v_0  + K (\cos (\omega + f) + e\, \cos (\omega))\\
\text{s.t. } \cos f & = \frac{\cos E - e}{1 - e \cos E} \\
M & = \frac{2\pi t}{P} - \phi_0 =E - e \sin E
\end{align}
$$
where $\theta = \{v_0, K, P, e, \omega, \phi_0\}$
- $v_0$ is the barycenter velocity
- $K$ is the velocity semi-amplitude
- $P$ is the period of the orbit
- $e$ is the eccentricity
- $\omega$ is the pericenter argument
- $\phi_0$ is the pericenter phase

We assume iid Gaussian measurement error, so your likelihood becomes
$$
\log p (\mathcal D | \theta) = \frac{1}{2}\sum_{i=1}^N \frac{(v(t_i; \theta) - v^{RV}_{t_i})^2}{\sigma_{t_i}^2} + \text{Constant}
$$
$P,e,\omega,\phi_0$ are non-linear w.r.t $\log p(\mathcal D | \theta)$ so it'll be difficult to estimate them.

The priors are given in (https://arxiv.org/pdf/1610.07602).

### Examples of Posteriors
A slight problem is the posterior don't make themselves to affine invariant / mcmc methods as they're highly multimodal. This is because you can fit any set of data points using a cosine with lower and lower period. 

*Example: Corner plot of posterior*
 ![[2025-01-12 posterior corner.png]]
*Example: 128 samples of a posterior samples, plotted $v(t; \theta_{sample})*![[2025-01-12 posterior velocities.png]]
### How to make high-dimensional:
To make this a higher dimensional problem, consider a $n$-body model $\theta = \{v_0, K^{(1)}, P^{(1)}, e^{(1)}, \omega^{(1)}, \phi_0^{(1)}, ..., K^{(n)}, P^{(n)}, e^{(n)}, \omega^{(n)}, \phi_0^{(n)} \}$. Let's assume the planets don't interact (they're highly spaced out \& of negligible mass compared to the their star), this yields a likelihood of
$$
\log p (\mathcal D | \theta) = - \frac{1}{2}\sum_{i=1}^N \sum_{j=1}^n\frac{(v^{(j)}(t_i; \theta^{(j)}) - v^{RV}_{t_i})^2}{\sigma_{t_i}^2} + \text{Constant}
$$
where $v^{(j)}(t; \theta)$ is the radial velocity contributed by the $j$'th planet. 

More in-depth analysis would require looking at the interacting model, and perhaps looking at the model evidence to infer how many planets are orbiting a single star.

## 2) Modeling Fields in Pixel Space
This recommendations comes from Mike Blanton. Back in the day they used to model the CMB, as if you took a photo of it. He couldn't recall in detail, so he left it up to me to google around... Here's what I found.

*Examples*
Astronomical
- https://iopscience.iop.org/article/10.1088/0004-6256/146/1/7/pdf (Brewer, DFM, Hogg). Produces catalogs from images of stellar fields using a hierarchical Bayesian model. They use diffusive nested sampling (DNS).
Cosmological (More likely what Mike Blanton was referencing)
- https://arxiv.org/pdf/0708.2989 (Taylor, Ashdown, Hobson, 2008). Samples posterior, which is the joint over power spectrum coefficients $C \in \mathbb R^{2 * 10^5}$ using Hamiltonian Monte Carlo.
- https://www.aanda.org/articles/aa/full_html/2019/01/aa34117-18/aa34117-18.html (Ramanah, Lavaux, Jasche, Wandelt, 2018). Generates three-dimensional primordial and present-day matter fluctuations using Hamiltonian Monte Carlo.
What's funny is that David is an author on all some of these, but didn't really remember it / found it to be a good example after Mike brought it up. So perhaps he thinks these are solved / not interesting problems. The cosmological examples might be out of his knowledge.

I'll go over the (Brewer, DFM, Hogg) article, as it's more likely Erik might find it interesting... Also the cosmological examples the comparison is simple, just try our method and compare it to their method lol.
### The Problem
We are concerened with inferring the positions of stars from an image of them. Images are pixels, meaning you have a finite resolution to the position, so we need to make a sub-pixel inference as to their positions. Also note there are an unknown number of stars... So you can see this problem becomes difficult quickly...

Here's the mathematical setup. We shall assume that there are an unknown number of stars $N$ in the field (*note* the previous section used $N$ for number of planets, hold number of stars = 1 constant). Each star has an unknown position $(x_i, y_i)$ in the plane of the sky and an unknown flux $f_i$. We also describe the distribution of fluxes (commonly known as the luminosity function) of the stars by some parameters denoted collectively by $\beta$. In summary, the unknown parameters are $\theta = \{N, \beta \{x_i, y_i\}_{i=1}^N, \{f_i\}_{i=1}^N \}$.
	The most troublesome quantity is $N$ (number of stars). So we'll have to do some thinking about making a sampler which has a jump process.
	*Heuristic illustration:* (1) change number of stars $N' \sim T(N \to N')$. (2) sample $(\beta', \{x_i', y_i', f_i'\}_{i=1}^{N'}) \sim \text{HMC}(\beta, \{x_i, y_i, f_i\}_{i=1}^{N})$. (3) accept/reject according to a M-H ratio.

The data observed is a $n \times m$ matrix $I_{ij}$ which describes intensities of light at pixel locations (indexed by $i,j$). We assume the observed noise is Gaussian (as per usual). The map between the central position of the pixel $(X_{ij}, Y_{ij})$ (because the pixels describe positions in the sky!) and pixel positions are given by a point spread function $\mathcal P$
$$
\begin{align}
I_{ij} & = \epsilon_{ij} + \sum_{k=1}^N f_k \, \mathcal P(X_{ij}-x_k, Y_{ij}-y_k)\\
\mathcal P (x, y) & = \frac{w}{2\pi s_1^2} \exp (- (x^2  + y^2) / 2 s_1^2) + \frac{1- w}{2 \pi s_2^2} \exp(- (x^2 + y^2) / 2 s_2^2)
\end{align}
$$
Meaning the likelihood is 
$$
\log p(\mathcal D | \theta) = - \frac{1}{2}\sum_{i,j=1}^{n, m} \frac{(I_{ij} - \sum_{k=1}^N f_k \, \mathcal P(X_{ij} - x_k, Y_{ij} - y_k) )^2}{ \sigma_{ij}^2} + \text{Constant}
$$
As a recap of the parameter
- $\mathcal D = \{I_{ij}\}_{i,j=1}^{n, m}$ which is the intensities of pixels (i,j) from an picture.
- $\theta = \{N, \beta, \{x_k, y_k\}_{k=1}^N, \{f_k\}_{k=1}^N \}$ 
	- $N$ number of stars
	- $\beta$ are parameters for the point spread function $\mathcal P$
	- $(x_k, y_k)$ is the position of the $k$'th star
	- $f_k$ is the flux of the $k$'th star

_____
# Note to Clark

# Thoughts
As I was writing this up for Yifan, I had a couple questions / realizations
- I should understand Nested Sampling lol
- What is bridge sampling? 
- What if I could do annealed importance sampling using Hamiltonian dynamics instead of Langevin (add this to my reading list https://arxiv.org/pdf/1205.1925)

Make new note in research journal later...