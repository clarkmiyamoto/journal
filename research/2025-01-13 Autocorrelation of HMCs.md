#hemcee 
```table-of-contents
title: **Table of Contents**
style: nestedList # TOC style (nestedList|nestedOrderedList|inlineFirstLevel)
minLevel: 0 # Include headings from the specified level
maxLevel: 4 # Include headings up to the specified level
include: 
exclude: 
includeLinks: true # Make headings clickable
hideWhenEmpty: false # Hide TOC if no headings are found
debugInConsole: false # Print debug info in Obsidian console
```

 In the [original ChEES paper](https://proceedings.mlr.press/v130/hoffman21a/hoffman21a-supp.pdf) the authors calculate the lag-1 autocorrelation (assuming continuous hamiltonian dynamics, discretization error is negligible) for a Gaussian distribution. They do this to justify the creation of the $\text{ChEES}$ criterion. Here I do the following
 - Recreate their results (dynamics follow the regular hamiltonian dynamics)
 - Redo their calculations, but using the side and walk hamiltonian dynamics.

# Results
Denote $\rho^{(1)}$ as the lag-1 autocorrelation of the first moment, and $\rho^{(2)}$ for the 2nd moment.

$$
\begin{align}
\rho^{(1)} & = \frac{\mathbb E[(x - \mathbb E[x])(\tilde x -\mathbb E[x])]}{\mathbb E[(x - \mathbb E[x])^2]}\\
\rho^{(2)} & = \rho^{(1)}= \frac{\mathbb E[(x^2 - \mathbb E[x^2])(\tilde x^2 -\mathbb E[x^2])]}{\mathbb E[(x^2 - \mathbb E[x^2])^2]}
\end{align}
$$

where $\tilde x$ is the proposed position after leapfrogging for time $t$, and $x$ is starting position $x(t=0)$ .

Assume the leapfrog integrators as walking along the distribution $\mathcal N(0, \sigma^2 \mathbb I_d)$, and assume walkers are all at stationarity.

**Regular Leapfrog**
Consider a walker's position $x$, projected along the $i$'th direction. The leapfrog dynamics has the mass matrix $M = \text{diag}(m_1, ..., m_d)$.

$$
\begin{align}
\rho^{(1)} & = \cos\left(\frac{t}{\sqrt{\sigma^2 \, m_i} } \right)\\
\rho^{(2)} & = \cos^2 \left(\frac{t}{\sqrt{\sigma^2 \, m_i} } \right) = \frac{1}{2} \left(1 + \cos \left (t/\sqrt{\sigma^2 m_i} \right)\right) 
\end{align}
$$

**Side Move Leapfrog**
Consider a walker's position $x$, and the projection due to the complement ensemble $v = \frac{1}{\sqrt{2d}}(x_i - x_j)$. We look at the autocorrelation in the $x \cdot \hat v$ direction

$$
\begin{align}
\rho^{(1)} & = {}_1F_1 \left(\frac{d}{2}; \frac{1}{2} ;- \frac{t^2}{2d} \right)\\
\rho^{(2)} & = \frac{1}{2} \left[1 + {}_1F_1 \left(\frac{d}{2}; \frac{1}{2} ;- \frac{2 t^2}{d} \right) \right]
\end{align}
$$

![[2026-01-13 First Moment SideHMC.png]]
- First moment's autocorrelation
- It's affine invariant!
- Has dimensional scaling, which wasn't there in the regular hamiltonian dynamics.
	- I wish, the leapfrog time when the autocorrelation is zero, would be independent of dimension...
- I highly suspect $\lim_{d\to \infty} {}_1 F_1 (\frac{d}{2}; \frac{1}{2} ; - \frac{t^2}{2 d}) = \cos(t)$, implying the dynamics are stable in the infinite dimensional limit. Which recovers the perfectly tuned regular HMC.

![[2026-01-13 Second Moment SideHMC.png]]
- Second moment's autocorrelation
- Once again the result is affine invariant.
- The result mirrors the regular HMC result! *whew*
- Interestingly we have a (sorta) blessing of dimensionality. The achievable autocorrelation gets smaller as the dimension grows. However this just recovers the regular HMC. Reaffirms our suspicion that a well tuned regular HMC is better than an affine invariant HMC.

**Walk Move Leapfrog**
Consider a walker's position $x$, we just look at it in the $i$'th direction. In the derivation, there is some random matrix theory, so I have to take $d,N \to \infty$ (dimension, number of walkers) while keeping $\gamma \equiv \frac{d}{N/2 - 1}$ constant.

$$
\begin{align}
\rho^{(1)} & = \mathbb E_{\lambda \sim \text{MP}(\gamma, 1)}[\cos(\sqrt{\lambda} \, t)]\\
\rho^{(2)} & = \frac{1}{2} \left( 1 + \mathbb E_{\lambda \sim \text{MP}(\gamma, 1)}[\cos(\sqrt{\lambda} \, \, 2t)] \right) & \text{(Conjecture)}
\end{align}
$$

![[2026-01-13 First Moment WalkHMC.png]]
- First moment's autocorrelation
- Just keep in mind $\gamma = \frac{d}{N/2 - 1}$, when $d,N \to \infty$.  This algorithm only has affine invariance when $N/2 > d$.
	- You'd expect as $N/2 \gg d$ (meaning $\gamma \to 0$) the algorithm would perform more favorably. I suppose we see this in that there are more $t$'s which have zero autocorrelation.
	- I was hoping there was a method of seeing it's autocorrelation keeping $d/N$ fixed, but increasing $d$.
![[2026-01-15 Second Moment WalkHMC.png]]
- Second moment's autocorrelation
# Leapfrog Integrators
The leapfrog integrators of interest
- **Regular leapfrog**: $(\tilde x, \tilde p) = \text{Leapfrog}_{t, M}(x, p)$ are the proposed steps using the leapfrog integrator for a total time $t$ with a mass matrix $M$. Where $x \in \mathbb R^d$ and $p \in \mathbb R^d$. 

	$$
	\frac{dx}{dt} = M^{-1} \, p , \ \frac{dp}{dt}= -\nabla V(x )
	$$

- **Walk-move leapfrog**: $(\tilde x, \tilde p) = \text{Leapfrog}_{t, B} (x, p)$, where $x \in \mathbb R^d$ and $p \in \mathbb R^{N/2}$.

	$$
	\frac{d x}{dt} = B \, p, \ \frac{dp}{dt} = - B^T \nabla V(x)
	$$

	where $B \in \mathbb R^{d \times N/2}$ is the centered positions of the component ensemble. Note that when the ensemble reaches equilibrium, $\mathbb E[B B^T] = \Sigma$ . 
- **Side-move leapfrog**: where $x \in \mathbb R^d$ and $p \in \mathbb R$

	$$
	\frac{dx}{dt} = \frac{1}{\sqrt{2d}}(x_i - x_j) p  , \ \frac{dp}{dt} = -\frac{1 }{\sqrt{2d}}(x_i - x_j)^T \nabla V(x)
	$$

	where $x_i, x_j \in \mathbb R^d$ are the positions of two randomly chosen walkers from the complement ensemble.

# Note on Autocorrelation
Given a sequence of random variables $X_1, X_2, ...$ the lag-k autocorrelation $\rho_k$ is given by

$$

\rho_k = \frac{\mathbb E[(X_t - \mu)(X_{t+k} -\mu)]}{\mathbb E[(X_{t} - \mu)^2]} = \frac{\text{Cov}(X_{t+k}, X_t)}{\text{Var}(X_t)}

$$

where $\mu$ is the stationary mean. Note the random variables can be transformations $X_i = f(\theta_i)$ of the outputs of the MCMC algorithm.

Properties
- First is unity: $\rho_0 = 1$
- Bounded: $\rho_k \in [-1, +1]$.
- If $X_{t+k}$ is affine equivariant, then $\rho_k$ is affine invariant.

The [[integrated autocorrelation time]] is given by "integrating" (summing) up all the lag-k autocorrelations.

$$

\rho = \sum_{k=-\infty}^\infty \rho_k = 1 + 2 \sum_{k=1}^\infty \rho_k

$$

Now we can get to the interesting part...
# Analysis of Lag 1 Autocorrelation
In the [original ChEES paper](https://proceedings.mlr.press/v130/hoffman21a/hoffman21a-supp.pdf), they analyze the lag-1 autocorrelation for a $d$-dimensional Gaussian $\mathcal N(0, \Sigma)$ where the covariance is diagonal $[\Sigma]_{ii} = \sigma^2_i$. They assume the discretization of the leapfrog integrator is negligible, allowing the analysis to be done on exact Hamiltonian dynamics (which I list above).

They are allowed to assume the Gaussian has zero mean, and diagonal components on covariance, because the ChEES criterion is invariant under shifts and rotations. 

We can still assume the Gaussian has zero mean, and only diagonal components. Because we just want to highlight the affine invariance, which can be effectively described by it's condition number $\lambda_{max}(\Sigma) / \lambda_{\min}(\Sigma)$ where $\lambda(\cdot)$ is the eigenvalue of the input.

**Outline of analysis**
We are analyzing the distribution

$$

p = \mathcal N(0, \Sigma), \ [\Sigma]_{ii} = \sigma_i^2

$$

Meaning the potential $V(x)$ (aka negative log-likelihood without normalization constant) is given by

$$

V(x) = \frac{1}{2}x^T \Sigma^{-1} x \implies \nabla V(x) = \Sigma^{-1} x

$$

In the lag-1 autocorrelation, let $X_{t+1}$ to be the output after using leapfrog dynamics for total time $T = \epsilon L$ (in the limit where the discretization error $\epsilon \to 0$ holding $\epsilon L$ constant, allowing us to use exact Hamiltonian dynamics).
1. Calculate the equations of motion. That is $(\tilde x, \tilde p) = \text{Leapfrog}(x, p)$
2. You can directly calculate the lag-1 autocorrelation.
## Regular Leapfrog
The equations of motion for $V(x) = \frac{1}{2} x^T \Sigma^{-1} x$ is

$$

\frac{dx}{dt} = M^{-1} \, p , \ \frac{dp}{dt}=  \Sigma^{-1}x

$$

where $p(0) \sim \mathcal N(0, M)$.

Let's do something simple where $M = \text{diag}(m_1, m_2)$ and $\Sigma = \text{diag}(\sigma_1^2, \sigma_2^2)$. The solution becomes 

$$

\begin{align}
\tilde x & = x_i(t) = x_i(0) \cos(\omega_i t) + \frac{p_i(0)}{m_i \omega_i} \sin(\omega_i  t)\\
\tilde p & = p_i(t) = p_i(0) \cos(\omega_i t) - x_i(0) \, m_i \omega_i  \, \sin(\omega_i t)
\end{align}

$$

where $\omega_i = 1/(\sigma_i \sqrt{m_i})$. 

Assume the chain is already at stationarity, therefore $x_i(0) \sim \mathcal N(0, \Sigma)$ and $p_i(0) \sim \mathcal N(0, \mathbb I)$.

### (Regular) First Moment's Lag-1 Autocorrelation
Consider the first moment's autocorrelation

$$

\rho^{(1)}= \frac{\mathbb E[(x - \mathbb E[x])(\tilde x -\mathbb E[x])]}{\mathbb E[(x - \mathbb E[x])^2]}

$$

Assuming the chain is already at stationarity, $\mathbb E[x_i] = 0$ and $\mathbb E[x_i x_j] = \sigma_i^2 \delta_{ij}$. 

The only weird thing to calculate is $\mathbb E[\tilde x_i x_i]$

$$

\begin{align}
 \mathbb E[\tilde x_i x_i] & = \mathbb E[x_i x_i \cos(\omega_i t)] & x_i(0),p_i(0) \text{ are independent random variables}\\
 & = \sigma_i^2 \cos(\omega_i \, t)
\end{align}

$$

Therefore the first moment's lag-1 autocorrelation is 

$$

\rho_1 = \cos(\omega_i t)

$$

- Not affine invariant. To get good autocorrelation in all dimension,  you need to tune the mass matrix.
### (Regular) Second Moment's Lag-1 Autocorrelation
Take $X = f(x)$, where $f(x) = x^2$.

$$

\rho^{(2)}= \frac{\mathbb E[(x^2 - \mathbb E[x^2])(\tilde x^2 -\mathbb E[x^2])]}{\mathbb E[(x^2 - \mathbb E[x^2])^2]}

$$

Assuming the chain is already at stationarity, $\mathbb E[x_i^2] = \sigma_i^2$ and $\mathbb E[x_i^4] = 6 \sigma_i^4$.

The denominator becomes

$$

\begin{align}
\mathbb E[(x_i^2 - \mathbb E[x_i^2])^2] & = \mathbb E[x_i^4] - \mathbb E[x_i^2]^2\\
& = 5 \sigma_i^4
\end{align}

$$

The numerator is more lengthy

$$

\begin{align}
\mathbb E[(x^2 - \mathbb E[x^2])(\tilde x^2 -\mathbb E[x^2])] & = \mathbb E[x^2 \tilde x^2 - \tilde x^2 \mathbb E[x^2] - x^2 \mathbb E[x^2] + \mathbb E[x^2]^2]\\
& = \mathbb E[x^2 \tilde x^2] -  \mathbb E[\tilde x^2] \mathbb E[x^2] 
\end{align}

$$

We can now compute

$$

\begin{align}
\tilde x^2 & = x^2 \cos^2(\omega t)  + \frac{p^2}{m^2 \omega^2} \sin^2 (\omega t) + \frac{x \, p}{m \omega} \cos(\omega t) \sin(\omega t)\\
\mathbb E[\tilde x^2] & = \sigma^2 \cos^2(\omega \, t) + \frac{1}{\omega^2} \sin^2(\omega t)\\
\mathbb E[x^2 \tilde x^2] & = 6 \sigma^4 \cos^2(\omega t) + \frac{\sigma^2}{\omega^2} \sin^2(\omega t)
\end{align}

$$

Thus the second moment lag-1 autocorrelation becomes

$$

\begin{align}
\rho^{(2)} = \frac{5 \sigma^4_i \cos^2(\omega_i t)}{5 \sigma_i^4} =  \cos^2(\omega_i t) = \frac{1}{2} ( 1 + \cos(2 \omega_i t))
\end{align}

$$

- One again not affine invariant.
- Interestingly the time when the autocorrelation is zero is the same for $\rho^{(1)}$
## Side-Move Leapfrog
The state space is $x \in \mathbb R^d$ and $p \in \mathbb R$, with the dynamics

$$

\frac{dx}{dt} = v p  , \ \frac{dp}{dt} = -v^T \Sigma^{-1} x

$$

where $v \equiv \frac{1}{\sqrt{2d}}(x_i - x_j)$ s.t. $x_i, x_j \in \mathbb R^d$ are the positions of two randomly chosen walkers from the complement ensemble. At the beginning of the dynamics $p(0) \sim \mathcal N(0, 1)$.

Here I'll simplify my life by choosing a Gaussian with covariance $\Sigma = \sigma^2 \mathbb I_d$. 

$$
\frac{d^2 x}{dt^2} = - \frac{vv^T}{\sigma^2} x
$$

The dynamics project along the line spanned by $\hat v$, so let's look at the equations of motion there. Consider $x_v \equiv x \cdot \hat v$. $$
	\frac{d^2x_v}{dt^2} = - \left( \frac{vv^T}{\sigma^2} x \right)^T \hat v = - x^T \frac{vv^T}{\sigma^2} \frac{v}{\|v\|}= -  \frac{v^2}{\sigma^2}x_v

$$
Let $\omega^2 = v^2/\sigma^2$. You get the dynamics
$$

x_v(t) = x_v(0) \cos(\omega \, t) + B \sin(\omega \, t)

$$
To get the boundary condition $\dot x_v(t) = - x_v(0) \omega  \sin(\omega \, t) + B \omega \cos (\omega t)$. At time zero,
$$

\begin{align}
\dot x(0) \cdot \hat v & = \dot x_v(0) \\
\dot x(0) \cdot \hat v & = B \omega\\
vp(0) \cdot \hat v& =B \omega\\
p(0) \, \|v\| & = B \omega
\end{align}

$$
Therefore $B = p(0) \|v\| / \omega = p(0) / \|v\| \sigma$. Leaving us with
$$

\begin{align}
x_v(t) & = x_v(0) \cos(\omega \, t) + \frac{p(0)}{ \|v\| \sigma} \sin(\omega \, t)\\
& = x_v(0) \cos(\|v\| t / \sigma) + \frac{p(0)}{ \|v\| \sigma} \sin(\|v\| \, t / \sigma)
\end{align}

$$
Assume the chain is already at stationarity for a target distribution $\pi = \mathcal N(0, \sigma^2 \mathbb I_d)$, therefore $x(0) \sim \pi$ and $p(0) \sim \mathcal N(0, 1)$. Also assume the complement walkers $x_i, x_j \sim^{iid} \pi$ and are independent from $x(0)$. I'll also note that $x_i - x_j \sim \mathcal N(0, 2 \sigma^2 \mathbb I_d)$, therefore $v \sim \mathcal N(0, \tfrac{\sigma^2}{d} \mathbb I_d)$. 

The dynamics can be rewritten in a natively affine invariant manner by choosing a whitened $v = \frac{\sigma}{\sqrt{d}} z$, where $z \sim \mathcal N(0, \mathbb I_d)$. This yields
$$

x_v(t) = x_v(0) \cos(\|z\| \, t/\sqrt{d}) + \frac{p(0)}{\sqrt{d}} \sin(\|z\| \, t /\sqrt{d})

$$

### (Side) First-Moment's Lag-1 Autocorrelation
Recall the lag-1 autocorrelation is given by

$$

\rho^{(1)}= \frac{\mathbb E[(x_v - \mathbb E[x_v])(\tilde x_v -\mathbb E[x_v])]}{\mathbb E[(x_v - \mathbb E[x_v])^2]}

$$

There are few new things to compute

In the denominator
$$

\begin{align}
\mathbb E[x_v] & =  \sum_i \mathbb E [x_i] \mathbb E[ \frac{\hat v_i}{\|v\|}] = 0\\
\mathbb E[x_v^2] & =  \sum_{i} \mathbb E [x_i^2] \mathbb E[ \frac{v_i^2}{\|v\| ^2 } ]  + \sum_{i \neq j} \mathbb E[x_i] \mathbb E[x_j] \mathbb E[\frac{v_i v_j}{\|v\|^2}]\\
& =  \sum_{i} \mathbb E [x_i^2] \mathbb E[ \frac{v_i^2}{\|v\| ^2 } ] \\
& = \sum_{i=1}^d \sigma^2 \frac{1}{d} = \sigma^2
\end{align}

$$

Meaning the denominator in total is just $\sigma^2$. 

Now for the numerator, 

$$

\mathbb E[(x_v - \mathbb E[x_v])(\tilde x_v -\mathbb E[x_v])] = \mathbb E[x_v \tilde x_v] - \mathbb E[\tilde x_v]\mathbb E[x_v] 

$$

The proposed position

$$

\begin{align}
\mathbb E[\tilde x_v] & = \mathbb E\left[x_v(0) \right] \mathbb E \left [ \cos(\|z\| \, t/\sqrt{d}) \right ] +  \mathbb E \left [\frac{p(0)}{\sqrt{d}}  \right] \mathbb E \left[\sin(\|z\| \, t /\sqrt{d}) \right]    = 0
\end{align}

$$

and the correlation between current and proposed

$$
\begin{align}
\mathbb E[\tilde x_v x_v] & = \mathbb E \left[  x_v(0)^2 \cos(\|z\| \,  t / \sqrt{d})\right] \\
& = \sigma^2 \,  \mathbb E \left[ \cos \left(\|z\| t/\sqrt{d}\right)\right]\\
& = \sigma^2 \int \cos(\|z\| t/\sqrt{d}) \frac{e^{-z / 2}}{\sqrt{(2\pi)^d}}\, dz_1 ... dz_d \\
& = \sigma^2 \int_0^\infty \cos (r \, t/\sqrt{d}) r^{d-1} \frac{e^{- r^2 / 2}}{\sqrt{(2\pi)^d}} \, dr \, d\Omega & \text{Sphereical volume element}\\
& = \sigma^2 \frac{1}{\sqrt{(2\pi)^d}} \frac{2 \pi^{d/2}}{\Gamma(d/2)} \int_0^\infty \cos (r \, t/\sqrt{d}) r^{d-1} {e^{- r^2 / 2}} \, dr\\
& = \sigma^2 \frac{1}{\sqrt{(2\pi )^d}} \frac{2 \pi^{d/2}}{\Gamma(d/2)} \frac{1}{2} \Gamma(d/2) 2^{d/2} \, {}_1 F_1 \left(\frac{d}{2}; \frac{1}{2} ;- \frac{t^2}{2 d} \right)\\
& = \sigma^2 \, {}_1 F_1 \left(\frac{d}{2}; \frac{1}{2} ;- \frac{ t^2}{2 d} \right)
\end{align}
$$
The last integral was evaluated using Claude, verified by Mathematica. Note that $| {}_1 F_1 (n; 1/2; - \tau^2) | < 1$ for $n \in \mathbb N$ and $\tau \in \mathbb R$, so it has a property that the autocorrelation has.

So in total, the first moment (projected along the chosen direction) has a lag-1 autocorrelation of

$$

\rho^{(1)} = {}_1F_1 \left(\frac{d}{2}; \frac{1}{2} ;- \frac{t^2}{2d} \right)

$$

Here's a picture of the result. 
![[2026-01-13 First Moment SideHMC.png]]
- The result is affine invariant!
- It's nice to see the algorithm is stable in the infinite dimensional limit. 
	- I wish, the leapfrog time when the autocorrelation is zero, would be independent of dimension...
- In 1-dimension, the algorithm is quite bad. When we look at the second moment autocorrelation, we'll see it's not even defined in one dimension!
- I highly suspect $\lim_{d\to \infty} {}_1 F_1 (\frac{d}{2}; \frac{1}{2} ; - \frac{t^2}{2 d}) = \cos(t)$. Which recovers the perfectly tuned regular HMC.
	- If you just interchange the limit and sum, then it works. Claude also says this is the case, but I can't find an online source which confirms/denies this.
	- I do believe it tho because the 2nd moment will also reduce to the regular HMC in the infinite dimensional limit.
![[2026-01-13 First Moment SideHMC different dimensional scaling.png]]
- I was also wonder what happens if you adjusted the dimensional scaling on the $v = \frac{1}{\sqrt{2d}} (x_i - x_j)$ to $\frac{1}{\sqrt{2d^\alpha}}...$. You end up getting ${}_1 F_1 (\frac d 2 ; \frac 1 2; - \frac{t^2}{2 d^\alpha})$. 
- If $\alpha > 1$ higher dimensional distributions take longer to get good convergence.
- If $\alpha < 1$ you need to integrate for less time (sometimes at numerical precision). More importantly, the time to get zero autocorr seems to not be stable in the infinite dimensional limit.
- If $\alpha = 1$, you get that time (for zero autocorr) to be stable across dimensions.

### (Side) Second-Moment's Lag-1 Autocorrelation
Nothing nice comes for free. Recall the second moment' lag-1 autocorrelation is given by

$$

\rho^{(2)}= \frac{\mathbb E[(x_v^2 - \mathbb E[x_v^2])(\tilde x_v^2 -\mathbb E[x_v^2])]}{\mathbb E[(x_v^2 - \mathbb E[x_v^2])^2]} = \frac{\mathbb E[x_v^2 \tilde x_v^2] - \mathbb E[\tilde x_v^2] \mathbb E[x_v^2]}{\mathbb E[x_v^4] - \mathbb E[x_v^2]^2}

$$

Let's try the denominator

$$

\begin{align}
\mathbb E[x_v^4] & = \mathbb E[(x \cdot  \hat v)^4]\\
& = 3 \sigma^4
\end{align}

$$

To evaluate this, you need to realize that $x \cdot \hat v \sim \mathcal N(0, \sigma^2)$. 

Ok onto the evil numerator

The first term is

$$

\begin{align}
\mathbb E[\tilde x_v^2] & =  \mathbb E \left[ \left (x_v(0) \cos(\|z\| \, t/\sqrt{d}) + \frac{p(0)}{\sqrt{d}} \sin(\|z\| \, t /\sqrt{d}) \right )^2 \right]\\
& = \mathbb E \left[ x_v(0)^2 \cos^2(\|z\| \, t/\sqrt{d}) + \frac{p(0)^2}{d} \sin^2(\|z\| \, t /\sqrt{d}) \right]\\
& = \sigma^2 \, \mathbb E\left [ \cos^2 (\| z \| t / \sqrt{d}) \right] + \frac{1}{d} \mathbb E\left[ \sin^2 (\|z\| t / \sqrt{d})\right]\\
\mathbb E[\tilde x_v^2 x_v^2] & = \mathbb E \left[ \left (x_v(0)^2 \cos(\|z\| \, t/\sqrt{d}) + \frac{p(0) x_v(0) }{\sqrt{d}} \sin(\|z\| \, t /\sqrt{d}) \right )^2 \right]\\
&  =\mathbb E \left[x_v(0)^4 \cos^2(\|z\| \, t/\sqrt{d}) + \frac{p(0)^2 x_v(0)^2}{d} \sin^2(\|z\| \, t /\sqrt{d}) \right]\\
& = 3\sigma^4 \mathbb E[\cos^2(\|z\| \, t /\sqrt{d})] + \frac{\sigma^2}{d} \mathbb E[\sin^2(\|z\| \, t /\sqrt{d}) ]\\
\mathbb E[x_v^2 \tilde x_v^2] - \mathbb E[\tilde x_v^2] \mathbb E[x_v^2] & = 2 \sigma^2 \, \mathbb E[\cos^2(\|z\| \, t /\sqrt{d})] 
\end{align}

$$

For the $\cos^2$ term, we can use the trig identity $\cos^2(x) = \frac{1}{2}(1 + \cos(2 x))$. And because we know how to already compute expectation value of that cosine, we are done. 

Our resulting autocorrelation is

$$
\rho^{(2)} = \frac{1}{2}\left(1 + {}_1F_1 \left(\frac{d}{2}; \frac{1}{2} ;- \frac{2 t^2}{d} \right)\right)
$$

![[2026-01-13 Second Moment SideHMC.png]]
- Once again the result is affine invariant.
- The result mirrors the regular HMC result! *whew*
- Interestingly we have a (sorta) blessing of dimensionality. The achievable autocorrelation gets smaller as the dimension grows. However this just recovers the regular HMC. 

## Walk-Move Leapfrog
The equations of motion

$$

	\frac{d x}{dt} = B \, p, \ \frac{dp}{dt} = - B^T \Sigma^{-1} x

$$

where $B \in \mathbb R^{d \times N/2}$ is the centered positions of the complement ensemble. Note that $p \sim \mathcal N(0, \mathbb I_{N/2 \times N/2})$ this time. To be specific on the structure of $B$, it looks like

$$

B = \begin{pmatrix}
 |  & ... & |\\
 \vec a_1 - \frac{1}{N/2}\sum_{i=1}^{N/2} \vec a_i & ... & \vec a_{N/2} - \frac{1}{N/2}\sum_{i=1}^{N/2} \vec a_i\\
 | & ... & |
\end{pmatrix}

$$

where $\vec a_i$ is the position of the $i$'th complement walker.

The equations of motions can be written by exponentiation the time-evolution operator.

$$

\begin{align}
	\begin{pmatrix} x(t) \\ p(t)\end{pmatrix} & = \exp\left[ t \begin{pmatrix} 0 & B \\ - B^T \Sigma^{-1} & 0 \end{pmatrix}  \right]\begin{pmatrix} x(0) \\ p(0)\end{pmatrix}\\
	x(t) & = \cos(\Omega_x t)x(0) + \Omega^{-1}_x  B \sin(\Omega_x t) p(0)\\
	p(t) & = \cos(\Omega_p) \, p(0) - \Omega_p^{-1} B^T \Sigma^{-1} \sin(\Omega_p t) \, x(0)
 \end{align}

$$

where $\Omega_x = \sqrt{BB^T \Sigma^{-1}} \in \mathbb R^{d \times d}$  and $\Omega_p = \sqrt{B^T \Sigma^{-1} B} \in \mathbb R^{N/2 \times N/2}$.

In the analysis, we'll assume all the walkers (including complement ensemble) are properly mixed and the dynamics are stationary w.r.t. the distribution. I'll choose the target distribution $\pi = \mathcal N(0, \Sigma)$, and choose $\Sigma = \sigma^2 \mathbb I_d$ to save myself a headache. This causes $\Omega_x = \sqrt{BB^T} / \sigma, \Omega_p =  \sqrt{B^T B} / \sigma$. Since the complement ensemble is also mixed, $BB^T \sim W_d(\frac{N}{2} - 1, \sigma^2 \mathbb I_d)$ is drawn from a Wishart distribution. 

### (Walk) First Moment's Lag-1 Autocorrelation
Since the walkers are mixing in a isotropic distribution. I'll analyze the projection against the $i$'th dimension. $x : = x(t) \cdot e_i$ 

Consider the first moment's autocorrelation

$$

\rho^{(1)}= \frac{\mathbb E[(x - \mathbb E[x])(\tilde x -\mathbb E[x])]}{\mathbb E[(x - \mathbb E[x])^2]}

$$

The proposed position

$$

\mathbb E[\tilde x] = \mathbb E[\cos(\Omega_x t)] \mathbb E[x] + \mathbb E[\Omega^{-1}_x  B \sin(\Omega_x t)] \mathbb E[ p(0)] = 0

$$

The overlap between proposed & original position
$$

\begin{align}
\mathbb E[\tilde x x] & = \mathbb E[\cos(\Omega_x t)] \mathbb E[x^2]\\
& = \sigma^2 \mathbb E[\cos (\Omega_x t)] 
\end{align}

$$
where $\Omega_x = \sqrt{BB^T} / \sigma$.  So as alluded to in [[2026-01-15 Expectations of Wishart Distributions]], this will end up being a trace, but we need to classify the eigenvectors & eigenvalues of the sqrt of a Wishart distribution.

Consider two matricies $A = MM^T$ and $B = (MM^T)^{1/2}$, where the square root is constructed via eigenvalue decomposition. That is if $B = V \Lambda V^T$ (eigen-decomposition), then $B^2 = A =  V\Lambda^2 V$. This means the eigenvalues of $A$ are just squared of the eigenvalues of $B$, and they share the same eigenvectors. Wow!

Going back to our problem. $BB^T \sim W_d(\frac{N}{2} - 1, \sigma^2 \mathbb I_d)$. This means that $(BB^T)^{1/2}$ has a similar distribution. A fact about Wishart matricies is that the eigenvectors and eigenvalues are independent random variables, therefore

$$

\begin{align}
\mathbb E[\cos (\Omega_x t)]  & = \mathbb E[\cos((BB^T)^{1/2} t/ \sigma)]\\
& = \mathbb E[\cos(\sqrt{\lambda} t/\sigma)] \mathbb I_d
\end{align}

$$
$\lambda$ has some probability distribution, but if we take $N,d\to \infty$ while keeping $\frac{d}{N/2 - 1} =: \gamma$ constant, we can use the Marchenko–Pastur distribution $\lambda \sim \text{MP}(\gamma, \sigma)$. One final thing to note, $\lambda / \sigma^{2} \sim \text{MP}(\gamma, 1)$.

Finally, our lag-1 autocorrelation is given by
$$

\rho^{(1)} = \mathbb E_{\text{MP}(\gamma, 1)}[\cos(\sqrt{\lambda} \, t)]

$$
![[2026-01-13 First Moment WalkHMC.png]]
- It's affine invariant *whew*
- Just keep in mind $\gamma = \frac{d}{N/2 - 1}$, when $d,N \to \infty$.  This algorithm only has affine invariance when $N/2 > d$.
	- You'd expect as $N/2 \gg d$ (meaning $\gamma \to 0$) the algorithm would perform more favorably. I suppose we see this in that there are more $t$'s which have zero autocorrelation.
	- I was hoping there was a method of seeing it's autocorrelation keeping $d/N$ fixed, but increasing $d$.

### (Walk) Second Moment's Lag-1 Autocorrelation
I don't know if we can calculate this one... But this is what I expect.
$$

\rho^{(2)} = \frac{1}{2}\left(1 + \mathbb E_{\lambda \sim \text{MP}(\gamma,1)}[\cos(\sqrt{\lambda} \, 2t)] \right)

$$
![[2026-01-15 Second Moment WalkHMC.png]]
- As expected, as the number of walkers increase, then you get better autocorrelation.