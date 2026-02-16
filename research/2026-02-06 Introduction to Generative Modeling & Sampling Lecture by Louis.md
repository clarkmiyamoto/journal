#sampling #bayesianinference #generative #diffusion #normalizingflows
Lecture: [[Louis Grenioux]] @ CCM Flatiron

```table-of-contents
```
# Week 1: Generative Modeling Setup
## Setup
**Generative Modeling:** You're given a target distribution $\pi$. Given a family of parameterized distribution $\{p^\theta\}_\theta$, your goal is to find the "closest" to the target, by only have access to a dataset $\mathcal D = \{x_i\}_i$ of samples drawn from the target $x_i \sim \pi$.

Hopefully our solution can generate samples that weren't in the original dataset, but seem like they came from $\pi$.

So a couple questions come out
- What is my family $\{p^\theta\}_\theta$?
- Given the dataset $\mathcal D$, how do I find the best $\hat \theta$?
- Once I find it, how do I sample from it?

## The Family of Parameterized Distributions 
### Normalizing Flows
The idea is to learn a pushforward that moves you from a base $\rho$ to the target $\pi$. Hopefully, you can sample $\rho$ iid, therefore you can sample $\pi$ in an iid way as well.

In particular the pushforward is a diffeomorphism (differentiable and invertible). This implies $\pi$ and $\rho$ are both on the same state space. Consider 
$$
y \sim p^\theta  \iff y = T^\theta(z) , z \sim \rho
$$
Notice by change of variable formula you can access the density of the model via the density of the base $\rho$
$$
p^\theta(x) = p(T^{-1}_\theta(x)) |J_{T^{-1}_\theta}(x)|
$$
For today, we'll take this to be our family $\{p^\theta\}_\theta$.

There's a couple of requirements of our parameterization
- Getting this in one pushforward, would make $T$ a very complex function. So instead we iteratively break it up to make it simple
	$$
	T^\theta = T_0^\theta \circ T_1^\theta \circ ... \circ T^\theta_{k-1}
	$$
- We hope that $T^\theta$ is easily invertible. Note that a regular MLP is not invertible, so we need to construct this in a slick way...
- Also evaluating the Jacobian $J_{T_\theta}$ should also be quick!

An example is **Real NVP** which has the structure
$$
\begin{align}
T^\theta & : \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} \mapsto \begin{pmatrix} x_1 + f_\theta(x_2) \\ x_2\end{pmatrix}\\
(T^\theta)^{-1} & :\begin{pmatrix} y_1 \\ y_2\end{pmatrix} \mapsto \begin{pmatrix} y_1 - f_\theta(y_2) \\ y_2\end{pmatrix}\\
J_T & =  \begin{pmatrix}1 & ... \\ 0 & 1 \end{pmatrix}
\end{align}
$$where $f_\theta$ is a neural network. The $T^{-1}$ doesn't inverting a neural network, so you can make $f_\theta$ as complex as you want. And the det Jacobian is just 1!

There are theoretical results that show that normalizing flows can't express multimodal distribution. However with diffusion models, you construct a Markov chain which can!
### Diffusion Models
Consider an SDE
$$
dX_t = b_t(X_t) \, dt + \sigma_t(X_t) dW_t, \ x_0 \sim \pi
$$
where $\pi$ is the target density.  We can choose $b_t(X_t) = f(t) X_t$ and $\sigma_t(X_t) = g(t)$, and this would make it into an OU process which has a variable noise scheduler
$$
dX_t = f(t) X_t \, dt + g(t) \, dW_t
$$
And you choose $f,g$ such that a later time $X_T \sim \rho$ you're sample the base distribution. You can solve this
$$
\begin{align}
X_t & = X_s + \int_s^t f(u) X_u\, du + \int_s^t g(u) dB_u\\
& = \exp \left( \int_s^t f(u) du \right)  X_s + \int_s^t g(u) \exp \left( \int_u^s f(n) dn \right) dB_u\\
& = \underbrace{\exp\left( \int_s^t f(u) du \right)}_{\alpha_{t|s}} X_s + \underbrace{\sqrt{\int_s^t g^2(u) \exp \left( 2 \int_u^s f(n) \, dn\right) du}}_{\sigma_{t|s}} \, Z
\end{align}
$$
where $Z \sim \mathcal N(0,1)$. In the first move, I use Ito's lemma. And the 2nd move, I use Ito's Isometry.

Consider the simple case $f=0$ and $g=1$. Then $\alpha_{t|s} = 1$ and $\sigma_{t|s} = t-s$. It has the solution
$$
X_t = X_0 +  t \, Z
$$
As $t \gg 1$, it destroys the data contained in $X_0 \sim \pi$ and you end up with just white noise.

In the generic solution, you can write this as
$$
P_{X_t | X_s = X_s} (x_t) = P_{t|s} (x_t | x_s) = \mathcal N(x_t ; \alpha_{t|s} x_s , \sigma_{t|s}^2 \mathbb I)
$$
You can use the reverse SDE to get
$$
dX_t = [f(t) X_t - g^2(t) \nabla \log p_t(X_t)] \, dt + g(t) d \tilde B_t, \ X_T \sim p
$$
where you solve it backwards in time. It shares the same marginal distribution as the forward SDE. An important quantity is the score $\nabla \log p_t(X_t)$ 

Euler discretization of the SDE gives
$$
\hat P_{s|t}(x_s | x_t) = P(X_s; x_t - (t-s)) [f(t) X_t - g^2(t) \nabla \log p_t(x_t)] 
$$

A comment on the marginal score 
$$
p_t(x_t) = \int p_{t|0}(x_t | x_0) \pi(x_0) \, dx_0
$$where $\pi$ is the target. It's quite difficult to compute this integral, thus it's difficult to compute $\nabla \log p_t(x_t)$. So we'll use a neural network to best approximate the distribution $s_t^\theta \approx \nabla \log p_t(x_t)$.

So your parametric family
$$
p^\theta(x_{0:k-1}) = p(x_{k-1}) \prod_{i=0}^{k-2} \hat p_{k | k+1}^\theta (x_k | x_{k+1})
$$
It may seem there are $k$ neural networks, but in practice you use one neural network which takes in $k$ are an input parameter.

## How do we find the best $\hat \theta$?
### KL Divergence (Generative Model setting)
We'll minimize the KL divergence between the target and the normalizing flow
$$
\begin{align}
	\mathcal L(\theta) = \text{KL}(\pi \| p^\theta) & = \mathbb E_\pi \left[ \log \frac{\pi(x)}{p^\theta(x)}\right]\\
	& = - \mathbb E_\pi [\log p^\theta(x)] + \text{Constant}\\
	& = - \frac{1}{| \mathcal D |} \sum_{x \in \mathcal D} \log p^\theta(x) + \text{Constant}
\end{align}
$$
A statistician would now call this a likelihood! Now you can minimize this using your normal techniques.

### Best Candidate (in mean squared error)
Recall Tweedie's formula. The best (in mean squared error) score which will denoise for our forward process is given by:
$$
\nabla \log p_t(x_t) = \mathbb E \left[ - \frac{x_t - \alpha_{t|0} x_0}{\sigma^2_{t|0}} \Big | X_t = x_t\right]
$$
Recall (from linear regression) if 
$$
f(x) = \mathbb E[Y | X=x] \iff f = \arg \min_f
 \mathbb E[\| f(X) - Y\|^2]
$$
This means you can find the best score as an optimization problem
$$
\nabla \log p_t = \arg \min_{\hat s_t}  \underbrace{\mathbb E \left[ \left \|\hat s_t(x_t) - \left(\frac{x_t - \alpha_{t|0} x_0}{\sigma^2_{t|0}} \right) \right \|^2 \right]}_{\mathcal L(\theta)}
$$
This constructs our loss function!

## Papers
KL Divergence Method
- https://arxiv.org/pdf/2406.07423. Methods for sampling (including diffusion mdoels), also gives benchmark problems!
- https://arxiv.org/abs/2502.06685. Somewhat of a review, other ways to generative modeling sampling

Tweedy's Formula
- Stochastic Localization via Iterative Posterior Sampling, ICML 2024

# Chat w/ Louis After the Lecture, How to Sample Distributions
How do we learn diffusion models w/o data. Louis prefer the diffusion time (t=0 is target $\pi$ and after K steps you're at the simple/base distributoin $\rho$)

$$
\pi(x_{0:k})   \pi(x_0) \prod_{k=0}^{k-2} p_{k+1  | k}(x_{k+1} | x_k)
$$
where $p_{k+1|k}$ is the noising kernel. Using the reverse SDE, we derive the reverse process. 
$$
p^\theta(x_{0:k-1}) = p(x_{K-1}) \prod_{k=0}^{K-2} p_{k | k+1} p(x_k | x_{k+1})
$$
where the denoising kernel is given by

One way to denoising is to construct the **forward KL**. This is the **generative models** case
$$
\text{KL} (\pi | p^\theta) = ... = \sum_k \mathbb E[\| s^\theta_{t_k}(x_k) - \frac{z}{\sigma_{t_{k+1} | 0}} \|^2]
$$
In the **Sampling case** you get the **reverse KL**
$$
\begin{align}
\text{KL}(p^\theta(x_{0:k-1}  | \pi(x_{0:k-1}))) & = \mathbb E_{p^\theta} \left[ \log \frac{p^\theta}{\pi}\right]\\
& = \mathbb E_{p^\theta} \left [ \right]
\end{align}
$$
- Reverse KL ends up having mode seeking behavior. But this is not a problem because it's not multimodal in the joint! You have tunnels!
- This is not useful in stochastic interpolants. Only useful in diffusion because you have an analytic forward pass..
- Another benefit compared to stochastic interpolants is you only have to learn one function (the denoising oracle / score), vs SI you need velocity & diffusion field

No one has ever done a paper, Louis has an example in an appendix in a paper.
The reason is you need covarge at the beginning and ending, so you need your approximated score to be well initalized...

The previous discussion was in discrete time, but you can do this in continuous time using Girsonov's theorem + the integration by parts score matching trick you get
$$
\text{KL}(P \| P^\theta) = \int \mathcal L_{DSM} \, dt
$$
Looking at the reverse KL. If you apply Girsonv's theorem you'll see you won't be able to get rid of a call to $\nabla \log p_t$. But if you introduce a reference measure $\pi^*$ you get
$$

$$
The idea is to use the emperical distribution over samples that an MCMC algorithm generates as your $\pi^*$. That is SOTA. Louis also has a paper where he learns $\pi^*$ (Google Louis + Reference)...

The scientific community has seemed to converge that this will be a dead end...
## Non Distance/KL Criterions
Recall Tweedie's formula
$$
\begin{align}
\nabla \log p_t(x_t) & = \mathbb E \left[ \frac{x_t - \alpha_{t|0} x_0}{\sigma^2_{t|0}} \Big | X_t = x_t\right]
\end{align}
$$
$$
= \int \frac{-x_t - \alpha_{t|0} x_0}{\sigma^2} p_{0|t}(x_0 | x_t) \, dx_0
$$
and you see $p_{0|t} \propto p_{t|0}(x_t | x_0) \pi(x_0)$, where $p_{t|0}$ is our noising kernel. So you can use MCMC to sample the backwards process (written similarly to a posterior)!

In the previous method every condition expectation was linked an optimization problem, but now it's a sampling problem.

# Week 2: Louis Conti.
## Recap

|                                   | Normalizing Flows                                                                                       | Diffusion Models                                                                                                                                                                                                              |
| --------------------------------- | ------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **User defines**                  | Base distribution                                                                                       | Noise schedule $f, g$                                                                                                                                                                                                         |
| **Parameterization**              | $T_{\theta} = T_{0\mid1}^{\theta} \circ \cdots \circ T_{K-2\mid K-1}^{\theta}$                          | Scores $s_1^{\theta}, \ldots, s_{K-1}^{\theta}$                                                                                                                                                                               |
| **Inference**                     | $X_h = T_{h \mid h+1}^\theta (X_{h+1})$, where $X_{K - 1} \sim \rho$. Transformation are deterministic. | $X_{h }\sim p_{h \mid h+1}^\theta(\cdot \mid X_{h+1}) = \mathcal N(...)$. Transforms are non-deterministic.                                                                                                                   |
| **Optimum** denoted by $\theta^*$ | $\theta^* = \arg \min_\theta \text{KL}( \pi \| p^\theta)$                                               | $\theta^* = \arg \min_\theta \mathbb E_{k,x,z}[\| s^\theta_k (\alpha_{k\mid 0} x + \sigma_{k \mid -} z) + z / \sigma_{k \mid 0}\|^2]$<br>where $k \sim \text{Unif}$, $x \sim p_{data}$, and $z \sim \mathcal N(0, \mathbb I)$ |

Some comments
- The optimum score matching objective is convex in the function $s^\theta$, while the flows objective is not convex in the functions $T^\theta$. The heuristic idea is that your function has less local minima, so the score matching objective is much nicer.

## Sampling
### Setup
**Given:** You have oracle access to the target distribution $\pi$ (which may or may not be unnormalized).
**Want:** To compute expectations from $\pi$, that is $\mathbb E_\pi[f(x)]$ 
**Method:** Given $\pi(x)$ we want to find $\theta^*$ s.t. $p^{\theta^*} \approx \pi$.

This is very different than the setup last week, because you were given samples from the target distribution, but no oracle access.



## Normalizing Flows?
Recall the training objective
$$
\text{KL}(\pi \| p^\theta) = \mathbb E_{\pi} \log \frac{\pi}{p^\theta}
$$
But we can't compute $\mathbb E_{\pi}$, previously you had those samples so this was a solved problem. Instead you should use
$$
\text{KL}(p^\theta \| \pi) = \mathbb E_{p^\theta} \log \frac{p^\theta}{\pi}
$$
However, taking the gradient of this is horrible. As you take derivatives of this expectation, it's a noise estimate. A draw back is this has mode-seeking behavior; we'll find that diffusion will fix this problem.
### Importance Sampling
Recall the quick trick of importance sampling
$$
\begin{align}
\mathbb E_{\pi}[f(x)] & = \mathbb E_{p^\theta} \left[\frac{\pi(x)}{p^\theta(x)} f(x) \right]\\
& \approx \frac{1}{N} \sum_i w_i f(x_i)
\end{align}
$$
where $w_i = \pi(x_i) / p^\theta(x_i)$, where $x_i \sim^{iid} p^\theta$. However you only get this if your estimator is of finite variance, which is quite difficult in practice. In particular, as the dimension of the distribution increases, you need to get exponential accuracy in KL; however your optimization gets exponentially worse in dimension, so this is quite hard in high dimensions (above 1000s of dimensions).

In practice,  $p^\theta$ ends up being close enough to $\pi$, so you don't have to worry about this.

### MCMC
Recall that you could use the normalizing flow as a pushforward
$$
p^\theta = p(T_\theta^{-1}(x)) | J_{T^{-1}_\theta}(x)| \approx \pi(x)
$$
It happens there was another formula using the pushback
$$
q^\theta(z) = \pi(T_\theta(z)) |J_{T_\theta}(z)| \approx p_{easy}(z)
$$
where $z \sim p_{easy}$. 

The new idea is to sample
- Sample $q^\theta(z)$ with anything (MCMC)
- Then you get samples by using the pushback $x_i = T_\theta(z_i)$.





