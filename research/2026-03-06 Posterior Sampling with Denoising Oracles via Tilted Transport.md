Paper by Joan Bruna and Jiequn Han
https://arxiv.org/pdf/2407.00745

# Notation
**Quadratic Tilt**
The quadratic tilt operation $T_{Q,b}$ transforms a base measure $\pi$ by
$$
\begin{align}
T_{q,b} \, \pi(x) \propto \exp(-\tfrac 1 2 x^T Q x - b^T x) \pi(x)
\end{align}
$$

**SDEs**
OU Semigroup (variance preserving)
$$
\begin{align}
dX_t & = - X_t dt + \sqrt 2 dW_t,\; X_0 \sim \pi\\
X_t &= e^{-t} X_0 + \sqrt 2 e^{-t} \int_0^t e^s \, dW_s \\
(X_t | X_0 = x)& \overset{d}{=}  \mathcal N(e^{-t} x , 1-e^{-2t})
\end{align}
$$ 
We say that $\pi_t = O_t^* \pi$, where $O_t f(x) = \mathbb E[f(X_t) | X_0 = x]$ is the OU semigroup. Explicitly, it is given by dilated Gaussians convolutions, $O_t^* := C_{\beta_t} D_{\alpha_t}$, with $\beta_t = 1 - e^{-2t}$ and $\alpha_t = e^t$. 

**Dilation Operator** $D_a \pi(x) := a^d \pi(a \, x)$
- The $a^d$ comes out to keep it normalized
**Convolution Operator** $C_\beta \pi(x) := \pi * (D_{\beta^{-1/2}} \gamma_d)$

Heat Semigroup (variance exploding)
$$
\begin{align}
H_t^* \pi = \pi * \gamma_{2t}
\end{align}
$$

# Setup
Suppose you have a trained generative model for $\pi$ using the DDPM objective. I'll denote the model as the **denoising oracle**
$$
\begin{align}
\text{DO}_\pi(y, \sigma) := \mathbb E[x|y]
\end{align}
$$
where $y = x + \sigma w$ with $x \sim \pi$ and $w \sim \gamma_d := \mathcal N(0, d \mathbb I_d)$ independent. 

Suppose we now measure $y = Ax + \sigma w$, where again $x \sim \pi$ and $w \sim \gamma_d$ are independent, but now $A \in \mathbb R^{d' \times d}$ is a known linear operator different (which is not identity). We are interested in a posterior sampling of $x$ given $y$. We'll denote this posterior as $\nu_{y,A}$ as
$$
\begin{align}
\nu_{y,A}(x) := p(x|y) \propto \exp \left( - |A x - y|^2 / 2\sigma^2 \right) \pi(x)
\end{align}
$$
thus we can write it as the quadratic tilt of $\pi$
$$
\begin{align}
\nu = T_{Q,b} \, \pi,\; \text{ where } Q = \sigma^{-2} A^TA ,\; b=-\sigma^{-2} A^T y
\end{align}
$$

The research question: **can the power of denoising oracles be provably transferred to posterior sampling?**

# Solution: Posterior Sampling via Tilted Transport
Sampling the target $\pi$ with the denoising oracle $\text{DO}_\pi$ is easy. We can use the Fokker Planck equation to start at a Gaussian $\pi_{t=T}$, and then transport samples back to $\pi_{t=0} = \pi$.

Can we replicate this scheme to transport $\nu_{t=T}$ and a target posterior $\nu$, which only relies on a pre-trained prior $\text{DO}_\pi$?

**Hamilton Jacobi Equation**
Let $\pi_t$ solve the Fokker Planck equation
$$
\begin{align}
\partial_t \pi_t = \nabla \cdot(\mu_t \, \pi_t) + \Delta \pi_t,\; \pi_0 = \pi
\end{align}
$$
Now consider the time-evolution on the potential $f_t := \log \pi_t$
$$
\begin{align}
\partial_t \pi_t & = \partial_t f_t\, \pi_t\\
\nabla \cdot (\mu_t \pi_t)& = [(\nabla \cdot \mu_t) + \mu_t \cdot \nabla f_t]  \pi_t\\
\Delta \pi_t & =  [\Delta f_t + (\nabla f_t)^2] \pi_t
\end{align}
$$
You can see this now solves a Hamilton-Jacobi equaiton
$$
\begin{align}
\partial_t f_t = \nabla \cdot \mu_t + \mu_t \cdot \nabla f_t + \Delta f_t + \|\nabla f_t\|^2
\end{align}
$$
In the OU semigroup $\mu_t(x) = x$, so $\nabla \cdot \mu_t = d$ Getting 
$$
\begin{align}
\partial_t f_t = d + \Delta f_t + \|\nabla f_t\|^2 + x \cdot \nabla f_t, \; f_0 = f
\end{align}
$$
In the heat semigroup, $\mu_t(x) = 0$, so you get
$$
\begin{align}
\partial_t f_t = \Delta f_t + \|\nabla f_t\|^2
\end{align}
$$
One would hope that quadratic tilts of the original measure $\nu = T_{Q,b} \pi$ (in log space $\log \nu = f - \frac 1 2 x^T Q x + x \cdot b$) would still solve this equation. But due to the non-linear term $\|\nabla f_t\|^2$, your hopes are slightly ruined. 

**Hamilton-Jacobi Equation in Reverse Time**
The backwards Fokker Planck is given by
$$
\begin{align}
\partial_t p_t =  \nabla \cdot ((\mu_t + 2 \nabla \log \pi_t)p_t) - \Delta p_t
\end{align}
$$
This leads to the HJ equation
$$
\begin{align}
\partial_t g_t = \nabla\cdot \mu_t + \Delta f_t -(\Delta f_t + (\nabla f_t)^2)
\end{align}
$$
where $g_t := \log p_t$

**Tilt Transport Equation**
The trick is to consider time-varying quadratic tilts $T_{Q_t, b_t}$, and then solve their dynamics. Let
$$
\begin{align}
\nu_t := T_{Q_t, b_t} \pi_t, \; Q_0 = Q, b_0 = b
\end{align}
$$
Let's compute how $\log \nu_t = f_t - \frac 1 2 x^T Q_t x + x \cdot b_t$ evolves.
$$
\begin{align}
\partial_t f_t & = \partial_t \log \nu_t + \frac 1 2 x^T \dot Q_t x - x \cdot \dot b_t\\
\nabla f_t& = \nabla \log \nu_t + Q_t x - b_t\\
\Delta f_t & = \Delta \log \nu_t + \text{Tr}(Q_t) \\
\|\nabla f_t\|^2 & =  \|\nabla \log \nu_t + Q_t x - b_t\|^2 \\
& = \| \nabla \log \nu_t\|^2 + \|Q_t x - b_t\|^2 + 2 (\nabla \log \nu_t ) \cdot (Q_t x - b_t)
\end{align}
$$
I want to see how the tilted distribution $\nu_t$ performance at test time. So let's look at how it solves the reverse-time HJ-equation. Let's start w/ the OU semigroup. In this case $\mu_t = -(x + \nabla f_t)$
$$
\begin{align}
\partial_t f_t & = \nabla \cdot \mu_t + \mu_t \cdot \nabla f_t + \Delta f_t + \|\nabla f_t\|^2\\
& = -\nabla \cdot (x + \nabla f_t) - (x + \nabla f_t) \cdot \nabla f_t + \Delta f_t + \|\nabla f_t\|^2\\
& = -d 
\end{align}
$$

Plugging this into the HJ-equation on $f_t$, and requiring $\log \nu_t$ to solve it's own HJ-equation, we find
$$
\begin{align}
\frac 1 2 x^T \dot Q_t x - x \cdot \dot b_t = \text{Tr}(Q_t)+ \|Q_t x - b_t\|^2 + 2 (\nabla \log \nu_t ) \cdot (Q_t x - b_t) + x^T Q_t x - x \cdot b_t
\end{align}
$$
The dependence on $\nabla \log \nu_t$ prevent $Q_t,b_t$ from having nice equations of motion. Instead we can ask 