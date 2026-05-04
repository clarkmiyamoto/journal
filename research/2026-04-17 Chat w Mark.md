#architecture #stochasticinterpolants 

# What is happening?
**Setup:** Solving inverse problems. 
$$
\begin{align}
\mathcal F_{\text{MRA}}(X) = G \cdot X. + W
\end{align}
$$
My toy-problem setup, multi-reference alignment
- $X$ is an MNIST image
- $G \sim \mathcal G$ (sampled uniformly from translations w/ periodic boundary conditions) and $W \sim \mathcal N(0, \sigma^2 \mathbb I)$
Therefore $\mathcal F(X)$ is a randomly translated and noised MNIST image. 

We want to generate a plausible $X$ given an observed $y = \mathcal F(x)$.

*Solution Method:* Consider the self-consistent stochastic interpolant
$$
\begin{align}
I_t^{(k+1)} | Y=y & \overset{(d)}{=} \alpha_t Z + \beta_t \Phi_{\Theta^{(k)}}(Z' | y)\\
Y & = \mathcal F_{\text{MRA}}(X)
\end{align}
$$
- $Y$ is randomly translated & noised MNIST. $X$ is clean MNIST. We only have a dataset for $Y$, so can't place $X$ in the interpolant.
- $\Phi_{\Theta^{(k)}}$ is the ODE flow induced by $I^{(k)}$, initalized using a different white noise $Z'$, conditoined on $y$. So you hope as we increment $k$, $\Phi_{\Theta^{(k)}}$ converges to a fixed point the inverse problemthat solves the inverse problem. That is ideally $\Phi_{\Theta^{*}}(Z | Y) \overset{\text{want}}{=} X$. 
- $Z,Z' \sim \mathcal N(0, \mathbb I)$

**Neural network's objective:** We are learning the drift field
$$
\begin{align}
b_t^{\textsf{opt}}(x|y) = \mathbb E[\dot I_t | I_t = x , Y =y]
\end{align}
$$
This places the crux of the problem on how to do conditioning with the neural network.

### Gaining intuition with a worked example
Let $X \sim \mathcal N(\mu, \Sigma)$, therefore $Y = \mathcal F(X) \sim \frac{1}{|\mathcal  G|}\sum_{g \in \mathcal G} \mathcal N(g\mu, g^T \Sigma g + \sigma^2 \mathbb I)$

I can derive an analytic expression for the drift
$$
\begin{align}
	\boxed{b_t(x|y) = \frac{\dot \alpha_t}{\alpha_t} x + \left( \dot \beta_t - \frac{\dot \alpha_t}{\alpha_t} \beta_t \right) \sum_{g \in \mathcal G} p(g | I_t = x, Y=y)  \, m_{g,t} (x,y)}
\end{align}
$$
where
$$
\begin{align}
	p(g | I_t = x, Y=y)  & = \frac{\mathcal N \left(\begin{pmatrix}
		x \\ y
	\end{pmatrix};\begin{pmatrix}
			\beta_t \mu_g \\ \mu_g
		\end{pmatrix}, \begin{pmatrix}
			\beta_t^2 \Sigma_g + \alpha_t^2 \mathbb I & \beta_t \Sigma_g \\ \beta_t \Sigma_g & \Sigma_g + \sigma^2 \mathbb I
		\end{pmatrix} \right) }{\sum_{g' \in \mathcal G}\mathcal N\left( \begin{pmatrix}
		x \\ y
	\end{pmatrix};\begin{pmatrix}
			\beta_t \mu_{g'} \\ \mu_{g'}
		\end{pmatrix}, \begin{pmatrix}
			\beta_t^2 \Sigma_{g'} + \alpha_t^2 \mathbb I & \beta_t \Sigma_{g'} \\ \beta_t \Sigma_{g'} & \Sigma_{g'} + \sigma^2 \mathbb I
		\end{pmatrix} \right)}\\
		m_{g,t}(x,y) & = \left( \Sigma_g^{-1} + \left(\tfrac{1}{\sigma^2} + \tfrac{\beta_t^2}{\alpha_t^2}  \right) \mathbb I\right)^{-1} \left(\Sigma_g^{-1} \mu_g + \tfrac{1}{\sigma^2} y + \tfrac{\beta_t}{\alpha_t^2} \, x  \right)
\end{align}
$$
> ❗️Reminder: show photos of the ODE flow.

A property of the drift is that it's jointly equivariant
$$
\begin{align}
b_t^{\textsf{opt}}(g \cdot x| g \cdot y) = g \cdot b_t^{\textsf{opt}}(x|y)
\end{align}
$$


## Current Architecture
Let
- $b_t^\theta (x | y)$ be the neural network.
- $\tau = \text{FourierEmbeddings}(t)$
- $\mu_\theta(\tau), \lambda_\theta(\tau) := \text{ScalarMLP}_\theta(\tau) \in \mathbb R$
$$
\begin{align}
b_t(x | y) = \mu^\theta(\tau) x + \lambda^\theta(\tau)\ \text{UNet}^\theta_\tau(x|y)
\end{align}
$$
However, my UNet is modified from the usual.
### UNet
Encoder is a composition of these blocks.
$$
\begin{align}
h_1 & = \phi\!\left(\mathrm{FiLM}_1\!\left(\mathrm{GN}_1\!\left(\mathrm{Conv}^{\circ}_1(h)\right), t_{\text{feat}}\right)\right),\\
h_2 & = \phi\!\left(\mathrm{FiLM}_2\!\left(\mathrm{GN}_1\!\left(\mathrm{Conv}^{\circ}_2(h_1)\right), t_{\text{feat}}\right)\right),\\
\text{Block}(h, t_{feat})& = h_2 + \text{Skip}(h)
\end{align}
$$
where $\phi$ is the SiLU activation function, $\mathrm{FiLM}(u,t)=u\odot(1+s(t))+b(t)$, and $\text{GN}$ is group norm. The decoder is the same.

The $x$ and $y$ get processed separately, then conjoined at the middle block.

We expect some sort of group average over $\mathcal G$, so let's just hardcode that in the network.

Define learned maps:
- $\phi_y(y)=\text{Conv} \circ \text{Encoder} \circ y$ 
- $\phi_x(x) = \text{Conv} \circ \text{Encoder} \circ x$
- $\psi_x(x) = \text{Conv} \circ \text{Encoder} \circ x$

Then:
$$
\begin{align}
\ell_g & = \frac{1}{\tau_t}\left\langle \phi_y(y),\; T_g\phi_x(x)\right\rangle\\
w_g & = \frac{\exp(\ell_g)}{\sum_{g' \in G} \exp(\ell_{g'})}\\
\mathrm{out} & = \sum_{g \in G} w_g \, T_g \psi_x(x)
\end{align}
$$
where
- $\langle \cdot,\cdot \rangle$ sums over channels and spatial locations.
- $\tau_t > 0$ is a time-dependent temperature.

So the block does three things:
1. Score all translations of \(x\) against \(y\).
2. Softmax the scores over translations.
3. Average translated value maps using those weights.

# notes


$$X_t = \alpha_t X_0 + \beta_t X_1$$
noise is $X_0$ and data is $X_1$ 

then $\nabla \log p_t(x) = -\frac{1}{\alpha_t}E[X_0 | X_t = x]$ 



