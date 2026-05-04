#stochasticinterpolants 

Thinking about [[2026-04-01 Chat w Joan]]. We talked a lot about how to construct lifted interpolants. I don't think I full appreciate that so here's some stuff about me thinking about it.

Consider a forward model $\mathcal F: \mathcal X \to \mathcal Y$. 

Typically when $\mathcal X = \mathcal Y$, we can construct a stochastic interpolant
$$
\begin{align}
I_t = \alpha_t \, x + \beta_t \, \mathcal F(x)
\end{align}
$$
Since $x , \mathcal F(x)$ live on the same space, addition is well defined. But when $\mathcal X \neq \mathcal Y$, the addition can become ill defined. 

There's a couple ways to fix the problem.
# Solution 1: Lifting

## Setup
Lift the inference space to $\Omega = \mathcal X \times \mathcal Y$, and construct a forward model on the lifted space
$$
\begin{align}
\tilde{\mathcal F}  : \Omega &\to \Omega\\
 (x, y) & \mapsto (w^x, \mathcal F(x)), \qquad w^x \sim \mathcal A
\end{align}
$$
where $x, w^x \in \mathcal X$, and $y, \mathcal F(x) \in \mathcal Y$. The $\mathcal A$ is an auxiliary distribution, which lives on the same space as $x \sim \pi$ (clean data distribution distribution); usually $\mathcal A = \mathcal N(0, \mathbb I_{\text{dim}(\mathcal X)})$.

So now consider the lifted interpolant $\tilde I_t = (I_t^x, I_t^y) \in \Omega$. On the $\mathcal X$ space, you move $\pi$ to $\mathcal A^x$; on the $\mathcal Y$ space, we move $\mathcal A^y$ to $\mu$. Let $\tilde x = (x, w^y) \in \Omega$, where $x \sim \pi$ and $w^y \sim \mathcal A^y$. 
$$
\begin{align}
\tilde I_t & = \alpha_t \, \tilde x + \beta_t \, \tilde{\mathcal F}(\tilde x)\\
I_t^x & = \alpha_t \, x + \beta_t \, w^x\\
I_t^y & = \alpha_t \, w^y + \beta_t \, \mathcal F(x)
\end{align}
$$
If we set $\mathcal A^x, \mathcal A^y$ as Gaussians in their according spaces, we can think of this has two diffusions. One learning to generate $x \sim \pi$ (clean distribution), and another noising corrupted data $\mathcal F(x) \sim \nu$. 

## Self Consistent Scheme
In the inverse problem setting, you only have access to empirical samples $y \sim \nu$ and the forward model $\mathcal F(x) = y$. The idea is to use a backwards solver $\tilde \Phi_{\Theta^{(k)}}$ takes in $\tilde y = (w^x, y)$ and outputs $\tilde \pi = \pi \otimes \mathcal A^y$. Explicitly
$$
\begin{align}
\tilde \Phi_{\Theta^{(k)}}(\tilde y) = (\Phi^x_{\Theta^{(k)}}(w^x | y), \Phi^y_{\Theta^{(k)}}(w^x | y)) = (\hat x, \hat w^y)
\end{align}
$$
The lifted SCSI becomes
$$
\begin{align}
\tilde I_t^{(k+1)} & = \alpha_t \tilde \Phi_{\Theta^{(k)}}(\tilde y) + \beta_t \tilde{\mathcal F}(\tilde \Phi_{\Theta^{(k)}} (\tilde y))\\
(I_t^x)^{(k+1)} & = \alpha_t \Phi_{\Theta^{(k)}}^x(w^x| y) + \beta_t w^x \\
(I_t^y)^{(k+1)} & = \alpha_t \Phi^y_{\Theta^{(k)}}(w^x| y) + \beta_t \mathcal F(\Phi^x_{\Theta^{(k)}}(w^x| y))
\end{align}
$$

# Solution 2: Conditioning
You could also solve the inverse problem by asking the interpolant to solve a conditional generation problem. Your objective is to learn to sample $x | \mathcal F(x)$.

Consider the lifted interpolant $\tilde I_t = (I_t^x, I_t^y)$
$$
\begin{align}
I_t^x & = \alpha_t x + \beta_t w \\
I_t^y & = y
\end{align}
$$
where $x \sim \pi$ and $y \sim \nu$. The velocity field satisfies $\tilde b(t, \tilde x) = \mathbb E[\dot{\tilde I}_t | \tilde I_t = \tilde x] = (b_t(x | y) , 0)$.

In the inverse problem, you don't have access to $x \sim \nu$. But only $y \sim \nu$ and $\mathcal F(x) = y$.
$$
\begin{align}
(I_t^x)^{(k+1)} & = \alpha_t \Phi_{\Theta^{(k)}}(w^x | y) + \beta_t w\\
(I_t^y)^{(k+1)} & = \mathcal F( \Phi_{\Theta^{(k)}}(w^x | y))
\end{align}
$$
The ODE satisfies $\tilde b(t, \tilde x) = (b_t(x | y), 0)$.

# Notes from the meeting

## SI Algorithm
1. For $\_$ in $1,2,..., N_{epoch}$:
	1. Sample $x \sim \pi$ (clean data)
	2. $t \sim \text{Uniform}([0,1])$
	3. $I_t = \alpha_t x + \beta_t \mathcal F(x)$
	4. $\mathcal L(\hat b) = \mathbb E_t \mathbb E_{I_t}[(\hat b_t(I_t)- \dot I_t )^2]$ 
## SI Algorithm (Lifted)
1. For $\_$ in $1,2,..., N_{epoch}$:
	1. Sample $x \sim \pi$ (clean data)
	2. $z \sim \mathcal N(0, \mathbb I)$ and $t \sim \text{Uniform}([0,1])$
	3. $I_t = \alpha_t z + \beta_t \mathcal F(x)$
	4. $\mathcal L(\hat b) = \mathbb E_{t} \mathbb E_{I_t}[(\hat b_t(I_t | x) - \dot I_t)^2]$ 
## SCSI Algorithm
1. For _ in  $1, 2,.., K$:
	1. Sample $y \sim \nu$ (corrupted data)
	2. $x = \Phi_{\Theta^{(k-1)}}(y)$ 
	3. $\hat y = \mathcal F(x)$
	4. $t \sim \text{Uniform}([0,1])$
	5. $I_t^{(k)} = \alpha_t x + \beta_t \hat y$
	6. $\mathcal L (\hat b) = \mathbb E_{t, I_t} [(\hat b_t(I_t^{(k)}) - \dot I_t^{(k)})^2]$ 
## SCSI Algorithm (Lifted)
1. For $k$ in $1,2,..., K$
	1. Sample $y \sim \mu$ (corrupted data data)
	2. Sample $a \sim \mathcal  N(0, \mathbb I)$, then flow $x = \Phi_{\Theta^{(k-1)}}(a | y)$
	3. $\hat y = \mathcal F(x)$ 
	4. $z \sim \mathcal N(0, \mathbb I)$ and $t \sim \text{Uniform}([0,1])$
	5. $I_t = \alpha_t x + \beta_t z$
	6. $\mathcal L(\hat b) = \mathbb E_t \, \mathbb E_{I_t} [(\hat b_t(I_t | \hat y) - \dot I_t)^2]$ 

In my case, let $\pi = \text{MNIST Images}$ and $(\mathcal F \phi)(u) = \phi(u - u_0) + w(u)$ (where $u_0 \sim G$ Haar measure over translation group, and $w(u) \sim^{iid} \mathcal N(0, \sigma^2 \mathbb I)$) be the MRA channel. The observed data $\mu = \mathcal F_{\#} \pi$.


