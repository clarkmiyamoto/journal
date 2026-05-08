#paper #scsi 

Paper (Homotopy): https://arxiv.org/pdf/2509.14394v1
Paper (Two Parameter): https://cims.nyu.edu/~pehersto/preprints/tpf.pdf
### Setup
Consider inverse problem of reconstructing an image $x \in \mathbb R^n$ from a corrupted image $y = H x + e \in \mathbb R^m$ (s.t. $m \leq n$).  

If $H$ is very lossy, this make make reconstruction more difficult. A way to solve this is to construct a good initial guess for $x$ and then gradient descent from there. However this is not always possible

Instead consider embedding the corruption channel $H$ inside a family of corruptions $H_k$, and then you update $H_{k+1}$ over various EM steps.

**In our notation** This would amount to embedding $\mathcal F$ (corruption channel) inside a family of corruption channels $\mathcal F_k$, and updating it to become closer to the original channel as we make EM steps.
___
The thing is, why don't we just learn this in a more continuous manner? I'm hoping that that this "two parameter flows" paper will serve as a inspiration for this.

Let $t$ be the physical time, and $s$ be the stochastic interpolant's time.
1. Learn the transport of reference (base) distribution to each time marginal $\rho(t)$
2. Use the induced structure to obtain dynamics in physics time direction, to evolve $\rho(t)$ over $t$. 
The observation is once the transport form base to $\rho(t)$ (for all $t$) is fixed, there is a unique consistent way to connect these transport across physics time $t$.


Consider $\rho(t)$ for $t \in [0,T]$ and a base distribution $\nu$. Consider the velocity field $v : \mathbb R^d \times [0,T] \times [0,1] \to \mathbb R^d$ s.t.
$$
\begin{align}
\frac{d}{ds} x_{t,s} = v(x_{t,s}, t, s), \qquad x_{t,s=0} = a \sim \nu
\end{align}
$$
generates samples $x_{t,s}|_{s=1} \sim \rho(t)$ at final sampling time $s=1$ (for all $t \in [0,T]$).


Ok reading more about this is not super useful to this problem.
____
# Inverse Problems via Non Equilibrium Sampling?
Let $\mathcal F$ be the corruption channel you're trying to invert.  In the original SCSI algorithm, we construct a family of interpolants $(I_t^{(k)})_k$
$$
\begin{align}
I_t^{(k+1)} | \mathcal F(X^{(k)}) = \alpha_t Z + \beta_t X^{(k)}
\end{align}
$$
where $X^{(k)}$ is the result of the flow on the previous interpolant$$
\begin{align}
X^{(k)} = \Phi^{(k)}(Z' | Y) = Z' + \int_0^1 b_t^{(k)}(X_t | Y)\, dt
\end{align}
$$and the drift is learned from a previous interpolant
$$
\begin{align}
b_t^{(k+1)}(x|y) = \mathbb E[\dot I_t^{(k+1)} | I_t^{(k+1)} = x | \mathcal F(X^{(k)}) = y]
\end{align}
$$
After iterating in $k$, you hope that the fixed point $X^* = X$ (original distribution).

We find that the initial interpolant $I_t^{(0)}$ must be ansantz/guessed/pretrained carefully otherwise the algorithm doesn't converge. An idea is to use a family of corruption channels $(\mathcal F_\ell)_{\ell \in [0,1]}$ s.t. $\mathcal F_{\ell =0}$ is easy to learn and $\mathcal F_{\ell = 1} = \mathcal F$ is the original corruption.
$$
\begin{align}
I_t^{(k+1)} | \mathcal F_{\ell(k)}(X^{(k)}) = \alpha_t Z + \beta_t X^{(k)}
\end{align}
$$
where $\ell : [0, K] \to [0,1]$. 

A couple questions come up
- For every distribution, does there exists an $\mathcal F$ where the SCSI algorithm has a fixed point. In particular $\mathcal K$ (integral operator corresponding to $\mathcal F$ must be injective).
	- **Conjecture:** The identity map $\mathcal F_{\textsf{id}}(x) = x$ has this property.
		- However this is a map $\mathcal F : \mathcal X \to \mathcal X$
	- Sharpening the question. For every distribution does there exists $\mathcal F : \mathcal X \to \mathcal Y$ (where $\text{dim}(\mathcal  Y) < \text{dim}(\mathcal X)$) which has a unique fixed point. S.t. the fixed point is not just collapse to delta function? If so give an example of what it is.
		- **Conjecture:** $\mathcal F$ must be stochastic.
- If you know a lot about $\mathcal F$, can you accelerate convergence rate using a properly chosen $(\mathcal F_\ell)_\ell$?
- How do you choose the proper parameterization $\ell(k)$?
	- You can think of this as unadjusted annealed Langevin $dX_t = - \nabla \log \pi_t(X_t) dt + \sqrt 2 dW_t$. That is ULA takes infinite steps to converge, so really you need to run this SDE at fixed $t$ for a long time, and then make an update on $t$.
		- In a similar way, the SCSI algorithm needs to runs a lot of EM steps for a fixed $k$, and then make an update on $k$.
	- In the sampling problem (oracle access to $\pi(\cdot)$) there are fixes. E.g. NETS algorithm. In the inverse problem (oracle access to $\mathcal F$ and samples of $\mathcal F_\# \pi$), can anything be done?

### First idea
Ok here's my first thoughts for an algorithm which solves this.

**Given** 
- Corruption function $\mathcal F$
- Samples of corrupted data $\{y_i\}$
**User specifies**
- Family of corruption functions $(\mathcal F_\ell)_{\ell \in [0,1]}$ where $\mathcal F_{\ell = 1} = \mathcal F$
  *Example (Linear interpolation):* $\mathcal F_\ell(x) = (1-\ell) \mathcal F_{\textsf{id}}(x) + \ell \, \mathcal F_{\textsf{MRA}}(x)$ 
  *Example ($\alpha_t, \beta_t$ interpolation):* $\mathcal F_\ell(x) = \tilde \alpha_\ell \mathcal F_{\textsf{id}}(x) + \tilde \beta_\ell \mathcal F_{\textsf{MRA}}(x)$

- Neural network architecture $b_t^\theta(x | y , \ell)$ 
**Algorithm** (new parts are in blue)
```pseudo
\begin{algorithm}
\begin{algorithmic}
\For{$\textcolor{blue}{\ell \in [0,1]}$}
	\For{$k \in [0, K_\textcolor{blue}{\ell}]$}
		\State Sample $z, z' \sim \mathcal N(0, \mathbb I)$
		\State Sample $y \sim \nu$ (corrupted)
		\State $\hat x \leftarrow \Phi^{(k)}(z' | y)$
		\State $\hat y \leftarrow \mathcal F_{\textcolor{blue}{\ell}}(\hat x)$
		\State $I_t = \alpha_t z + \beta_t \hat x$
		\State $\mathcal L(\theta) = | b_t^\theta(I_t| \hat y, \textcolor{blue}{\ell}) - \dot I_t |^2$
    \EndFor
\EndFor
\end{algorithmic}
\end{algorithm}
```
The idea is you start at an easy inverse problem, and make it progressively harder. If you believe the neural network generalizes a little outside of a given $\ell$, then it serves as a good warm start for the next.

**Simple modifications to test**
- Sample $\ell \in \text{Unif}[0,1]$ and use that in the loss function.
	- Or if you if you swap the $\ell$ and $k$ `For loop`. Meaning you do an EM step at a set $\ell$, move to move to the next $\ell$. This reminds me of a Picard-type iteration (see https://arxiv.org/pdf/2508.18413); however this is formulated as $x_t = f_t(x_{t-1}$ ), so in ours, you need to find a $f_t$ s.t. $f_1 \circ f_{1-\Delta \ell} \circ ... \circ f_{\Delta \ell} \circ f_0 = \mathcal F$. 
		- One might say, what if I adjust the "corruption level" of $\mathcal F$, and iteratively apply that. Often times that is not true. For example, JPEG corruption.
- Remove $\ell$ in the neural network

```pseudo
\begin{algorithm}
\begin{algorithmic}
\For{$k \in [0, K]$}
	\State Sample $\ell \sim \text{Unif}[0,1]$
 	\State Sample $z, z' \sim \mathcal N(0, \mathbb I)$
	\State Sample $y \sim \nu$ (corrupted)
	\State $\hat x \leftarrow \Phi^{(k)}(z' | y)$
	\State $\hat y \leftarrow \mathcal F_{\textcolor{blue}{\ell}}(\hat x)$
	\State $I_t = \alpha_t z + \beta_t \hat x$
	\State $\mathcal L(\theta) = | b_t^\theta(I_t| \hat y, \textcolor{blue}{\ell}) - \dot I_t |^2$
\EndFor
\end{algorithmic}
\end{algorithm}
```

