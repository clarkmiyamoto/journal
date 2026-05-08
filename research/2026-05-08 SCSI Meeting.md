#scsi #meeting 

```table-of-contents
```
___
# Non Research Questions
- Priority allocation on HPC
- Funding over summer

____
# Initialization of $\pi^{(0)}$
## Extra Corruption Initialization

Let $F$ be the MRA forward channel, $\nu = \mathcal F_\# \pi$ (corrupted distribution)

**Extra Corrupted Initialization**
1. Train on supervised $(\nu, \mathcal F_\# \nu)$ for 2,000 epochs. Fully solve supervised problem
![[2026-05-08 epoch2000-3.png]]
![[2026-05-08 epoch2000-1.png]]
> Weird memorization after training too long.
2. Train on supervised $(\nu, \mathcal F_\# \nu)$ for 1,000 epochs. Partially solve supervised problem
![[2026-05-08 epoch1000-4.png]]
![[2026-05-08 epoch1000-2.png]]
![[2026-05-08 epoch1000-3.png]]
![[2026-05-08 epoch1000-1.png]]
3. Train on supervised $(\nu, \mathcal F_\# \nu)$ for 1 epoch. 
![[2026-05-08 epoch1-3.png]]
![[2026-05-08 epoch1-2.png]]
![[2026-05-08 epoch1.png]]
The size of the blocks are 4 pixels, and my DiT uses a patchify size of 4 pixels. If we take patchify = 1 this should go away, but it's difficult to pla

We need a better way to initialize, in the real problem $\mathcal F : \mathcal X \to \mathcal Y$. So you can't just run $\mathcal F_\# \nu$...

## Idea: Annealing over Corruptions
User specifies
- A family of corruption channels $(\mathcal F_\ell)_{\ell \in [0,1]}$ s.t. $\mathcal F_{\ell =0}$ is easy to learn and $\mathcal F_{\ell = 1} = \mathcal F$ is the original corruption. 
	- *Example (Linear interpolation):* $\mathcal F_\ell(x) = (1-\ell) \mathcal F_{\textsf{simple}}(x) + \ell \, \mathcal F_{\textsf{MRA}}(x)$, where $\mathcal F_{\textsf{id}}(x) = x$.
- Neural network architecture $b_t^\theta(x | y , \ell)$ 
$$
\begin{align}
I_t^{(k+1)} | \mathcal F_{\ell(k)}(X^{(k)}) = \alpha_t Z + \beta_t X^{(k)}
\end{align}
$$
where $\ell : [0, K] \to [0,1]$. 
**Algorithm** (new parts are in blue)
```pseudo
\begin{algorithm}
\begin{algorithmic}
\State \textbf{Warmup Phase:}
\For{$\textcolor{blue}{\ell \in [0,1]}$} 
\Comment{Annealing over forward channel}
	\For{$k \in [0, K_\textcolor{blue}{\ell}]$} \Comment{EM Steps}
		\State Sample $z, z' \sim \mathcal N(0, \mathbb I)$
		\State Sample $y \sim \mathcal \nu$ (corrupted)
		\State $\hat x \leftarrow \Phi^{(k)}(z' | y)$
		\State $\hat y \leftarrow \mathcal F_{\textcolor{blue}{\ell}}(\hat x)$
		\State $I_t = \alpha_t z + \beta_t \hat x$
		\State $\mathcal L(\theta) = | b_t^\theta(I_t| \hat y, \textcolor{blue}{\ell}) - \dot I_t |^2$
    \EndFor
\EndFor
\State \textbf{Regular Phase:} Use trained neural network for regular SCSI algorithm.
\end{algorithmic}
\end{algorithm}
```
The idea is you start at an easy inverse problem, and make it progressively harder. If you believe the neural network generalizes a little outside of a given $\ell$, then it serves as a good warm start for the next.

**Implementation**
I use a DiT and treat $\ell$ as class labels.
![[Pasted image 20260507204351.png]]
![[Pasted image 20260507205025.png]]
![[Pasted image 20260507204502.png]]
- Top: $\ell \in [0, 0.1, ..., 0.9, 1.0]$ with 10 EM steps per level, 5 epochs per EM step.
- Middle $\ell \in [0, 0.1, ..., 0.9, 1.0]$, with 20 EM steps per level, with 5 epochs per EM step.
- Bottom $\ell \in [0, 0.1, ..., 0.9, 1.0]$, with 5 EM steps per level, with 20 epochs per level.

The EM in the title refers to EM steps during the *Regular Phase*. This is their results for the same amount of wall clock time (as the pretraining will take different amounts of time).

## Idea: Consistency for Annealing 
**Idea:** Sample $\ell \sim \text{Unif}[0,1]$ (or perhaps a fancier distribution) and use that in the loss function.

**Question:** Are there constraints on the velocity field $b_t(x | y, \ell)$ which enforce consistency over $\ell$ space?

**Example (Martingale Property):**
Let the family $(\mathcal F_\ell)_\ell$ form a Markov chain as $\ell$ increases. In this notation, $\ell_1 < \ell_2$, meaning $\ell_2$ is the "harder" inverse problem. 
$$
\begin{align} b_t^*(x | Y_{\ell_2}, \ell_2) &= \mathbb E \big[ \dot I_t \mid I_t = x, Y_{\ell_2} \big] \\ 
&= \mathbb E \Big[ \mathbb E \big[ \dot I_t \mid I_t = x, Y_{\ell_1}, Y_{\ell_2} \big] \;\Big| \; I_t = x, Y_{\ell_2} \Big]\\
&= \mathbb E \Big[ \mathbb E \big[ \dot I_t \mid I_t = x, Y_{\ell_1} \big] \Big| I_t = x, Y_{\ell_2} \Big] \\
& = \mathbb E \Big[ b_t(x | Y_{\ell_1}, \ell_1) \;\Big| \; I_t = x, Y_{\ell_2} \Big]\end{align}
$$
Since $I_t = \alpha_t Z + \beta_t X$, and Markov assumptions $X \to Y_{\ell_1} \to Y_{\ell_2}$, we can drop the $Y_{\ell_2}$ from the inner expectation. This yields to a consistency-esq loss
$$
\begin{align}
\mathcal L(\theta) = \mathbb E_{(\ell_1, \ell_2)} \mathbb E_{t, X, Z, Y_{\ell_1}, Y_{\ell_2}} \left[ \left | b_t^\theta(I_t | Y_{\ell_2}, \ell_2)  - \texttt{stopgrad}(b_t^\theta(I_t | Y_{\ell_1}, \ell_1))\right | \right]
\end{align}
$$
To make this amicable to the inverse problem, replace $X \mapsto X^{(k)}$ and $Y_\ell \mapsto Y^{(k)}_\ell = \mathcal F_\ell(X^{(k)})$.


# Coupling between $Z,Z'$
**Question:** Why is convergence rate for Gaussian AWGN slower? I know it's EM but it begs the question *why* should $I_t = \alpha_t \mathcal F(X) + \beta_t X$ quicker?

**Hypothesis:** The coupling between $I_{t=0}$ and $I_{t=1}$ is stronger. In the lifted setting, when we use iid noise $Z,Z'$ in the interpolant $I_t = \alpha_t Z + \beta_t X$ and flow $\Phi(Z' | Y)$ you have implicitly made the coupling weaker.

**Proposal:** Partially sample new noise. $I_t = \alpha_t Z + \beta_t X$ and $\Phi(Z' | Y)$. Use $\gamma$ percent $Z$ for $Z'$ in the batch. I'll denote this as the `cf` (coupling fraction).

![[2026-05-08 stronger coupling.png]]
![[2026-05-08 no coupling.png]]
Top photo: `cf = 10%` (meaning $Z=Z'$ 10% of the time). Bottom photo: `cf = 0%` (meaning $Z \neq Z'$). Visually you can see using the coupling helps convergence.

![[2026-05-08 coupling ablation convergence.png]]
![[2026-05-08 coupling ablation histogram.png]]
Over various coupling you can see the higher `cf` make the histogram more tight (meaning quicker convergence). But you also see it leads to worse reconstruction. 

If you make the corruption fraction `cf` higher than 0.5 you get bad looking samples

> Any other hypothesis? I think there's something to be said the coupling rather than "it's just EM".

