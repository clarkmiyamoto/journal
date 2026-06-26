#scsi #meeting 

```table-of-contents
```

___
# Todo's
1. Find easy dataset for CryoEM
2. Find diffusion model architecture which acts on 3D but conditions on 2D.

___
# Non Research Questions
- Priority allocation on HPC
> Joan: Put me in CILVR slack w/ person
- Funding over summer
> Joan: Reaching out!

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
The size of the blocks are 4 pixels, and my DiT uses a patchify size of 4 pixels. If we take patchify = 1 this could go away, but this gets computational difficult in higher dimensions.

We need a better way to initialize, in the real problem $\mathcal F : \mathcal X \to \mathcal Y$. So you can't just run $\mathcal F_\# \nu$...

> Joan: Increase the noise level. 
> Eric: DiT is very optimized for ImageNet

> How do we continue?
> - The MRA is a toy-problem for CryoEM. 
> 	1. Do CryoEM channel
> 		- What is the architecture we're using?
> 		- How do you construct the supervised warm start?
> 	2. Keep thinking about running the channel on discrete space. Text corruptions.
> 		- Montanari: continuous diffusion K-SAT problem
> 		- Follmer process 
## Idea: Annealing over Corruptions
User specifies
- A family of corruption channels $(\mathcal F_\ell)_{\ell \in [0,1]}$ s.t. $\mathcal F_{\ell =0}$ is easy to learn and $\mathcal F_{\ell = 1} = \mathcal F$ is the original corruption. 
	- Examples: 
		- (Linear interpolation) $\mathcal F_\ell(x) = (1-\ell) \mathcal F_{\textsf{simple}}(x) + \ell \, \mathcal F_{\textsf{MRA}}(x)$
		- (Markovian) Choose $\mathcal F_\ell$ s.t. it's Markov chain as $\ell$ increases
- Neural network architecture $b_t^\theta(x | y , \ell)$ 

**Interpolant**
$$
\begin{align}
I_t^{(k+1)} | \mathcal F_{\ell}(X^{(k)}) = \alpha_t Z + \beta_t X^{(k)}
\end{align}
$$
where $X^{(k)} = \Phi^{(k)}(Z' | Y , \ell=1)$ (result of flow induced by interpolant)

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
		\State $\hat x \leftarrow \Phi^{(k)}(z' | y, \textcolor{blue}{\ell = 1})$
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
- Annealing scheduler: $\mathcal F_{\ell}(x) = (1-\ell) x + \ell \mathcal F(x)$ 
- Model: DiT and treat $\ell$ as class labels.
![[2026-05-08 annealing 4.png]]
![[2026-05-08 annealing 1.png]]
![[2026-05-08 annealing 2.png]]
- Top: $\ell \in [0, 0.1, ..., 0.9, 1.0]$ with 10 EM steps per level, 5 epochs per EM step.
- Middle $\ell \in [0, 0.1, ..., 0.9, 1.0]$, with 20 EM steps per level, with 5 epochs per EM step.
- Bottom $\ell \in [0, 0.1, ..., 0.9, 1.0]$, with 5 EM steps per level, with 20 epochs per level.
- The EM in the title refers to EM steps during the *Regular Phase*. This is their results for the same amount of wall clock time (as the pretraining will take different amounts of time).

Something problems with this scheme
1. User needs to play around with $\ell$'s scheduler. 
	- EM algorithm takes time to converge, how do you know when to update $\ell$? Similar problem with using annealed Langevin.
2. You're not taking advantage of making sure $\mathcal F_{\ell}(X^{(k)})$ and $\mathcal F_{\ell'}(X^{(k)})$ are consistent
3. What is a simple $\mathcal F_{\ell =0} : \mathcal X \to \mathcal Y$ (when $\text{dim}(\mathcal Y) < \text{dim}(\mathcal X)$)? Or to be specific to our problem, what is a good choice of $\mathcal F_\ell$ in the CryoEM setting $\mathcal F(x) = P G x + \sigma W$?
## Idea: Consistency-esq Annealing 
**Idea:** Sample $\ell \sim \text{Unif}[0,1]$ (or perhaps a fancier distribution) and use that in the loss function.

**Question:** Are there constraints on the velocity field $b_t(x | y, \ell)$ which enforce consistency over $\ell$ space?

**(Martingale Property):**
Let the family $(\mathcal F_\ell)_\ell$ form a Markov chain as $\ell$ increases. In this notation, $\ell_1 < \ell_2$, meaning $\ell_2$ is the "harder" inverse problem. 
$$
\begin{align} b_t(x | Y_{\ell_2}, \ell_2) &= \mathbb E \big[ \dot I_t \mid I_t = x, Y_{\ell_2} \big] \\ 
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
To make this amicable to the unsupervised setting, replace $X \mapsto X^{(k)}$ (result of flow) and $Y_\ell \mapsto Y^{(k)}_\ell = \mathcal F_\ell(X^{(k)})$.

> - Sanity check this with Gaussian setting. See if this can accelerate for Gaussian
> - In very difficult problems, it seems that a good warm-start won't be able to be beat.
> - Get a more difficult problem working before we start playing around w/ the algorithm

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

There's a tension, we want $Z = Z'$ but this leads to the bad fixed point $(b_t= 0)$.

> Eric: You can do the corruption on the $Z$... Not sure what he means by this...


```pseudo

```