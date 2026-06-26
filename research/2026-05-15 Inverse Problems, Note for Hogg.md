
Hi Hogg, here's a note on a general method for solving inverse problems. Curious to hear what kind of problems in your world could be solved here. Please give some toy problems and some novel ones!
# Setup
**What you have access to**
- Black-box access to corruption function (stochastic or deterministic) $\mathcal F : \mathcal  X \to \mathcal Y$
- Dataset of corrupted data $\mathcal D = \left\{ y_i\right\}_i$, s.t. $y_i \in \mathcal Y$.

**Your objective**
- Make a generative model on the clean data $x_i \in \mathcal X$ s.t. $\mathcal F(X) \overset{d}{=} Y$
	- The constructed generative model can be conditional of a single observation $p(x | y)$, multiple observations $p(x | y_1, ..., y_n)$ or unconditional $p(x)$.

**Assumptions (which we can attempt to break free from)**
- No model misspecification. That is the corruption function actually generates the $y$'s from the $x$'s
- Corruption function is "quick". In this algorithm, it is applied to each data point during the gradient descent of a neural network. And the algorithm seems to take about 20k iterations to converge.

**Optional things which are helpful to the algorithm**
- If you have some examples of $x$, to warm-start the algorithm.
	- Examples need not be true $x$'s, e.g. they can be simulations of a real life phenomena. But the algorithm
## Examples

| Corruption funciton $\mathcal F$            | Corrupted Data $y$   | Clean Data $x$             |
| ------------------------------------------- | -------------------- | -------------------------- |
| JPEG Algorithm                              | JPEG'd images        | Regular images             |
| Radon Transform                             | 2D CT Scan           | 3D reconstruction of brain |
| Lens distortion, <br>instrument white noise | Image from telescope | Clean image (lol)          |

## Situations where this method is interesting
1. Single observation, but many data points. In more classical methods, you need to have access to multiple measurements ${y_1, ..., y_n}$ of the same clean object $x$, but this method can work when you only have one observation of many different objects.
<img src="Screenshot 2026-05-15 at 10.52.27 AM.png" width="50%">
2. It has the improvement property of an EM algorithm. If you run algorithm longer, you're guaranteed to get a closer generation of $X$ or stay the same.
# Technical Explanation of How the Algorithm Works
For simplicity assume $\mathcal F : \mathcal X \to \mathcal X$

Consider a stochastic interpolant
$$
\begin{align}
I_t = (1-t) \mathcal F(X) + t \, X
\end{align}
$$
where $X$ is a random variable drawn from the clean data, therefore $I_t$ is also a random variable.

Similar to score-based diffusion, we can write an ODE/SDE which has the same probability distribution as $I_t$. 
$$
\begin{align}
dX_t = b_t(X_t) \, dt, \quad X_{t=0} \sim \mathcal F(x)
\end{align}
$$
If $b_t(x) = \mathbb E[\dot I_t | I_t = x] =  \mathbb E[X - \mathcal F(X) | I_t = x]$, then $X_{t} \overset{d}{=} I_t$. We can write a conditional expectation as a minimizer of a MSE, from here we'll use a neural network to fit $b_t(x)$.

However this assumes you have access to $X$, in our case, we only have access to $Y$'s. So an idea is to use the result of the ODE as a guess for $X$ (since this is an iterative scheme, I'll denote the $k$'th guess as $X^{(k)}$). As your guesses get better and better, we hope $\lim_{k \to \infty} \mathcal F(X^{(k)}) = Y$. Another way to say this is we're checking for self consistency.

Putting this back into the stochastic interpolants framework
1. Construct your guess for the $k+1$ interpolant using the result of the previous interpolant $$I_t^{(k+1)} = (1-t) \mathcal F(X^{(k)}) + t X^{(k)}$$
2. Use a neural network to train the drift $b_t^{(k+1)}$ by minimizing the MSE associated $I_t^{(k+1)}$
3. Generate new samples $X^{(k+1)}$ by integrating $dX_t^{(k+1)} = b_t^{(k+1)}(X^{(k+1)}) dt$
Iterate over $k$ until $X^{(k)}$ converges

**Summarize Procedure**
**Require:** Initialization for drift $b_t^{(1)}$

For $k \in [1,..., K]$ 
1. $y \sim$ Corrupted Data
2. Integrate $dX_t = \hat b_t^{(k)} (X_t) dt$ with boundary condition $X_{t=0} = y$. Denote $X_{t=1} = \hat x$
3. Construct interpolant $I_t = (1-t) \mathcal F(\hat x) + t \hat x$
4. Train neural network $$\mathcal L(\hat b_t^{(k+1)}) = \| \hat b_t^{(k+1)}(I_t) - \dot I_t \|^2 $$

