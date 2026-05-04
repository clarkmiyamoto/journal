#note #probability 

# Loewner Ordering
**Definition** (PSD): Let $A \in \mathbb R^{d\times d}$ be a symmetric matrix, is it positive semidefinite (PSD) if $\forall x \in \mathbb R^d$:
$$
\begin{align}
x^T A x \geq 0
\end{align}
$$
It is called just positive definite if the inequality is $>$.

**Fact:** If $A$ is PSD, then all eigenvalues are $\geq 0$.

*Proof:* Since $A$ is a symmetric matrix, it has a eigenbasis $A v_i = \lambda_i v_i$ s.t. any vector can be written as a linear combination of $v_i$. Additionally, the eigenvalues $\lambda_i \geq 0$.

$$
\begin{align}
x^T A x = \sum_{n,m} \alpha_n \alpha_m \,  v_n^T A  v_m = \sum_{n,m} \alpha_n \alpha_m \lambda_m \, \underbrace{v_n^T v_m}_{\delta_{nm}} =  \sum_n \alpha_n^2  \lambda_n
\end{align}
$$
where $\alpha_n = \langle v_n, x\rangle$. Notice that $\alpha_n^2 \geq 0$. Since our choice of $x$ is arbitrary, let $x = v_p$ (the $p$'th eigenvector), then $\alpha_n = \delta_{np}$ 
$$
\begin{align}
x^T A x = \lambda_p
\end{align}
$$
Since $x^T A x \geq 0$ for all $x$, then $\lambda_p \geq 0$ for all $p$. 
$\blacksquare$

**Definition** (Loewner Ordering) Let $A,B$ be a square matrices. $A \succeq 0$ denotes $A$ is PSD. The notation $A \succeq B$, means $A - B \succeq 0$.

Some properties that are easy to prove
- **Transitivity:** If $A \succeq B$ and $B \succeq C$, then $A \succeq C$.
- **Addition:** Prove if $A \succeq B$ and $C \succeq D$, then $A + C \succeq B+ D$
- **Non-negative Scaling:** Let $\alpha \geq 0$. If $A \succeq B$, then $\alpha A \succeq \alpha B$. 

**Facts:** Let $A$ be PSD w/ eigenvalues $\lambda_1 \leq ... \leq \lambda_d$. 
1. $\lambda_1 \mathbb I \preceq A \preceq \lambda_d \mathbb I$
   *Proof:* Consider $x^T(A - \lambda_d \mathbb I) x$. Set $x = \sum_i \alpha_i v_i$, this gets $\sum_{n} \alpha_n^2 (\lambda_n - \lambda_d) \leq 0$. Do the same for $x^T (A - \lambda_1 \mathbb I) x$.
2. Let $M$ be defined s.t. the matrix multiplication work

# Strongly Log Concave
**Definition** ($c$-strongly log-concave)
Let $p \propto e^{-V(x)}$ be a probability distribution. It is said to be $c$-strongly log concave, if $V(x)$ is $c$-strongly log convex; that is
$$
\begin{align}
\nabla^2 V \succeq c \mathbb I
\end{align}
$$
where $\nabla^2$ denotes the Hessian.

*Example:* What is the concave parameter $c$ for a Gaussian $\gamma_{\mu, \Sigma}$?
$$
\begin{align}
\gamma_{\mu, \Sigma} = \frac{1}{Z} \exp \left(- \frac 1 2 (x-\mu)\Sigma^{-1} (x-\mu) \right)
\end{align}
$$
Therefore $V(x) = \frac 1 2 (x - \mu) \Sigma^{-1} (x-\mu)$, where $\Sigma \succeq 0$. The Hessian becomes
$$
\begin{align}
\nabla^2V = \Sigma^{-1}
\end{align}
$$
Let $\Sigma^{-1}$ have eignevalues $\lambda_1 \leq .... \leq \lambda_d$. Recall the fact $\lambda_1 \mathbb I \succ \Sigma^{-1} \succ \lambda_d \mathbb I$. We can read this off and find $c = \lambda_d$.  B



## Heat Equations affects on Strongly Log Concave Distributions
Consider the heat equation
$$
\begin{align}
\partial_t \pi_t = \frac 1 2 \Delta \pi_t, \qquad \pi_{t=0} = \pi
\end{align}
$$
If you let $\text{Law}(X_t) = \pi_t$, you can rewrite this into an addition of random variables.
$$
\begin{align}
X_t = X + \sqrt t \, Z
\end{align}
$$
where $X \sim \pi$ and $Z \sim \mathcal N(0, \mathbb I)$.

**Problem:** If $\pi$ is $c$-strongly log-concave, is $\pi_t$ strongly log-concave?  What is the constant $c_t$ associated w/ the $\pi_t$'s log-concavity?