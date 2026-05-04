#math #probability 
https://www.stat.cmu.edu/~larry/=stat325.01/chapter5.pdf

Something I'm embarrassed to admit, is I don't know all the different ways to talk about convergence of random variables. Here's to learning.

**Definition** (Convergence in quadratic mean) $X_n \to^{qm} X$
Consider a sequence of random variables $X_1, X_2 ...$ which have finite 2nd moments. $X_n$ converges to $X$ in *quadratic mean* if
$$
\lim_{n \to \infty} \mathbb E[(X_n - X)^2] = 0
$$
**Definition** (Convergence in probability) $X_n \to^p X$
$X_n$ converges to $X$ in probability if, for all $\epsilon > 0$
$$
\lim_{n \to \infty} \, \mathbb P(|X_n - X| > \epsilon) = 0
$$
**Definition** (Convergence in distribution) $X_n \to^d X$
Let $F_n$ be the cumulative function of $X_n$. $X_n$ converges to $X$ in distribution if, 
$$
\lim_{n \to \infty} F_n(t) = F(t)
$$
at continuity points $t$.
___
**Example**
Let $X_n = \mathcal N(0, 1/n)$, does it converge to $0$ as $n \to \infty$? If so, in what sense?

*Quadratic mean*
$$
\begin{align}
\lim_{n \to \infty}\mathbb E[(X_n - 0)^2] & = \lim_{n \to \infty}\mathbb E[Z^2/n] = \lim_{n\to\infty} 1/n = 0
\end{align}
$$
where $Z \sim \mathcal N(0,1)$.

*Probability*
$$
\mathbb P(|X_n| > \epsilon) = \mathbb P(|Z|/\sqrt{n} >  \epsilon)
$$
As $n \to \infty$, $\mathbb P(|Z| / \sqrt{n} > \epsilon) \to 0$.

*Distribution*
The $0$ as a random variable, is a delta function at $0$.
$$
\begin{align}
F_n(t) & = \mathbb P(X_n < t)\\
& = \mathbb P(Z/\sqrt{n} < t)\\
& = \mathbb P(Z < \sqrt{n} \, t)
\end{align}
$$
If $t >0$ then $F_n(t) \to 1$, and if $t < 0$ then $F_n(t) \to 0$. 

So it converges in all senses.
___
**Theorem**
1. If $X_n \to^{qm} X$, then $X_n \to^p X$
2. If $X_n \to^p X$, then $X_n \to^d X$
3. If $X_n \to^d X$ and $\mathbb P(X=c) = 1$ for some real number $c$, then $X_n \to^p X$.

*Proof*
1. Recall Chebyshev. For any $\alpha > 0$ $$
	\mathbb P(|X - \mu| \geq \alpha) \leq \frac{\text{Var}(X)}{\alpha^2}
	$$ Consider a random variable $Y = X_n - X$, then by Chebyshev $$ \mathbb P(|X_n - X| \geq \alpha) \leq \frac{\mathbb E[(X_n - X)^2]}{\alpha^2}
	$$ Since we've assumed $X_n \to^{qm} X$, by definition, $\mathbb E[(X_n - X)^2] \to 0$. Therefore $$\mathbb P(|X_n - X| \geq \alpha) \to 0$$ Since Cheybyshev holds for any $\alpha > 0$, we suffice the definition for $X_n \to^p X$.

2. 