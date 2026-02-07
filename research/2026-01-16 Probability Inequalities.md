#math #probability 

In [[2026-01-16 Optimally Tuned ChEES]] I thought about bounding the variance of the empirical estimation of the ChEES criterion. 

So in hopes of getting there, here is a review on probability inequalities (that I should know).


# Probability Inequalities

**Jensen's Inequality** Consider a convex function $f$, then

$$
f(\mathbb E[X]) \leq \mathbb E[f(X)]
$$

*Proof:*  Recall the definition of a convex function is for $\lambda_1,\lambda_2 \in [0,1]$ s.t $\lambda_1  + \lambda_2 = 1$, a function is convex if $f(\lambda_1 a + \lambda_2 b) \leq \lambda f(a) + \lambda_2 f(b)$ for all $a,b$ in the domain of the function. Notice that these coefficients have the same properties the probabilities assigned to outcomes of a random variable (that is they're between 0-1, and they sum to 1). 

You can show a convex function must satisfy $f(\sum_i \lambda_i a_i) \leq \sum_i \lambda_i  f(a_i)$ s.t. $\sum_i \lambda_i = 1$ for all $a_i$ in the domain of the function. If we have a random variable $A$ w/ probability $\lambda_i$ associated with event $a_i$, then $\sum_i \lambda_i a_i = \mathbb E[A]$. 

$$\tag*{$\Box$}$$

*Example (Norms):* Let $X$ be a random variable, then
- $\| \mathbb E X \| \leq \mathbb E\|X\|$
- Consider $0 \leq a \leq b < \infty$, then $\|X\|_{L^a} \leq \| X \|_{L^b}$ 

*Proof:* Recall the definition of a norm $\|X\|_p \equiv ( \mathbb E[|X|^p])^{1/p}$.

Notice that $f(x) = x^r$ for $r \geq 1$ is convex. Let $a < b$, which implies $b/a > 1$.  Consider the random variable $Y = |X|^c$, then
$$
f(\mathbb E[Y]) \leq \mathbb E[f(Y)] \implies (\mathbb E[|X|^c])^{b/a} \leq  \mathbb E[(|X|)^{b\, c / a}].
$$
Now you can choose $c = a$, then $b$'th rooting both sides yields our inequality! 

$$\tag*{$\Box$}$$
**Minkowski Inequality** Let $X,Y$ be a random variables, where $p \in [0, \infty)$ 
$$
\|X + Y\|_p  \leq \|X \|_p + \|Y\|_p
$$
*Proof:* By definition
$$
\begin{align}
\|X + Y\|_p & = (\mathbb E[| X + Y|^p])^{1/p } \\
& \leq (\mathbb E[|X|^p] + \mathbb  E[|Y|^p])^{1/p} & \text{Triangle inequality}\\
& \leq \| X\|_p + \|Y\|_p & 
\end{align}
$$
Recall triangle inequality $|a + b| \leq |a| + |b|$, this implies $|a+b|^p \leq (|a| + |b|)^p$. Now allI need to use is figure out

$$
(a + b)^p \leq a^p + b^p
$$
where $a ,b \geq 0$

$$
(a + b)^n = \sum_{k=1}^n C(k, n) a^k b^{n-k} = a^n + b^n + \text{positive constants}
$$
This implies $(a + b)^n > a^n + b^n$ 



# Appendix, Regular Inequalities

### Scalar Inequalities
Power
- For $p > 1$
	- $x^n \leq x$ for $x \in [0,1]$
	- $x^n \geq x$ for $x \geq 1$
	- $f(x) = x^p$ is convex on $x \in [0, \infty)$
- For $p \in [0,1]$, it gets flipped
	- $x^n \geq x$ for $x \in [0,1]$
	- $x^n \leq x$ for $x \geq 1$. 

Triangle inequality: $|a + b| \leq |a| + |b|$ 
*Proof:*  Notice that $\alpha \leq \beta$ and $- \alpha \leq \beta$, implies $|\alpha| \leq \beta$. A configuration which satisfies this is $\alpha = a + b$ and $\beta = |a| + |b|$. QED.

Absolute value is convex
*Proof:* $f(x) = |x|$, 
$$
\begin{align}
f(\lambda_1 x + \lambda_2 y)  & = | \lambda_1 x + \lambda_2 y|\\
& \leq |\lambda_1 x| + | \lambda_2 y| & \text{Triangle inequality}\\
& = \lambda_1 |x| + \lambda_2 |y| \\
& = \lambda_1 f(x) + \lambda_2 f(y)
\end{align}
$$
Note for convexity, $\lambda_1 = 1 - \lambda_2$ s.t. $\lambda_2 \in [0,1]$, meaning $|\lambda_i| = \lambda_i$.

### Vector Inequalities

**Triangle inequality**
Consider vector norm $\| x\| = \sqrt{\sum_i x_i^2}$ 
$$
\| a + b\| \leq \|a \| + \|b\|
$$

Minkowski Inequality: $\|a + b\|_p \leq \|a\|_p + \|b\|_p$, where $\| x\|_p = (\sum_i |x_i|^p)^{1/p}$ for $p \geq 1$.
*Proof*