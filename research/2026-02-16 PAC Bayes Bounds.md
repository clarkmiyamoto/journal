#math #learningtheory

Reference: https://arxiv.org/pdf/2110.11216

# Setup
Say you observe feature-label pairs from $(X,Y) \sim P$ some distribution. Our objective is to find model parameters $\theta \in \Theta$ which minimizes the (generalization) risk
$$
R(\theta) = \mathbb E_{(X,Y) \sim P}[\ell(f_\theta(X), Y))]
$$
In practice, you have finite $n$ observations of the distribution $(X_i, Y_i) \sim^{iid} P$. The empirical risk $r$ is defined as
$$
r(\theta) = \frac{1}{n} \sum_{i=1}^n \ell_i(\theta)
$$
where $\ell_i(\theta) = \ell(f_\theta(X_i), Y_i)$. Since the data is drawn iid, this is an unbiased estimator $\mathbb E_{(X_1, Y_1), ..., (X_n, Y_n)} [r(\theta)] = R(\theta)$. For shorthand, I'll denote $\mathbb E_{(X_1, Y_1), ..., (X_n, Y_n)} = \mathbb E_S$. However the variance of this estimator may not be as simplistic... 
# PAC Bounds
**Proposition** For any $\theta \in \Theta$ and $\delta \in [0,1]$
$$
\mathbb P_S \left( R(\theta) > r(\theta) + C \sqrt{\frac{\log \frac{1}{\delta}}{2n}}  \right) \leq \delta
$$

**Lemma** (Hoffeding Inequality). Let $U_i$ be iid random variables which are bounded on $[a, b]$. Then for any $t \geq 0$.
$$
\mathbb E[e^{t \sum_{i=1}^n U_i - \mathbb E[U_i]}] \leq e^{n t^2(b-a)^2 / 8}
$$
*Proof of Proposition:* Using Hoffeding where $U_i = \mathbb E[\ell_i(\theta)] - \ell_i(\theta)$ 
$$
\mathbb E_S[e^{tn (R(\theta) - r(\theta))}] \leq e^{nt^2 C^2/8}
$$
Now we can use Markov to bound the probability, let $s >0$
$$
\begin{align}
\mathbb P_S(R(\theta) - r(\theta) > s) & = \mathbb P_S(e^{t n(R(\theta) - r(\theta))} > e^{nts})\\
& \leq \frac{\mathbb E_S[e^{nt (R(\theta) - r(\theta))}]}{e^{nts}}\\
& \leq e^{nt^2 C^2 / 8 - n ts}
\end{align}
$$
Since $t$ is arbitrary we can tighten the bound by minimizing over $t$, this yields $t=4s/C^2$.
$$
\mathbb P_S(R(\theta)  >r(\theta) + s) \leq e^{-2ns^2/C^2}
$$
If we let the LHS be defined at $\delta$. We can get our proposition. QED.

# PAC Bayes Bounds
**Theorem** (McAllester 1999). For any $\delta > 0$
$$
\mathbb P_S\left(\mathbb E_{\theta \sim p}[R(\theta)] \leq \mathbb E_{\theta \sim p}[r(\theta)] + \sqrt{\frac{\text{KL}(p\| \pi) + \log \frac{1}{\delta} + \frac{5}{2} \log (n) + 8}{2n - 1}} \right) \geq1 -\delta
$$


