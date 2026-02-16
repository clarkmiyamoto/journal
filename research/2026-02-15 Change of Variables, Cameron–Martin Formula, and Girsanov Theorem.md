#stochasticprocess #math 
References
- https://galton.uchicago.edu/~lalley/Courses/390/Lecture8.pdf
- https://galton.uchicago.edu/~lalley/Courses/390/Lecture10.pdf

```table-of-contents
```
# Cameron-Martin Formula
## Exponential Martingales
For any $\theta \in \mathbb R$, consider the stochastic process
$$
Z_t^\theta = \exp \left(\theta W_t- \theta^2 t /2 \right)
$$
**Fact:** $\mathbb E[e^{\theta W_t}] = e^{\theta^2 t / 2}$, therefore $\mathbb E[Z_t] = 1$. 

**Fact:** This is a martingale
*Proof:* Recall a martingale is a stochastic process s.t. $\mathbb E[|Z_t|^p] < \infty$ and $\mathbb E[Z_{s} | \mathcal F_t] = Z_t$ for all $s > t$. The first property is trivial, put the $p$ inside the exponential and take the expectation. The second property can be verified by direct calculation
$$
\begin{align}
\mathbb E[Z_{t+s} | \mathcal F_t] & = \mathbb E \left[ \exp \left(\theta W_{t+s}- \theta^2 (t+s) /2 \right) | \mathcal F_t \right]\\
& = \mathbb E[ \exp (\theta W_{s}) \exp(\theta (W_{t+s} - W_s)) | \mathcal F_s] \, e^{-\theta^2 (t+s)/2}\\
& = \exp(\theta W_s) \mathbb E[\exp(\theta (W_{t+s} - W_s)) | \mathcal F_s] \, e^{-\theta^2 (t+s)/2}\\
& = \exp(\theta W_s) \, e^{\theta^2 t / 2} e^{-\theta^2(t+s) /2 }\\
& = \exp(\theta W_s -\theta^2 s / 2) = Z_s
\end{align}
$$
QED.
## Radon Nikdoym Derivative and Likelihood Ratio
Let $\mathbb P, \mathbb Q$ be a probability measure defined the same state space. You can define a positive random variable $Z$ on $\mathbb P$, with $\mathbb E_{\mathbb P}[Z] = 1$ , which allows you to write the measure for $\mathbb Q$. That is for any event $A$.
$$
\begin{align}
\mathbb Q(A) & = \mathbb E_{\mathbb P}[\mathbb 1_A \, Z]
\end{align}
$$
where $\mathbb 1$ is the indicator function.

The **Radon Nikdoym derivative** (also known as the **likelihood ratio**) is an example of the change of measure described previously. It is a random variable $Z := \frac{d\mathbb Q}{d \mathbb P}$ with $\mathbb E[Z] = 1$ 

**Remark:** The exponential martingales satisfy the requirements to be a radon nikdoym derivative (positive random variable, expectation value of 1). But the question is measure does it transform into another?

## Statement & Proof of Simplified Cameron-Martin Theorem
**Theorem:** Consider the exponential martingale $Z^\theta_t \equiv \exp \left(\theta W_t- \theta^2 t /2 \right)$. Denote $\mathbb P_\theta$ and $\mathbb E_\theta$ as the measure & expectation value defined by $Z_t^\theta$. I'll define $\mathbb P_0, \mathbb E_0$ by choosing $Z_t^\theta$ as the Randon Nikdoym derivative
$$
\mathbb P_0(A) = \mathbb E_\theta [(Z^\theta_T)^{-1} \, \mathbb 1_A]
$$
Under probability measure $P_\theta$, the process $W_t$ is equal in law to a Brownian motion with drift $\theta$. Equivalently the process $W_t$ has the same law under $P_\theta$ as process $W_t + \theta t$ has under $P = P_0$.

*Proof:* We want to prove that the joint over $(\Delta W)_i \equiv W_{t_i} - W_{t_{i-1}}$ under $P_\theta$ is equal in distribution to $(\Delta W + \theta \Delta t)_i$ under $P_0$. Where the index in the time $t_i$ is given by $i \in 0,1,...,K$, with boundary conditions $t_K = T$ and $t_0 = 0$. 

To prove equality in distribution, we can show the characteristic functions are equal
$$
\begin{align}
\mathbb E_{\theta} \left[\exp\left(\sum_i \lambda (\Delta W)_i\right)\right] & = \mathbb E_0 \left[ Z^\theta_T \, \exp \left(\sum_i \lambda (\Delta W)_i  \right) \right] \\
& = \mathbb E_0 \left[ \frac{Z_{t_K}^\theta }{Z_{t_{K-1}}^\theta} \frac{Z_{t_{K-1}}^\theta }{Z_{t_{K-2}}^\theta} ... \frac{Z_{t_1}^\theta}{Z_{t_0}^\theta} Z_{t_0}^\theta \, \exp \left(\sum_i \lambda (\Delta W)_i \right)  \right]\\
& = \mathbb E_0 \left[ \exp\left( \sum_{i=1}^K \theta (\Delta W)_i - \theta^2 (\Delta t)_i /2  \right) Z_{t_0}^\theta \, \exp \left(\sum_i \lambda (\Delta W)_i \right) \right]\\
& =  \mathbb E_0 \left[ \exp\left( \sum_{i=1}^K \theta (\Delta W)_i - \theta^2 (\Delta t)_i /2  +\lambda (\Delta W)_i \right) \right] \mathbb E_0 \left [Z_{t_0}^\theta  \right]\\
& = \mathbb E_0[...] \mathbb E_0 \left [ \exp \left(\theta W_0- \theta^2 t_0 /2 \right) \right]\\
& = \mathbb E_{0}[...] \times 1\\
& = \mathbb E_0 \left[ \exp\left( \sum_{i=1}^K \theta (\Delta W)_i - \theta^2 (\Delta t)_i /2  +\lambda (\Delta W)_i \right) \right]\\
& = \prod_{i=1}^K \exp \left(\frac{(\theta + \lambda)^2 - \theta^2}{2} (\Delta t)_i \right)\\
& = \prod_{i=1}^K \exp \left( \left(\frac{\lambda^2}{2} +  \theta \lambda \right) (\Delta t)_i \right)\\
& = \mathbb E_{0} \left[ \exp\left(\sum_i \lambda ((\Delta W)_i  + \theta (\Delta t)_i) \right) \right]
\end{align}
$$
This is the characteristic function for $W_t + \theta t$. QED.

## General Cameron-Martin Theorem
The theorem describes how Weiner measures change under translations of elements of the Cameron–Martin Hilbert space. 

For motivation let's do this on Gaussian measures. Consider the map $T_h(x) = x + h$ and denote $(T_h)_\#$ as the corresponding pushforward. The change of Radon Nikdoym derivative is given by 
$$
\frac{d \, (T_h)_\# \mu}{d\mu} = \exp \left(\langle h, x\rangle - \frac{1}{2} \|h \|^2 \right)
$$
where $\langle \cdot, \star \rangle$ is the Euclidean inner product. This can be extended to Weiner measures.

# Girsanov Theorem
Girsanov Theorem is the most important special case of Cameron-Martin theorem. 

## Exponential Martingales
Once again consider an exponential martingale. But here it's slightly modified
$$
Z_t = \exp \left(\int_0^t \theta_s dW_s - \frac{1}{2}\int_0^t \theta_s^2 \, ds \right)
$$
**Fact:** $\mathbb E[Z_t] = 1$.
*Proof:* Notice that $\int_0^t \theta_s dW_s \sim \mathcal N(0, \int_0^t \theta_s^2 ds)$, and then use the moment generating function of a Gaussian to see $\mathbb E[\exp(\int_0^t \theta_s dW_s)] = \exp( \frac{1}{2} \int_0^t \theta_s^2 \ ds)$. QED.

## Statement & Proof of Girsanov Theorem
Consider the change of variables from $\mathbb P \to \mathbb Q$ implicitly defined by $Z_t$; $\mathbb E_{\mathbb Q}[f(x)] = \mathbb E_{\mathbb P}[Z_T  f(x)]$. Girsinov theorem is a statement about a Wiener process $W_t$ defined on $\mathbb P$ is equal in law to $\tilde W_t$ defined on $\mathbb Q$, where $\tilde W_t$ is
$$
\tilde W_t = W_t - \int_0^t \theta_s \, ds
$$
*Proof:* Using the generalized Girsanov theorem, this result comes for free. Instead let's prove this by hand using the characteristic function
$$
\begin{align}
\mathbb E_{\mathbb Q} \left[ \exp \left( \lambda \tilde W_t \right) \right] & = \mathbb E_{\mathbb Q} \left[ \exp \left( \lambda \left(W_t - \int_0^t \theta_s ds \right) \right) \right]\\
& = \mathbb E_{\mathbb P} \left[ \exp \left(\int_0^t \theta_s dW_s - \frac{1}{2}\int_0^t \theta_s^2 \, ds + \lambda W_t - \lambda \int_0^t \theta_s \, ds \right) \right]
\end{align}
$$
Notice that $\lambda W_t \sim \mathcal N(0, \lambda^2 \, t)$ and $\int_0^t \theta_s dW_s \sim \mathcal N(0, \int_0^t \theta_s^2 ds)$. The sum of them is distributed as $\mathcal N(0, \lambda t + \int_0^t \theta_s^2 \, ds + 2 \lambda \int_0^t \theta_s \, ds)$. We can compute the expectation value
$$
\begin{align}
& =\exp\left(\frac{1}{2}\lambda^2 t + \frac{1}{2} \int_0^t \theta_s^2 \, ds + \lambda \int_0^t \theta_s \, ds  \right) \exp \left(- \frac{1}{2}\int_0^t \theta_s^2 \,ds - \lambda \int_0^t \theta_s \, ds\right)\\
& = e^{\lambda^2 t / 2} = \mathbb E_{\mathbb P} [e^{\lambda W_t}]
\end{align}
$$
which is the characteristic function for $W_t$ under $\mathbb P$. QED. 