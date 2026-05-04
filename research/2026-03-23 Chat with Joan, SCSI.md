#meeting 

# Notes

Questions
- Can you "learn" better interpolants which yields easier drifts to train on
  You can think of this as a preconditioner (in an MCMC sense).
- For theoretical models, you should focus on a particular toy problem
	- **(Let's start here)* Extend results to Gaussian Mixture (isotropic covariance) model w/ AWGN. With the purpose of trying to understand the diffusion constant $\epsilon$. $$\pi^*(x)= \sum_k \alpha_k \, \mathcal N(b_k , \Sigma_k) \equiv M(\{\alpha_k, b_k, \Sigma_k\}_k)$$ s.t. $\sum_k \alpha_k = 1$. Now let the algorithm run, how does $\{\alpha_k, b_k , \Sigma_k\}$ update per iteration. As a warmup assume $\Sigma_k = \mathbb I$.	  
	- Or you could ask how does the Gaussian w/ AWGN w/ dynamic diffusion coefficient $\epsilon_t$ play?
	- Role of the diffusion coefficient. It seems that SDE makes you forget about the initial conditions quicker
- On discrete flow matching. We should have a specific problem in mind.
	- For example. You could do language, and the forward model is typos
	- Could you learn to sample from field theories / ising models. Or you could think of non-equilibrium systems where you use the "heating" (which is Heat group) as your corruption channel. 
	  You could use Langvin, but it's ill conditioned because any distribution goes to target. A better choice is annealing.
- Applications
	- CryoEM

# Worked out: AWGN on Gaussian Mixture
## Setup
Assume the **true data distribution** is
$$
\begin{align}
\pi= \sum_i \alpha_i \, \mathcal N(b_i , \Sigma_i) \equiv M(\{\alpha_i, b_i, \Sigma_i\}_i)
\end{align}
$$
For some notation, I'll denote $M(\{\alpha_i, b_i, \Sigma_i\}_i) = \sum_i \alpha_i \, \mathcal N(b_i , \Sigma_i)$, $M = M(\{\alpha_i, b_i, \Sigma_i\}_i)$ and $M^{(k)} = M(\{\alpha_i^{(k)}, b_i^{(k)}, \Sigma_i^{(k)}\}_i)$. A computation that will come in handy is the score
$$
\begin{align}
\nabla \log \tilde M(x) & = \frac{\nabla \tilde M(x)}{M(x)}\\
& = -\frac{\sum_i \alpha_i \mathcal N(x; b_i, \Sigma_i)\, \Sigma_i^{-1}(x-b_i)}{\sum_j \alpha_j \mathcal N(x; b_j, \Sigma_j)}\\
& = - \sum_i w_i(x) \Sigma_i^{-1}(x-b_i) & w_i(x) := \frac{\alpha_i \mathcal N(x; b_i, \Sigma_i)}{\sum_j \alpha_j \, \mathcal N(x; b_j, \Sigma_j)}
\end{align}
$$
We will use the AWGN corruption channel as the **forward model**
$$
\begin{align}
\mathcal F(x) := x + w, \qquad w \sim \mathcal N(0, \mathbb I)
\end{align}
$$
This means the **observed distribution** $\mu = \pi \star \gamma$ (where $\gamma = \mathcal N(0, \mathbb I)$). The convolution is linear and $\gamma_{b, \Sigma} \star \gamma = \gamma_{b, \Sigma + \mathbb I}$ therefore
$$
\begin{align}
\mu = \sum_i \alpha_i \mathcal N(b_i, \Sigma_i + \mathbb I) = M(\{\alpha_i, b_i, \Sigma_i + \mathbb I\}_i)
\end{align}
$$
And finally, we'll use the linear interpolant as the probability space.
$$
\begin{align}
I_t = (1 - \sqrt t ) x + \sqrt t  \mathcal F(x)  = x + \sqrt t\, w
\end{align}
$$
This means the pdf corresponds to the Heat semigroup
$$
\begin{align}
\partial_t p_t = \frac{1}{2} \Delta p_t
\end{align}
$$
Let $\tau = 1-t$ and $q_\tau = p_{1-\tau}$, the backwards PDE becomes
$$
\begin{align}
\partial_\tau q_\tau & = - \nabla \cdot \left( \frac{1+\epsilon}{2} \nabla \log q_\tau \,q_\tau \right) + \frac{\epsilon}{2} \Delta q_\tau
\end{align}
$$
## SCSI Algorithm
We run the backwards PDE multiple times, in the hopes that $\lim_{k \to \infty} q_{\tau=1}^{(k)} = \pi := \sum_i \alpha_i \mathcal N(b_i, \Sigma_i)$.
$$
\begin{align}
\partial_\tau q_\tau^{(k+1)} & = - \frac{1+\epsilon}{2} \nabla \cdot \left(  \nabla \log \pi_{1-\tau}^{(k)} \,q_\tau^{(k+1)} \right) + \frac{\epsilon}{2} \Delta q_\tau^{(k+1)}
\end{align}
$$
where 
- Initial condition of PDE: $q_{\tau=0}^{(k+1)} = \sum_i \alpha_i \mathcal N( b_i^{(k)}, \Sigma_i^{(k)} + \mathbb I)$ 
- Drift field. The $\pi_t^{(k)}(x) = \sum_i \alpha_i^{(k)} \mathcal N(b_i^{(k)}, \Sigma_i^{(k)} + t \mathbb I)$, therefore $$\nabla \log \pi_t^{(k)}(x) = -\frac{\sum_i \alpha_i^{(k)} \mathcal N(x; b_i^{(k)}, \Sigma_i^{(k)} + t \mathbb I)\, (\Sigma_i^{(k)} + t \mathbb I)^{-1} (x-b_i^{(k)})}{\sum_j \alpha_j^{(k)} \mathcal N(x; b_j^{(k)}, \Sigma_j^{(k)} + t \mathbb I)}$$ 
where $\alpha_i^{(k)}, b_i^{(k)}, \Sigma_i^{(k)}$ are read off from the Gaussian mixture model after the time evolution in the PDE, so it is the quantities of $q_{\tau = 1}^{(k)}$.

For simplicity set $\Sigma_i = \Sigma_i^{(k)} =  \mathbb I$. 

Ok I'm pretty sure this analysis won't go through. The score looks like a softmax, so it's not possible to evaluate analytically. For example, if $\alpha_1 = \alpha_2 = 0.5$ and $b_1 = - b_2$ you get $\tanh$, but this causes the PDE not to be solvable.e For example, here's a GMM which goes outside of its class.
![[2026-03-23 Simulation of GMM.png]]
## Mode Weights
Assume the Gaussian mixture is initialized with the correct mean and covariance, so you only need to look at the mode weights being updated. 

Consider the backwards PDE
$$
\begin{align}
\partial_\tau q_\tau & = - \nabla \cdot \left( \frac{1+\epsilon}{2} \nabla \log p_\tau \,q_\tau \right) + \frac{\epsilon}{2} \Delta q_\tau
\end{align}
$$
where $q_{\tau = 0} = M(\{\alpha_i, b_i, \Sigma_i + \mathbb I\}_i)$, and at $q_{\tau = 1} = M(\{\alpha_i^{(k+1)}, b_i, \Sigma_i\}_i)$. The score is the previous iteration $p_\tau = M(\{ \alpha_i^{(k)}, b_i, \Sigma_i\}_i)$. We can write this an ODE on the time evolution of the weights $\alpha_i$.

Consider the Gaussian $\gamma_{b_i, \Sigma_i + (1-\tau) \mathbb I}$, it satisfies the Heat equation 
$$
\begin{align}
\partial_t \gamma_{b_i, \Sigma_i + (1-\tau) \mathbb I}  = - \frac{1}{2} \Delta \gamma_{b_i, \Sigma_i + (1-\tau) \mathbb I}
\end{align}
$$
So let's inspect the terms in the backwards PDE on $q$
$$
\begin{align}
q_\tau  & = \sum_i \alpha_i(\tau) \, \gamma_{b_i, \Sigma_i + (1-t) \mathbb I}\\
\dot q_\tau & = \sum_i \dot \alpha_i (\tau) \, \gamma_{b_i, \Sigma_i + (1-t) \mathbb I} - \frac{1}{2} \alpha_i(\tau)  \Delta \gamma_{b_i, \Sigma_i + (1-t) \mathbb I}\\
\Delta q_\tau & = \sum_i \alpha_i(\tau) \Delta \gamma_{b_i, \Sigma_i + (1-t) \mathbb I}
\end{align}
$$



# Worked out: AWGN on Gaussian, Variable Noise Scheduler
Consider
$$
\begin{align}
dY_\tau =  -\frac{1 + \epsilon_\tau}{2} (\tilde \Sigma + (1-\tau) \mathbb I)^{-1}(Y_\tau - \tilde b) \, d\tau + \sqrt{\epsilon_\tau}\,  dW_{\tau}
\end{align}
$$
For simplicity, let $A(\tau) := -\frac{1 + \epsilon_\tau}{2} (\tilde \Sigma + (1-\tau) \mathbb I)^{-1}$ be the preconditioner matrix. So solve
$$
\begin{align}
dY_\tau = A(\tau) \, (Y_\tau - \tilde b) \, d \tau + \sqrt{\epsilon_\tau} dW_\tau
\end{align}
$$
Let $\tilde Y_\tau = Y_\tau - \tilde b$
$$
\begin{align}
d\tilde Y_\tau = A(\tau) \tilde Y_\tau \, d\tau + \sqrt{\epsilon_\tau} \, dW_\tau 
\end{align}
$$
Let $U_\tau = \Phi(\tau) \tilde Y_\tau$, meaning $dU_\tau = \dot \Phi(\tau) \tilde Y_\tau d\tau + \Phi(\tau) d\tilde Y_\tau$ 
$$
\begin{align}
dU_\tau = [\dot \Phi(\tau) + \Phi(\tau) A(\tau)] \tilde Y_\tau \, d\tau + \Phi(\tau) \sqrt{\epsilon_\tau}\, dW_\tau
\end{align}
$$
So if we can find a $\Phi(\tau)$ s.t. $\frac{d}{d\tau} \Phi = - \Phi \, A(\tau)$, then we'd have a solvable SDE. So let's determine what $\Phi$ should be
$$
\begin{align}
\Phi(\tau) = \exp\left( - \int_0^\tau A(s)\, ds \right)
\end{align}
$$
Since $[A(s),A(s')] = 0$, we don't need any time-ordering. From here, we can diagonalize $\tilde \Sigma = U \Lambda U^T$ and solve this per eigenvalue $\tilde \Sigma v_i = \lambda_i v_i$.
$$
\begin{align}
\int_0^\tau A(s) \, ds = U \left(\int_0^\tau -\frac{1 + \epsilon_s}{2} (\Lambda + (1-s) \mathbb I)^{-1} \, ds \right) U^T & =^{\text{ii'th component}} -\frac{1}{2} \int_0^\tau \frac{1 + \epsilon_s}{\lambda_i + 1 - s} \, ds 
\end{align}
$$
- Let's put a pin in this, I'll solve this for a couple choices of $\epsilon_\tau$

So after finding the proper choice of $\Phi$, we finally have
$$
\begin{align}
dU_\tau & = \Phi(\tau) \sqrt{\epsilon_\tau} dW_\tau\\
U_\tau & = U_0 + \int_0^\tau \Phi(s) \sqrt{\epsilon_s} dW_s\\
\tilde Y_\tau & = \Phi(\tau, 0)\, \tilde Y_0 + \int_0^\tau \Phi(\tau, s)\, \sqrt{\epsilon_s} \, dW_s & \tilde Y_\tau = [\Phi(\tau)]^{-1} U_\tau\\
Y_\tau & = \tilde b + \Phi(\tau, 0) (Y_0 - \tilde b) + \int_0^\tau \Phi(\tau, s) \sqrt{\epsilon_s} \, dW_s
\end{align}
$$
where $\Phi(\tau, s) := [\Phi(\tau)]^{-1}\Phi(s)$.

### Noise Schedulers
We'll consider a couple noise schedules
1. **Linear:** $\epsilon_t = \alpha\, t$. $$
   -\frac{1}{2} \int_0^\tau \frac{1 + \alpha s}{\lambda_i + 1 - s} \,ds  = \frac{\alpha(\lambda_i + 1) + 1}{2} \log \left( \frac{\lambda_i - \tau + 1}{\lambda + 1} \right) + \frac{\alpha \, \tau}{2}
   $$
   Meaning $$\exp \left(- \int A(s)\,ds\right) =^{diagonal} e^{-\alpha \tau / 2}\, \left( \frac{\lambda_i - \tau + 1}{\lambda + 1} \right)^{-\frac{\alpha(\lambda_i + 1) + 1}{2}} $$ In matrix form this becomes $$\begin{align} \Phi(\tau) & = \exp \left(- \int A(s)\,ds\right) = e^{-\alpha \tau /2} [(\tilde \Sigma + (1-\tau) \mathbb I) (\tilde \Sigma +\mathbb I)^{-1}]^{-\frac{\alpha}{2} \tilde \Sigma - \frac{1+\alpha}{2} \mathbb I} \\ \Phi(\tau, s)& = \Phi(\tau)^{-1} \Phi(s) = e^{\alpha(\tau - s)/2}[(\tilde \Sigma + (1-\tau)\mathbb I)(\tilde \Sigma + (1-s) \mathbb I)^{-1}]^{\frac{\alpha}{2} \tilde \Sigma +\frac{1 + \alpha}{2} \mathbb I}\end{align}$$
2. **Whitened scheduler:** $\epsilon_t = \lambda_i - t$. $$-\frac{1}{2} \int_0^\tau \frac{1 + \lambda_i - s }{\lambda_i + 1 - s}\, d\tau = -\frac{\tau}{2}$$ Meaning $$\Phi(\tau) =\exp(-\int A(s) \, ds) = e^{-\tau / 2} \mathbb I$$

Solve the SDE, in particular find $Y_1$ as a function of $Y_0$.
### Linear Scheduler
Let $Y_0 \sim \mathcal N(b, \Sigma + \mathbb I)$, and $\tilde \Sigma = \Sigma_k, \tilde b = b_k$. The SDE becomes
$$
\begin{align}
Y_1 & = \tilde b + \Phi(1, 0) (Y_0 - \tilde b) + \int_0^1 \Phi(1, s) \sqrt{\alpha s} \, dW_s
\end{align}
$$
where 
- $\Phi(\tau, s) = \Phi(\tau)^{-1} \Phi(s) = e^{\alpha(\tau - s)/2}[(\tilde \Sigma + (1-\tau)\mathbb I)(\tilde \Sigma + (1-s) \mathbb I)^{-1}]^{\frac{\alpha}{2} \tilde \Sigma +\frac{1 + \alpha}{2} \mathbb I}$. 
- $\Phi(1,0) = e^{\alpha / 2} [\tilde \Sigma (\tilde \Sigma + \mathbb I)^{-1}]^{\frac{\alpha}{2} \tilde \Sigma +\frac{1 + \alpha}{2} \mathbb I}$ 
- $\Phi(1,s) = e^{\alpha(1-s)/2} [\tilde \Sigma(\tilde \Sigma + (1-s) \mathbb I)^{-1}]^{\frac{\alpha}{2} \tilde \Sigma + \frac{1 + \alpha}{2} \mathbb I}$ 

What is the distribution of $Y_1$? Well since everything is Gaussian, the answer has to look like $Y_1  :\sim \mathcal N(b_{k+1}, \Sigma_{k+1})$.  By computing $\mathbb E[Y_1]$ and $\text{Cov}(Y_1)$ we can deduce $b_{k+1}, \Sigma_{k+1}$.
$$
\begin{align}
b_{k+1} & = \mathbb E[Y_1] \\
& = b_{k} + \Phi(1,0) (b - b_{k})
\end{align}
$$
$$
\begin{align}
\Sigma_{k+1} = \Phi(1,0)(\Sigma + \mathbb I) \Phi(1,0)^T + \alpha \int_0^1 s \Phi(1, s) \Phi(1,s)^T \, ds
\end{align}
$$

$$
\begin{align}
|\Sigma_k - \Sigma| \leq R^k |\Sigma_0 - \Sigma|
\end{align}
$$

## Whitened Scheduler
Recall
$$
\begin{align}
Y_\tau & = \tilde b + \Phi(\tau, 0) (Y_0 - \tilde b) + \int_0^\tau \Phi(\tau, s) \sqrt{\epsilon_s} \, dW_s
\end{align}
$$
where $\Phi(\tau, s) := [\Phi(\tau)]^{-1}\Phi(s)$. Our whitened scheduler, has $\Phi(\tau) = e^{-\tau / 2} \mathbb I$, so we get $\Phi(\tau, s) = e^{(\tau - s)/2} \mathbb I$
$$
\begin{align}
Y_\tau = \tilde b + e^{\tau/2} (Y_0 - \tilde b) + \int_0^\tau e^{(\tau-s)/2} \, \text{diag}(\sqrt{\lambda_i - s}) \, dW_s
\end{align}
$$
where $\text{Law}(Y_0) = \mathcal N(b, \Sigma + \mathbb I)$, set $\tilde b = b_k$. This mean
$$
\begin{align}
\text{Law}\, Y_1 = \mathcal N(b_k + e^{1/2}(b-b_k), ...)
\end{align}
$$
So you can find the update to the mean
$$
\begin{align}
b_{k+1} = b_k + e^{1/2} (b-b_k)
\end{align}
$$
Which converges as $|b_k - b| = |\sqrt{e} - 1|^k | b_0 - b|$

