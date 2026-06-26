#scsi 

Consider the SCSI interpolant
$$
\begin{align}
I_t^{(k)} & = \begin{pmatrix} \alpha_t Z + \beta_t X^{(k)}\\
Y^{(k)}\end{pmatrix}\\
\text{such that }X^{(k)} & = \Phi_{\Theta^{(k)}}(Z' | Y)\\
Y^{(k)} & = \mathcal F(X^{(k)})
\end{align}
$$
where $\Phi_{\Theta^{(k)}}$ is the result of the previous the flow $dX_t = b_t^{(k)}(X_t | Y)$ initalized at $X_{t=0} = Z' \overset{(d)}{=} Z$ ; and $\mathcal F(x) = P \circ G \circ x + W$ is the CryoEM corruption (projection, random rotation, additive white noise).

In my case, we are using the point-cloud representation for 3d objects $X^{(k)} \in \mathbb R^{N \times 3}$.

The SCSI drift and it's corresponding loss function are
$$
\begin{align}
b_t^{(k+1)}(x,y)& = \mathbb E \left[ \dot \alpha_t Z + \dot \beta_t X^{(k)} | (I_t^{(k)})^X = x , Y^{(k)} = y \right]\\
& = \frac{\dot \alpha}{\alpha_t} x + \beta_t \left( \frac{\dot \beta_t}{\beta_t} - \frac{\dot \alpha_t}{\alpha_t} \right) \mathbb E \left[ X^{(k)} | (I_t^{(k)})^X = x , Y^{(k)} = y \right] \\
\mathcal L_{\textsf{SCSI}}[b^{(k+1)}] & = \left | b_t^{(k+1)}(\alpha_t Z + \beta_t X^{(k)}, \ Y^{(k)}) - \dot \alpha_t Z + \dot \beta_t X^{(k)}  \right |^2
\end{align}
$$
The drift is dependent on the conditional expectation $\mathbb E \left[ X^{(k)} | I_t = x , Y^{(k)} = y \right]$. You can also write a loss function on the $X$-predictor, denoted as $f$
$$
\begin{align}
\mathcal L_{\textsf{SCSI}}[f] & = \left | f_t(I_t, Y^{(k)}) - X^{(k)} \right |^2 \\
& = \left | f_t(\alpha_t Z + \beta_t X^{(k)}, Y^{(k)}) - X^{(k)} \right |
\end{align}
$$
**My Heuristic Hypothesis of what's happening**
- During training
	1. At early times $t \approx 0$, this is $|f_0(Z , Y^{(k)}) - X^{(k)}|$. Since $Y^{(k)}$ is a random projection, there are many potential outputs of $f$ which are consistent with $Y$, so to minimize the error, the network regresses to the mean $f \to 0$.
	2. At late times $t \approx 1$, this is $|f_1(X^{(k)}, Y^{(k)}) - X^{(k)}|$ so the network learns to ignore $Y$ and we get $f(x,y) \approx x$ 
	3. At intermediate times, this is $|f_t(\alpha_t Z + \beta_t X^{(k)} , Y^{(k)}) - X^{(k)}|^2$. The network can choose to ignore the conditioner $Y$, and attempt to act as a denoiser; implying $f(Z + X , Y) \approx X$. 
- At inference
	1. At $t \approx 0$. For the linear interpolant $\alpha_t = 1-t$, this implies $b_{t}(x,y) \approx^{t \approx 0} \frac{-t}{1-t} x$ which pushes the particles more towards the origin.
	2. At $t \approx 1$. We hypothesized $f(x,y) \approx x$, which would imply $b_t(x,y) \approx 0$ (particles don't move).
	3. At intermediate times. We also hypothesized $f(x,y) \approx x$, thus particles don't move.

In summary the particles will get pushed towards the origin, and then stagnate.