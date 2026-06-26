#scsi 

# Summary

- Played around with point cloud representation more. Still found collapse. Couldn't find bugs in code. 
	- I think it's just point cloud representation & random rotations & SCSI aren't compatible.
	- SO(3) causes objects to shrink to a point
	- SO(2) causes objects to flatten to a pancake

# Notes

I've been working on the point cloud representation. I'm still finding collapse
![[2026-06-18 so3 em0.png]]
![[2026-06-18 so3 em23.png]]

I have a heuristic explanation (reference for my journal system [[2026-06-17 Instability of Point Cloud Representation]], pdfs can't clic, this)

The SCSI drift and it's corresponding loss function are
$$
\begin{align}
b_t^{(k+1)}(x,y)& = \mathbb E \left[ \dot \alpha_t Z + \dot \beta_t X^{(k)} | (I_t^{(k)})^X = x , Y^{(k)} = y \right]\\
& = \frac{\dot \alpha}{\alpha_t} x + \beta_t \left( \frac{\dot \beta_t}{\beta_t} - \frac{\dot \alpha_t}{\alpha_t} \right) \mathbb E \left[ X^{(k)} | (I_t^{(k)})^X = x , Y^{(k)} = y \right]
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

**Implications**
I think this would imply if I do an SO(2) rotation (random rotation about single axis, instead of 2), it would flatten to a pancake.
![[2026-06-18 so2 em29.png]]


