#scsi 

Consider the lifted joint interpolant on $\Omega = \mathcal X \times \mathcal Y$
$$
\begin{align}
I_t^X & = \alpha_t Z_X + \beta_t X\\
I_t^Y & = \alpha_t \mathcal F(X) + \beta_t  Z_Y
\end{align}
$$
Let $\omega \in \Omega$. The idea is you initalize your ODE solver at $(z_x, y)$ ($y$ is a corrupted $x$) and the flow will take you to the corresponding $x$.

This doesn't work because at $t=1$ you have lost information on your inputted $y \sim \mathcal F(X)$.

Here's a dumb idea flip it
$$
\begin{align}
I_t^X & = \alpha_t Z_X + \beta_t X\\
I_t^Y & = \alpha_t Z_Y + \beta_t \mathcal F(X)
\end{align}
$$
The information in $\mathcal F(X)$ becomes useful at the same time as $X$ needs to converge. The only problem is we want to generate $X$'s corresponding to a $y \sim \mathcal F(X)$. However since the path is linear between white noise and $y$, we can just construct the path ourselves by numerically just getting a single noise $z \sim Z_y$ and lineararly interpolating it, and feeding it into the ODE solver

```pseudo
\begin{algorithm}
\caption{Supervised Algorithm, Inference}
\begin{algorithmic}
\Require ODE integrator times $\{t_i\}_i$, pretrained drift $b_t(x,y)$, observed corrupted data $y$
\State Sample noise $z_y \sim Z_Y$
\State Construct $I_{t_i}^Y = \alpha_{t_i} z_y + \beta_t y$, don't resample $z$ for every time.
\State Solve ODE. Initalize at $\omega_{0} = (z_x, I_{t_0})$. Integrate but throw away $y$ output and replace with $I_{t_i}^Y$.
\Return Final state of ODE, the $X$ component is the decorrupted $y$.
\end{algorithmic}
\end{algorithm}
```
