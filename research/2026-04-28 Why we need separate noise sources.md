#scsi 

Eric proposed a change to the lifted SCSI algorithm
```pseudo
\begin{algorithm}
\begin{algorithmic}
\State $y \sim p_{\text{corrupt}}$
\State $z,\textcolor{blue}{z'}\sim \mathcal N(0, \mathbb I)$, $t \sim [0,1]$
\State $x_{\text{fake}} \leftarrow \Phi(\textcolor{blue}{z'} | y)$
\State $y_{\text{fake}} \leftarrow \mathcal F(x_{\text{fake}})$
\State $I_t \leftarrow \alpha_t z + \beta_t x_{\text{fake}}$, $\dot I_t \leftarrow \dot \alpha_t z + \dot \beta_t x_{\text{fake}}$
\State $\mathcal L = \| b_t(I_t | y_{\text{fake}}) - \dot I_t\|^2 $
\end{algorithmic}
\end{algorithm}
```
Before the $z'$ was just $z$. Like we didn't use two 


and he said to change the $z$'s the SCSI algorithm.

**Observation**
Consider the interpolant
$$
\begin{align}
I_t^{(k)} & = (1-t) Z + t X^{(k-1)}\\
\dot I_t^{(k)} & = X^{(k-1)} - Z
\end{align}
$$
where $X$ is construct from the ODE $dX_t= b^{(k-1)}_t(X_t) dt$, where $X_{t=0} = Z$ and I'm labeling $X_{t=1}=: X^{(k-1)}$. Note we've fixed the RNG for $Z$, meaning it's the same $Z$ in $I_t$ and in the initial condition for the ODE.

Using the MSE loss
$$
\begin{align}
\mathcal  L(\hat b) & = \mathbb E[(\hat b_t(I_t | Y) - \dot I_t)^2]\\
& = \mathbb E\Big[\big(\hat b_t((1-t)Z + t X\; |\; Y) - (X-Z)\big)^2\Big]
\end{align}
$$
you can see that $\hat b_t = 0$ is the trivial minimizer. This is because when $b_t = 0$, then $X_{t=0} = X_{t=1}$, therefore $Z = X$. This kills the right term leaving you with
$$
\begin{align}
\mathcal L(\hat b) & = \mathbb E[\hat b_t(Z|Y)^2] \\
\implies \hat b_t & = 0
\end{align}
$$

**How to fix** We can fix this by choosing two separate $Z$'s
$$
\begin{align}
I_t & = (1-t) Z + t X\\
dX_t & = b_t(X_t)dt \qquad X_{t=0} = Z' \text{ and } X_{t=1} :=X
\end{align}
$$
Now let's check if $\hat b_t =0$ still yields the trivial minimizer. This assumption implies $X = Z'$, which is an iid random variable from $Z$.
$$
\begin{align}
\mathcal L(\hat b) &= \mathbb E\Big[\big(\hat b_t((1-t)Z + t X\; |\; Y) - (X-Z)\big)^2\Big]\\
& = \mathbb E[(Z'-Z)^2] = 2d
\end{align}
$$
which is not zero. 
 
**Question:** Is $\hat b_t = 0$ a local minimizer? 
*Assumption (AWGN Channel):*  $Y = F(X) = X + \sigma W$, where $W \sim \mathcal N(0, \mathbb I)$.

Consider the variational derivative $\frac{d}{d\epsilon}\mathcal L(\hat b + \epsilon h)  \Big|_{\epsilon = 0, \hat b =0}$ . For $\hat b = 0$ to be a local minimizer, this quantity must be zero $\forall \eta$.

$$
\begin{align}
\mathcal L(\hat b + \epsilon h) & = \mathbb E[(b(I_t | Y) + \epsilon h(I_t | Y)  - \dot I_t)^2]\\
\frac{d}{d\epsilon } \mathcal L(\hat b + \epsilon h)& =  \mathbb E[2 (b(I_t | Y) + \epsilon h(I_t | Y)  - \dot I_t) (h(I_t | Y))]\\
\frac{d}{d\epsilon } \mathcal L(\hat b + \epsilon h) \Big|_{\epsilon =0, b=0} & = \mathbb E[2 \dot I_t \, h(I_t | Y)] \\
& = 2 \mathbb E[ (X - Z) h(I_t | Y)]\\
& = 2 \mathbb E[(Z' - Z) h(I_t | Y)]\\
& = 2 \mathbb E[\mathbb E[Z' - Z | I_t, Y] \, h(I_t |Y)]
\end{align}
$$
So $\hat b = 0$ is a critical point i.f.f. $\mathbb E[Z' - Z | I_t , Y] = 0$  

So we have
- $Z,Z' \overset{iid}{\sim} \mathcal N(0, \mathbb I)$
- $I_t = (1-t) Z + t Z'$
- $Y = Z' + \sigma W$, $W \sim \mathcal N(0, \mathbb I)$

This is computable because they're jointly Gaussian. Let $A = Z'-Z$. The conditional expectation becomes 
$$
\begin{align}
\mathbb E[A | I_t , Y] = (\Sigma_{A,I_t} \ \Sigma_{A,Y}) \begin{pmatrix}
\Sigma_{I_t} & \Sigma_{I_t , Y}\\
\Sigma_{I_t , Y} & \Sigma_{Y}
\end{pmatrix}^{-1} \begin{pmatrix} I_t \\ Y
\end{pmatrix}
\end{align}
$$

- $\mathbb E[A]= \mathbb E[I_t] = \mathbb E[Y] = 0$
- $\text{Var}(A) = \text{Var}(Z') + \text{Var}(Z) = 2$
- $\text{Var}(I_t) = (1-t)^2 \text{Var}(Z) + t^2\text{Var}(Z') = (1-t)^2 + t^2$
- $\text{Var}(Y) = 1 + \sigma^2$ 
- $\text{Cov}(A,I_t) = \text{Cov}(Z', I_t) - \text{Cov}(Z , I_t) = (1-t) - t = 1 - 2t$  
- $\text{Cov}(A, Y) = \text{Cov}(Z' - Z, Z) = -1$
- $\text{Cov}(I_t, Y) = t \, \text{Cov}(Z', Z')= t$

Mathematica says
$$
\begin{align}
\mathbb E[A | I_t , Y] =  \frac{I_t [(1 -t)+ \sigma^2(1-2t)] - Y[1-t]}{(-1+t)^2 + \sigma^2 (1-2t + 2t^2)}
\end{align}
$$
The denominator is always positive, so this equals zero when
$$
\begin{align}
I_t[(1-t) + \sigma^2(1-2t)] = Y (1-t)
\end{align}
$$
Ways to get this to be zero
- Set $t$ s.t. LHS and RHS are both zero, which is not possible, Except when $\sigma = 0$ (AWGN channel is off).
- Find set where $I_t, Y$ coincide according to equation above. However this set is measure zero, so you wouldn't hit it in practice.

____


