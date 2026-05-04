#scsi

```table-of-contents
```
# Summary of Result
**Setup:** Consider a true data distribution $\pi = \mathcal N(\mu, \Sigma)$, processed by a AWGN channel $\mathcal F(X) = X + \sigma W$ s.t. $W \sim \mathcal N(0, \mathbb I)$. The corrupted data distribution is $Y = \mathcal F(X) \sim \mathcal N(\mu, \Sigma + \sigma^2 \mathbb I) =: \nu$. 

**Objective:** Using the lifted conditional SCSI algorithm, we want to find the convergence rate $\|\Sigma_k - \Sigma\|$ 

**Lifted Conditional SCSI algorithm**
1. Let $X^{(k)} \sim \mathcal N(\mu_k , \Sigma_k)$, and $Y^{(k)} = \mathcal F(X^{(k)}) \sim \mathcal N(\mu_k, \Sigma_k + \sigma^2 \mathbb I)$ 
2. Construct interpolant $$I_t^{(k)} | Y^{(k)}=y \overset{d}{=} (1-t) Z+ t \, X^{(k)}$$
3. Assume you learn optimal drift $$b_t(x|y) = \mathbb E[\dot I_t | I_t = x, Y^{(k)} = y]$$
4. Solve ODE $$\frac{dX_t}{dt} = b_t(X_t | Y), \qquad X_{t=0} \overset{iid}{\sim} \mathcal N(0, \mathbb I)$$ and set $X_{t=1}:= X^{(k+1)} :\sim \mathcal N(\mu_{k+1}, \Sigma_{k+1})$.

	> A future direction is to try boundary $X_{t=0} = Z$, which is the $Z$ in the interpolant and see if there's some sort of mode collapse or multiple fixed points.

From here we can look at the convergence rate of $\|\Sigma_k - \Sigma\|$ and $\|\mu_k - \mu\|$ 

**Result:**
The mean and covariance update like
$$
\begin{align}
\mu_{k+1} & = \mu_k + B_k (\mu - \mu_k)\\
\Sigma_{k+1} & = \Sigma_k + B_k (\Sigma - \Sigma_k) B_k\\
\text{s.t. }B_k & = \Sigma_k(\Sigma_k + \sigma^2 \mathbb I)^{-1}
\end{align}
$$
Let $\eta = \min (\lambda_{min}(\Sigma), \lambda_{min}(\Sigma_0))$ and assume $\eta > 0$, then
$$
\begin{align}
\boxed{\|\Sigma - \Sigma_k \| \leq \left[ 1 - \left( \frac{\eta}{\sigma^2 + \eta} \right)^2 \right] \| \Sigma - \Sigma_k\|}
\end{align}
$$
> Compared to the original algorithm, **this corresponds to $\epsilon = 1$**. However we solved this using ODE dynamics, so the lifted algorithm converges slower than the original!  This seems to agree with our experience with the code, that the lifted was converging much slower...

___
# Proof
**1) Calculating drift**
Consider the lifted conditional interpolant
$$
\begin{align}
I_t^{(k)} | \mathcal F(X^{(k)}) =y & \overset{d}{=} (1-t) Z + t X^{(k)}\\
\end{align}
$$
The optimal drift is
$$
\begin{align}
b_t^{(k+1)}(x|y) & = \mathbb E[\dot I_t^{(k)} | I_t^{(k)} = x , \mathcal F(X^{(k)}) = y]\\
& = \mathbb E[X^{(k)} - Z | I_t^{(k)} = x , \mathcal F(X^{(k)}) = y]
\end{align}
$$
The conditional expectation can be calculated because everything is jointly Gaussian.
$$
\begin{align}
\mathbb E[A | B=b] = \mu_A + \Sigma_{AB} \Sigma_{BB}^{-1} (b-\mu_B)
\end{align}
$$
- $\mathbb E[X^{(k)} - Z] = \mu_k$
- $\mathbb E[I_t^{(k)}] = t \mu_k$
- $\mathbb E[Y^{(k)}] = \mu_k$
- $\text{Var}(I_t^{(k)}) = t^2 \Sigma_k + (1-t)^2 \mathbb I$
- $\text{Var}(Y^{(k)}) = \Sigma_k + \sigma^2 \mathbb I$
- $\text{Cov}(I_t^{(k)}, Y^{(k)}) = t \Sigma_k$
- $\text{Cov}(\dot I_t^{(k)}, I_t^{(k)}) = t \Sigma_k - (1-t) \mathbb I$
- $\text{Cov}(\dot I_t^{(k)}, Y^{(k)}) = \Sigma_k$ 
Leaving us with
$$
\begin{align}
b_t^{(k+1)}(x|y)& = \mu_k + (t \Sigma_k - (1-t) \mathbb I, \; \Sigma_k) \begin{pmatrix} (1-t)^2 \mathbb I + t^2 \Sigma_k & t \Sigma_k \\ t \Sigma_k & \Sigma_k + \sigma^2 \mathbb I\end{pmatrix}^{-1} \begin{pmatrix} x - t \mu_k \\ y - \mu_k \end{pmatrix}\\
\end{align}
$$
Make the relabeling $S_{k,t} = t\Sigma_k - (1-t) \mathbb I$, $\tilde S_{k,t} = t^2 \Sigma_k + (1-t)^2 \mathbb I$, and $\tilde y_k = y - \mu_k$.
Now we need to invert the block matrix $C := \begin{pmatrix} \tilde S_{k,t} & t \Sigma_k \\ t \Sigma_k & \Sigma_k + \sigma^2 \mathbb I\end{pmatrix}$.  Since everything commutes with each other, we can invert using scalar inverse. $C^{-1} = \frac{1}{ad - bc} \begin{pmatrix}d & -b \\ -c & a \end{pmatrix}$
$$
\begin{align}
\therefore (S_{k,t} ,\;  \Sigma_k) \, C^{-1} & = \Delta^{-1}_{k,t} (S_{k,t} ,\;  \Sigma_k) \begin{pmatrix} \Sigma_k + \sigma^2 \mathbb I & -t \Sigma_k \\ - t \Sigma_k & \tilde S_{k,t} \end{pmatrix}\\
& = \Delta^{-1}_{k,t}  \begin{pmatrix}\underbrace{ S_{k,t} (\Sigma_k + \sigma^2 \mathbb I) - t \Sigma_k^2}_{A_{k,t}}, & \underbrace{\Sigma_k (\tilde S_{k,t} -tS_{k,t})}_{B_{k,t}} \end{pmatrix}
\end{align}
$$
where $\Delta_{k,t}$ denotes the "block determinant"
$$
\begin{align}
\Delta_{k,t} &= \tilde S_{k,t}(\Sigma_k + \sigma^2 \mathbb I) - t^2 \Sigma_k^2
\\ & = (t^2 \Sigma_k + (1-t)^2 \mathbb I)(\Sigma_k + \sigma^2 \mathbb I) - t^2 \Sigma_k^2 \\ 
&= t^2 \Sigma_k^2 + t^2 \sigma^2 \Sigma_k + (1-t)^2 \Sigma_k + (1-t)^2 \sigma^2 \mathbb I - t^2 \Sigma_k^2 \\
&= (t^2 \sigma^2 + (1-t)^2) \Sigma_k + (1-t)^2 \sigma^2 \mathbb I
\end{align}
$$
Simplifying $A_{k,t}$ and $B_{k,t}$
$$
\begin{align} A_{k,t} &= S_{k,t}(\Sigma_k + \sigma^2 \mathbb I) - t \Sigma_k^2 \\ 
&= (t \Sigma_k - (1-t)\mathbb I)(\Sigma_k + \sigma^2 \mathbb I) - t \Sigma_k^2 \\ 
&= t\Sigma_k^2 + t\sigma^2\Sigma_k - (1-t)\Sigma_k - (1-t)\sigma^2\mathbb I - t\Sigma_k^2 \\ 
&= (t\sigma^2 - 1 + t) \Sigma_k - (1-t)\sigma^2 \mathbb I 
\end{align}
$$
$$
\begin{align}
B_{k,t} & = \Sigma_k\big([t^2 \Sigma_k + (1-t)^2 \mathbb I] - [t^2 \Sigma_k - t(1-t) \mathbb I ] \big)\\
& = \Sigma_k((1-t) +t)(1-t) \\
& = (1-t)\, \Sigma_k
\end{align}
$$
Therefore the drift becomes
$$
\begin{align}
b_t^{(k+1)}(x | y) = \mu_k + \Delta^{-1}_{k,t}\big[A_{k,t} (x-t\mu_k) + B_{k,t}(y-\mu_k) \big]
\end{align}
$$

**2) Solve ODE** with boundary condition $x_{t=0} = Z' \overset{iid}{\sim} \mathcal N(0, \mathbb I)$ and $y \overset{d}{=} Y \sim \mathcal N(\mu, \Sigma + \sigma^2 \mathbb I)$ 
$$
\begin{align}
\frac{dx_t}{dt} & = b_t^{(k+1)}(x_t | y)\\
& = \mu_k + \Delta^{-1}_{k,t}\big[A_{k,t} (x_t-t\mu_k) + B_{k,t}(y-\mu_k) \big]
\end{align}
$$
Let $z_t = x_t - t\mu_k \implies \dot z_t = \dot x_t - \mu_k$
$$
\begin{align}
\frac{dz_t}{dt} = \Delta_{k,t}^{-1}[A_{k,t}z_t + B_{k,t} (y- \mu_k)]
\end{align}
$$
Notice
$$
\begin{align} 
\frac{d}{dt} \Delta_{k,t} &= \frac{d}{dt} \big[ (t^2 \sigma^2 + (1-t)^2) \Sigma_k + (1-t)^2 \sigma^2 \mathbb{I} \big] \\ 
&= (2t\sigma^2 - 2(1-t)) \Sigma_k - 2(1-t)\sigma^2 \mathbb{I} \\ 
&= 2 \big[ (t\sigma^2 - (1-t))\Sigma_k - (1-t)\sigma^2 \mathbb{I} \big] \\ &= 2 A_{k,t} 
\end{align}
$$
Additionally just set all the non-$z_t$ terms to this $g_{t,k,y}$ placeholder
$$
\begin{align}
\frac{dz_t}{dt} = \frac{1}{2}\Delta^{-1}_{k,t} \dot \Delta_{k,t} z_t + g_{t,k,y}
\end{align}
$$
Now do substitution $a_t = \Phi_{t,k} z_t$, then $\dot a_t = \dot \Phi_{t,k} z_t + \Phi_{t,k} \dot z_t$ 
$$
\begin{align}
\frac{da_t}{dt} & = \dot \Phi_{t,k} z_t +\frac{1}{2}\Phi_{t,k} \Delta^{-1}_{k,t} \dot \Delta_{k,t} z_t + \Phi_{t,k} \,  g_{t,k,y}\\
& = \left(\dot \Phi_{t,k} + \frac{1}{2}\Phi_{t,k} \Delta^{-1}_{k,t} \dot \Delta_{k,t}  \right) z_t + \Phi_{t,k}  \, g_{t,k,y}
\end{align}
$$
I will choose $\Phi_{t,k}$ such that the middle term vanishes
$$
\begin{align}
\frac{d\Phi_{t,k} }{dt}  + \frac{1}{2}\Phi_{t,k} \Delta^{-1}_{k,t} \dot \Delta_{k,t} = 0 \implies \Phi_{t,k} = \Delta^{-1/2}_{k,t}
\end{align}
$$

Now we can start unpacking our answer
$$
\begin{align}
\frac{da_t}{dt} = g_{t,k,y} \implies a_1 = a_0 + \int_0^1 \Phi_{t,k} \,  g_{t,k,y} dt
\end{align}
$$
Recall $a_t = \Phi_{t,k} z_t = \Delta_{k,t}^{-1/2} z_t$ 
$$
\begin{align}
z_1 = \Delta^{1/2}_{k,1} \Delta^{-1/2}_{k,0} z_0 + \Delta^{1/2}_{k,1} \int_0^1 \Delta_{k,t}^{-3/2}  B_{k,t}(y-\mu_k) \, dt
\end{align}
$$
Recall
- $\Delta_{k,t} = (t^2 \sigma^2 + (1-t)^2) \Sigma_k + (1-t)^2 \sigma^2 \mathbb I$
	- $\Delta_{k,0} = \Sigma_k + \sigma^2 \mathbb I$
	- $\Delta_{k,1} = \sigma^2 \Sigma_k$
- $B_{k,t} = (1-t) \Sigma_k$

Since all terms commute, we can integrate the matrix in it's eigenbasis (treat it like scalars). Mathematica says the result is
$$
\begin{align}
\int_0^1 \Delta_{k,t}^{-3/2}  B_{k,t} \, dt  = \frac{1}{\sigma}(\Sigma_k + \sigma^2 \mathbb I)^{-1} \Sigma_{k}^{1/2}
\end{align}
$$
Plugging this in
$$
\begin{align}
z_1 & = \Delta^{1/2}_{k,1} \Delta^{-1/2}_{k,0} z_0 + \frac{1}{\sigma} \Delta^{1/2}_{k,1} (\Sigma_k + \sigma^2 \mathbb I)^{-1} \Sigma_{k}^{1/2}(y-\mu_k) \\
& = (\sigma \Sigma_k^{1/2}) (\Sigma_k + \sigma^2 \mathbb I)^{-1/2} z_0 +  \frac{1}{\sigma} (\sigma \Sigma^{1/2}_k) (\Sigma_k + \sigma^2 \mathbb I)^{-1} \Sigma_{k}^{1/2}(y - \mu_k)\\
& = \sigma \Sigma_k^{1/2} (\Sigma_k + \sigma^2 \mathbb I)^{-1/2} z_0 + \Sigma_k (\Sigma_k + \sigma^2 \mathbb I)^{-1}  (y - \mu_k)
\end{align}
$$
Recall $z_t = x_t - t\mu_k$, so $z_0 = x_0$ and $z_1 = x_1 - \mu_k$
$$
\begin{align}
x_1 = \mu_k + \sigma \Sigma_k^{1/2} (\Sigma_k + \sigma^2 \mathbb I)^{-1/2} x_0 + \Sigma_k (\Sigma_k + \sigma^2 \mathbb I)^{-1}  (y-\mu_k)
\end{align}
$$
**3) Moments of $x_1$.** Recall we had $x_0 \sim \mathcal N(0, \mathbb I)$ and $y \sim \mathcal N(\mu, \Sigma + \sigma^2 \mathbb I)$, which are drawn independently of each other. Since it's a linear combination of two Gaussians, $x_1$ itself is a Gaussian. The interpretation of $x_1$ is that we'll set it to $X^{(k+1)} = \mathcal N(\mu_{k+1}, \Sigma_{k+1})$. We want to find the update rule for $\mu_{k}, \Sigma_k$

The mean is
$$
\begin{align}
\mu_{k+1} & := \mathbb E[x_1] \\
\Aboxed{\mu_{k+1} &= \mu_k  + \Sigma_k (\Sigma_k + \sigma^2 \mathbb I)^{-1}  (\mu-\mu_k)}
\end{align}
$$
Let $B_k = \Sigma_k (\Sigma_k + \sigma^2 \mathbb I)^{-1}$. The covariance is
$$
\begin{align}
\Sigma_{k+1} & := \text{Var}(x_1)\\
& = \sigma^2 \text{Var}(B_k^{1/2} x_0) + \text{Var}(B_k y ) \\
& = \sigma^2 B_k + B_k (\Sigma + \sigma^2 \mathbb I) B_k
\end{align}
$$
Let $R_k = \Sigma_{k} - \Sigma$,
$$
\begin{align}
R_{k+1} & = \Sigma_{k+1} - \Sigma\\
& = \sigma^2 B_k + B_k(\Sigma_k - R_k + \sigma^2 \mathbb I) B_k - \Sigma\\
& = \sigma^2 B_k - B_k R_k B_k + B_k(\Sigma_k + \sigma^2 \mathbb I)B_k - \Sigma\\
& = \sigma^2 B_k - B_k R_k B_k + \Sigma_k B_k - \Sigma\\
& = (\Sigma_k + \sigma^2 \mathbb I)B_k - \Sigma -  B_k R_k B_k\\
\Aboxed{R_{k+1} & = R_k - B_k R_k B_k}
\end{align}
$$
Where we've defined $B_k = \Sigma_k (\Sigma_k + \sigma^2 \mathbb I)^{-1}$

**4) Convergence Rate** We can reuse the result from the original paper to find the convergence rate. Let $\eta = \min (\lambda_{min}(\Sigma), \lambda_{min}(\Sigma_0))$ and assume $\eta > 0$, then
$$
\begin{align}
\boxed{\|\Sigma -\Sigma_k \| \leq \left[ 1 - \left(\frac{\eta}{\sigma^2+\eta} \right)^{2}\right]^k \|\Sigma - \Sigma_0\|}
\end{align}
$$
This corresponds to $\epsilon = 1$ in the original paper. Wow there's a change in the convergence!
