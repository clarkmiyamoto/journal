#hemcee 

Here I redo the analysis in [[2025-01-13 Autocorrelation of HMCs]] but with leapfrog dynamics, not the continuous time Hamiltonian dynamics.

I'll try it for the regular leapfrog dynamics, and if it yields something interesting I'll try the affine invariant leapfrog.

I will analyze this on a target distribution $\pi = \mathcal N(0, \Sigma)$

# Leapfrog Dynamics
The leapfrog dynamics are written as
$$
\begin{align}
p^{(\ell + \frac 1 2)} & \leftarrow p^{(\ell)} - \frac{h}{2} \nabla V(x^{(\ell)})\\
x^{(\ell + 1)} & \leftarrow x^{(\ell)} + h M^{-1} p^{(\ell + \frac 1 2)}\\
p^{(\ell + 1)} & \leftarrow p^{(\ell + \frac 1 2)} - \frac h 2 \nabla V(x^{(\ell + 1)})
\end{align}
$$
Plugging $p^{(\ell + \frac 1 2)}$ into $x^{(\ell + 1)}$ and $p^{(\ell + 1)}$, and evaluating the gradient for $\pi \propto \exp(-V(x))$ yields
$$
\begin{align}
x^{(\ell + 1)} & = x^{(\ell)} + h M^{-1} p^{(\ell)} - \frac{h^2}{2} M^{-1} \nabla V(x^{(\ell)})\\
& = h M^{-1} p^{(\ell)} + \left(\mathbb I - \frac{h^2}{2}M^{-1} \Sigma^{-1} \right) x^{(\ell)}\\
p^{(\ell + 1)} & = p^{(\ell)} - \frac h 2 \nabla V(x^{(\ell)}) - \frac{h}{2} \nabla V(x^{(\ell + 1)})\\
& = p^{(\ell)} - \frac h 2 \Sigma^{-1} x^{(\ell)} - \frac h 2 \Sigma^{-1}\left( h M^{-1} p^{(\ell)} + \left(\mathbb I - \frac{h^2}{2}M^{-1} \Sigma^{-1} \right) x^{(\ell)}\right)\\
& = \left(\mathbb I - \frac{h^{2}}{2} \Sigma^{-1} M^{-1}  \right) p^{(\ell)} +  h  \left( - \mathbb I  + \frac{h^2}{4} \Sigma^{-1} M^{-1}  \right) \Sigma^{-1} x^{(\ell)}\\
\begin{pmatrix}  x^{(\ell + 1)}\\ p^{(\ell + 1)}  \end{pmatrix} & = \begin{pmatrix} \mathbb I - \frac{h^2}{2} M^{-1} \Sigma^{-1}& h M^{-1} \\ h  \left( - \mathbb I  + \frac{h^2}{4} \Sigma^{-1} M^{-1}  \right) \Sigma^{-1}  &  \mathbb I - \frac{h^{2}}{2} \Sigma^{-1} M^{-1}   \end{pmatrix} \begin{pmatrix} x^{(\ell )}\\ p^{(\ell)}   \end{pmatrix}
\end{align}
$$
I'll denote time evolution matrix as $A$, to be specific $\begin{pmatrix}  x^{(n)}\\ p^{(n)}  \end{pmatrix} = A^n \begin{pmatrix}  x^{(0)}\\ p^{(0)}  \end{pmatrix}$. Our objective will to be a closed form solution for $x^{(n)}$.

## Solving Leapfrog Dynamics
Our objective is to find $x_i^{(n)}$, as a function $x^{(0)}_i, p^{(0)}_i$ and $n$.

For simplicity, assume $M = \text{diag}(m_1, ..., m_d)$ and $\Sigma = \text{diag}(\sigma^2_1, ..., \sigma^2_d)$, then the dynamics decouple. Let $x^{(\ell)}_i,  p^{(\ell)}_i$ be the position & momentum in the $i$'th dimension. Then
$$
\begin{pmatrix} x^{(\ell +1)}_i\\ p^{(\ell)}_i   \end{pmatrix} = \underbrace{\begin{pmatrix} 1 - \frac{h^2 \omega_i^2}{2}& \frac{h}{m_i} \\ h  \left( - 1  + \frac{h^2 \omega^2_i}{4}  \right) \sigma_i^{-2}  &  1 - \frac{h^{2} \omega_i^2}{2}   \end{pmatrix}}_{\equiv A_i} \begin{pmatrix} x^{(\ell )}\\ p^{(\ell)}   \end{pmatrix}
$$
where $\omega_i^2 = 1/m_i \sigma_i^2$. As a sanity check, the time evolution is symplectic, that is $\det A_i = 1$.

Now introduce the variable $\phi \in [0,\pi]$ s.t. $\cos(\phi_i) = 1- \frac{h^2 \omega_i^2}{2}, \sin(\phi_i) =h \, \omega_i \sqrt{1- \frac{h^2 \omega_i^2}{4}}$. The time increment matrix becomes
$$
A_i = \begin{pmatrix}
\cos(\phi_i)                        & \frac{\hat \psi_i}{\omega_i \, m_i} \sin (\phi_i) \\ 
-\frac{1}{\hat \psi_i \, \omega_i \, \sigma_i^2  } \sin(\phi_i) & \cos(\phi_i)
\end{pmatrix}
$$
where $\psi_i = \frac{1}{\sqrt{1- h^2 \omega^2 /4}}$. Then the $n$'th power becomes

$$
A_i^n = \begin{pmatrix}
\cos(n \, \phi_i) & \frac{\psi_i}{\omega_i \, m_i}\sin(n \, \phi_i)\\
-\frac{m \, \omega_i}{\psi_i } \sin(n \, \phi_i) & \cos(n \, \phi_i)
\end{pmatrix}
$$

Meaning the position $x_i^{(n)}$ for $n$ leapfrog steps is given by
$$
x_i^{(n)} = x^{(0)}_i \cos(n \, \phi_i) + p_i^{(0)}\, \frac{\psi_i}{\omega_i m_i} \sin(n \, \phi_i)
$$
# Autocorrelation Analysis
## i'th Position, First Moment
Recall the lag-1 autocorrelation 
$$
\rho^{(1)} = \frac{\mathbb E[x_i^{(n)} x_i^{(0)}] - \mathbb E[x_i^{(n)}] \mathbb E[ x_i^{(0)}]}{ \mathbb E[(x_i^{(0)})^2] - \mathbb E[x_i^{(0)}]^2}
$$
For this analysis, we'll assume the walkers are already mixed., that is $x^{(0)} \sim \mathcal N(0, \Sigma)$, where $\Sigma = \text{diag}(\sigma_1^2 , ..., \sigma_d^2)$.

By linearity $\mathbb E[x_i^{(n)}] =\mathbb E[x_i^{(0)}] = 0$. All we need to compute is $\mathbb E[x_i^{(n)} x_i^{(0)}]$
$$
\begin{align}
\frac{\mathbb E[x_i^{(n)} x_i^{(0)}]}{\sigma^2_i} = \cos(n \, \phi_i)
\end{align}
$$
Meaning the autocorrelation becomes
$$
\rho^{(1)} = \cos(n \, \phi_i)
$$
where $\phi \in [0,\pi]$ s.t. $\cos(\phi_i) = 1- \frac{h^2 \omega_i^2}{2}, \sin(\phi_i) =h \, \omega_i \sqrt{1- \frac{h^2 \omega_i^2}{4}}$.

Let's derive what the mass matrix should be for a perfect occurs at $n \phi_i = \pi / 2$
$$
\begin{align}
\cos(\phi_i) & = \cos(\pi / 2n)\\
1 - \frac{h^2 \omega_i^2}{2} & = \cos(\pi / 2n)\\
\omega_i & = \sqrt{\frac{2}{h^2}(1 - \cos(\pi / 2n))}\\
\frac{1}{\sigma_i \sqrt{m_i}} & = \sqrt{\frac{2}{h^2}(1 - \cos(\pi / 2n))}\\
\Aboxed{m_i & = \frac{1}{\sigma_i^2} \, \left(\frac{h}{2 (1 - \cos(\pi / 2 n))} \right)}
\end{align}
$$
In the continuous time dynamics, you just set $m_i = \sigma^{-2}_i$, but we see the leapfrog discretization provides a subtle correction.
## i'th Position, Second Moment
We now inspect the lag-1 autocorrelation for the 2nd moment

$$
\rho^{(1)} = \frac{\mathbb E[(x_i^{(n)} x_i^{(0)})^2] - \mathbb E[(x_i^{(n)})^2] \mathbb E[ (x_i^{(0)})^2]}{ \mathbb E[(x_i^{(0)})^4] - \mathbb E[(x_i^{(0)})^2]^2}
$$

Now we can compute!

$$
\begin{align}
\mathbb E[(x_i^{(n)})^2] & =  \mathbb E \left[(x^{(0)}_i)^2 \cos^2(n \, \phi_i) \right] + \mathbb E \left[(p_i^{(0)})^2\, \left(\frac{\psi_i}{\omega_i m_i} \right)^2 \sin^2(n \, \phi_i) \right]\\
& =\sigma_i^2 \cos^2 (n \, \phi_i) + \frac{\psi^2_i}{\omega_i^2 m_i} \sin^2(n \, \phi_i)\\
\mathbb E[(x_i^{(n)} x_i^{(0)})^2] & = \mathbb E \left[(x^{(0)}_i)^4 \cos^2(n \, \phi_i) \right] + \mathbb E \left[(p_i^{(0)})^2\, (x^{(0)}_i)^2 \,\left(\frac{\psi_i}{\omega_i m_i} \right)^2 \sin^2(n \, \phi_i) \right]\\
& = 3 \sigma^4 \cos^2(n \, \phi_i) + \sigma^2 \frac{\psi_i^2}{\omega_i^2 m_i} \sin^2 (n\, \phi_i)
\end{align}
$$

The autocorrelation becomes

$$
\begin{align}
\rho^{(1)} = \frac{2 \sigma^4 \cos^2(n \, \phi_i)}{2 \sigma^4} = \cos^2(n \phi_i)
\end{align}
$$

Let's derive what the mass matrix should be for a perfect occurs at $n \phi_i = \pi / 2$. The mass matrix to reduce the autocorrelation becomes
$$
\boxed{m_i  = \frac{1}{\sigma_i^2} \, \left(\frac{h}{2 (1 - \cos(\pi / 2 n))} \right)}
$$

## Radius, First Moment
Let $r = \sqrt{x_1^2 + ... + x_d^2}$ be the radius. I'll now inspect the autocorrelation of this macrostate
$$
\rho^{(1)} = \frac{\mathbb E[r^{(n)} r^{(0)}] - \mathbb E[r^{(n)}] \mathbb E[ r^{(0)}]}{ \mathbb E[(r^{(0)})^2] - \mathbb E[r^{(0)}]^2}
$$
where $r^{(n)}$ is the radius after $n$ leapfrog steps.

Recall the position in the $i$'th dimension, after $n$ leapfrog steps, is given by
$$
x_i^{(n)} = x^{(0)}_i \cos(n \, \phi_i) + p_i^{(0)}\, \frac{\psi_i}{\omega_i m_i} \sin(n \, \phi_i)
$$
Assume the walkers are mixed, so $x_i^{(0)} \sim \mathcal N(0, \sigma^2_i \mathbb I_d)$ and $p_i^{(0)} \sim N(0, m_i)$. This implies the distribution of $x_i^{(n)}$ is given by
$$
x_i^{(n)} \sim \mathcal N(0, \sigma^2 \cos^2(n \phi_i) + \frac{\psi_i^2}{\omega_i^2 m_i} \sin^2(n \phi_i))
$$

## Radius, Second Moment
Let $r = \sqrt{x_1^2 + ... + x_d^2}$ be the radius. I'll now inspect the autocorrelation of this macrostate
$$
\rho^{(1)} = \frac{\mathbb E[(r^{(n)} r^{(0)})^2] - \mathbb E[(r^{(n)})^2] \mathbb E[ (r^{(0)})^2]}{ \mathbb E[(r^{(0)})^4] - \mathbb E[(r^{(0)})^2]^2}
$$
where $r^{(n)}$ is the radius after $n$ leapfrog steps.

