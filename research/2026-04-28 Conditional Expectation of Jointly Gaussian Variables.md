#probability 

I've had to use this formula so much that I'm writing it down and proving it.

## Fact
Say $(X,Y)$ are jointly Gaussian. Meaning 
$$
\begin{align}
\begin{pmatrix} X \\ Y \end{pmatrix} \sim \mathcal N\left(\begin{pmatrix} \mu_X \\ \mu_Y \end{pmatrix}, \begin{pmatrix} \Sigma_{XX} & \Sigma_{XY} \\ \Sigma_{XY} & \Sigma_{YY} \end{pmatrix} \right)
\end{align}
$$
Then the conditional expectation is
$$
\begin{align}
\mathbb E[X | Y] = \mu_X + \Sigma_{XY} \Sigma^{-1}_{YY} (Y - \mu_Y)
\end{align}
$$

## Proof


## Fact (Generalized)
Say $(X,Y,Z)$ are jointly Gaussian

