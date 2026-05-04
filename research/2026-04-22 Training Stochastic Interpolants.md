#stochasticinterpolants 

Consider the generalized diffusion interpolant
$$
\begin{align}
I_t = \alpha_t Z + \beta_t X
\end{align}
$$
where $z \sim \mathcal N(0, \mathbb I)$ and $x \sim \pi$ (target distribution).

I want to learn the ODE flow $dX_t = b_t(X_t) dt$, and the optimal drift is given by
$$
\begin{align}
b_t^{\textsf{opt}}(x) & = \mathbb E[\dot I_t | I_t =x]\\
& = \dot \alpha_t \mathbb E[Z | I_t =x] + \dot \beta_t \mathbb E[X|I_t=x]\\
& = \frac{\dot \alpha_t}{\alpha_t} \mathbb E[I_t - \beta_t X | I_t = x] + \dot \beta_t \mathbb E[X | I_t = x]\\
& = \frac{\dot \alpha_t}{\alpha_t} x + \left(\dot \beta_t -  \frac{\dot \alpha_t}{\alpha_t} \beta_t \right) \mathbb E [X| I_t = x]
\end{align}
$$
To avoid having to relearn the time-depending coefficients in-front, we should only learn the $\mathbb E[X | I_t = x]$.
$$
\begin{align}
\hat b_t^\theta(x) = \frac{\dot \alpha_t}{\alpha_t} x + \left(\dot \beta_t -  \frac{\dot \alpha_t}{\alpha_t} \beta_t \right) \hat x^\theta(x)
\end{align}
$$
So the loss function is
$$
\begin{align}
\mathbb E[X | I_t = \cdot] & = \arg \min_f \mathbb E[(f(I_t) - X)^2]\\
\implies \mathcal L(\hat x^\theta) & = \mathbb E_{t,z,x} \left[(\hat x_t^\theta(\alpha_t Z + \beta_t X ) - X)^2 \right]
\end{align}
$$

# Conditional Interpolants
Consider wanting to sample the interpolant
$$
\begin{align}
I_t |Y \overset{(d)}{=} \alpha_t Z + \beta_t X
\end{align}
$$
Basically I want to sample the corresponding $Y$ when I input a specific $X$. For example: $X$ is MNIST images; I want $Y$ to be that specific MNIST image after corruption; but this map need-not be deterministic. So you should think $X,Y \sim \gamma(x,y)$ from a joint distribution.
 
In the traditional case
$$
\begin{align}
b_t(x|y) & = \mathbb E[\dot I_t | I_t=x, Y=y]\\
& = \frac{\dot \alpha_t}{\alpha_t} x + \left( \dot \beta_t - \frac{\dot \alpha_t}{\alpha_t} \beta_t \right) \mathbb E[X | I_t =x, Y=y]
\end{align}
$$
If I will train a neural network $\hat x_t^\theta(x|y) \approx \mathbb E[X | I_t = x, Y=y]$. Yielding a loss function
$$
\begin{align}
\mathcal L(\hat x^\theta) & = \mathbb E_{t,z,\gamma(x,y)}[(\hat x_t^\theta(I_t | y) - x)^2]\\
& = \mathbb E_{t,z,\gamma(x,y)}\left[ \left(\hat x_t^\theta(\alpha_t z + \beta_t x | y) - x \right)^2 \right]
\end{align}
$$

# Self Consistent Interpolant
The objective to move from corrupted data to clean data. You only have access to $Y$ (corrupted data), you want to sample $X$ (clean data), related by a corruption channel $\mathcal F(X) = Y$. 
$$
\begin{align}
I_t^{(k+1)} & = \alpha_t \mathcal F(X) + \beta_t X\\
& = \alpha_t \mathcal F(\Phi^{(k)}(Y)) + \beta_t \Phi^{(k)}(Y)
\end{align}
$$
where $\Phi^{(k)}$ is the flow induced by the $I^{(k)}$ interpolant. The hope is this converges to a stationary point where $\Phi^{(k)}(Y) = X$. The optimal drift is 
$$
\begin{align}
b_t(x) = \mathbb E[\dot I_t | I_t=x]
\end{align}
$$
The loss function is
$$
\begin{align}
\mathcal L(\hat b) & = \mathbb E[(b_t(I_t^{(k)}) - \dot I_t^{(k)})^2]\\
& = \mathbb E_{t,y}\left\{\Big[ \hat b_t(\alpha_t \mathcal F(\Phi^{(k)}(y)) + \beta_t \Phi^{(k)}(y))-(\dot \alpha_t \mathcal F(\Phi^{(k)}(y) ) - \dot \beta_t \Phi^{(k)}(y))\Big]^2\right\}
\end{align}
$$

# Lifted Self Consistent Interpolant
Now consider the lifted version
$$
\begin{align}
I^{(k+1)}_t | \mathcal F(X) & = \alpha_t Z + \beta_t X\\
I^{(k+1)}_t | \mathcal F(\Phi^{(k)}(Z)) & = \alpha_t Z + \beta_t \Phi^{(k)}(Z)
\end{align}
$$
So the optimal drift should be
$$
\begin{align}
b_t^{(k+1)}(x|y) & = \mathbb E[\dot I_t^{(k+1)} | I_t=x , Y=y]\\
\end{align}
$$
The loss becomes
$$
\begin{align}
\mathcal L(\hat b) & = \mathbb E\Big[ \left( \hat b_t(I_t | Y_{fake}) - \dot I_t  \right)^2\Big]\\
& = \mathbb E \Big[\left( \hat b_t(\alpha_t Z + \beta_t X_{fake}  | Y_{fake}) - (\dot \alpha_t Z + \dot \beta_t X_{fake})\right)^2 \Big]
\end{align}
$$

You can also write a denoiser type of loss by expanding the expression more
$$
\begin{align}
b^{(k+1)}_t(x|y) &=\frac{\dot \alpha_t}{\alpha_t} x + \left( \dot \beta_t - \frac{\dot \alpha_t}{\alpha_t} \beta_t \right)  \mathbb E[\Phi^{(k)}(Z|Y) | I_t^{(k+1)}=x, Y=y]
\end{align}
$$
We will denote $\hat x^{(k+1)}(x|y) \approx \mathbb E[\Phi^{(k)}(Z|Y) | I_t^{(k+1)}=x, Y=y]$. And denote $Y_{\text{fake}}^{(k)} = \mathcal F \circ X_{\text{fake}}^{(k)}$ and $X_{\text{fake}}^{(k)} = \Phi^{(k)} (Z | Y)$. The loss function for this will become
$$
\begin{align}
\mathcal L(\hat x^{(k+1)}) & = \mathbb E_{t,Z,Y}\Big[ \left( \hat x^{(k+1)}(I_t^{(k+1) } | Y_{\text{fake}}^{(k)})-  X_{\text{fake}}^{(k)} \right)^2 \Big]\\
& = \mathbb E_{t,Z,Y}\Big[ \left( \hat x^{(k+1)}(\alpha_t Z + \beta_t X_{\text{fake}} | Y_{\text{fake}}^{(k)} ) - X_{\text{fake}}^{(k)}  \right)^2 \Big]\\
& = \mathbb E_{t,Z,Y}\Big[ \left( \hat x^{(k+1)}\left(\alpha_t Z + \beta_t \Phi^{(k)} (Z | Y) \; \middle| \;  \mathcal F(\Phi^{(k)}(Z|Y)) \right) - \Phi^{(k)} (Z | Y)  \right)^2 \Big]
\end{align}
$$
In pseudocode
```pseudo
\begin{algorithm}
\caption{Training - Lifted SCSI}
\begin{algorithmic}
\State Initialize base flow $\Phi^{(1)}$ (e.g., identity mapping or pretrained prior)
\For{$k = 1, 2, \dots, T_{\text{outer}}$}
    \For{$i = 1, 2, \dots, T_{\text{inner}}$}
        \State Sample batch of corrupted data $y \sim \nu$
        \State Sample noise $z \sim \mathcal N(0, \mathbb I)$
        \State Sample time $t \sim \mathcal U[0, 1]$
        
        \State Generate fake clean data: $x_{\text{fake}} = \Phi^{(k)}(z \mid y)$
        \State Simulate corruption: $y_{\text{fake}} = \mathcal F(x_{\text{fake}})$
        
        \State Construct interpolant: $I_t = \alpha_t z + \beta_t x_{\text{fake}}$
        
        \State Compute empirical loss: $\mathcal L(\theta) = \left\| \hat x_t^\theta(I_t \mid y_{\text{fake}}) - x_{\text{fake}} \right\|^2$
        \State Update parameters $\theta$ using $\nabla_\theta \mathcal L(\theta)$
    \EndFor
    \State Update flow $\Phi^{(k+1)}$ using the optimized network $\hat x_\theta$
\EndFor
\end{algorithmic}
\end{algorithm}
```

> ⚠️ I talked with Eric and this is not the only way. Using the same $z$ might lead to some sort of model collapse, seems better to instead do

```pseudo
\begin{algorithm}
\caption{Training - Lifted SCSI}
\begin{algorithmic}
\State Initialize base flow $\Phi^{(1)}$ (e.g., identity mapping or pretrained prior)
\For{$k = 1, 2, \dots, T_{\text{outer}}$}
    \For{$i = 1, 2, \dots, T_{\text{inner}}$}
        \State Sample batch of corrupted data $y \sim \nu$
        \State Sample noise $z, \textcolor{blue}{z'} \sim \mathcal N(0, \mathbb I)$
        \State Sample time $t \sim \mathcal U[0, 1]$
        
        \State Generate fake clean data: $x_{\text{fake}} = \Phi^{(k)}(z \mid y)$
        \State Simulate corruption: $y_{\text{fake}} = \mathcal F(x_{\text{fake}})$
        
        \State Construct interpolant: $I_t = \alpha_t \textcolor{blue}{z'} + \beta_t x_{\text{fake}}$
        
        \State Compute empirical loss: $\mathcal L(\theta) = \left\| \hat x_t^\theta(I_t \mid y_{\text{fake}}) - x_{\text{fake}} \right\|^2$
        \State Update parameters $\theta$ using $\nabla_\theta \mathcal L(\theta)$
    \EndFor
    \State Update flow $\Phi^{(k+1)}$ using the optimized network $\hat x_\theta$
\EndFor
\end{algorithmic}
\end{algorithm}
```
# Group Interpolants
Consider the generalized diffusion interpolant which maps to some $\mathcal G$-invariant distribution
$$
\begin{align}
I_t = \alpha_t Z + \beta_t G\cdot X
\end{align}
$$
where $G \sim \mathcal G$ (sampled uniformly via Haar measure). We can play the same game to learn the optimal drift
$$
\begin{align}
b^{\textsf{opt}}_t(x) & = \mathbb E[\dot I_t | I_t =x]\\
& = \frac{\dot \alpha_t}{\alpha_t} x + \left( \dot \beta_t - \frac{\dot \alpha_t}{\alpha_t} \beta_t \right) \mathbb E[G \cdot X | I_t =x]
\end{align}
$$
For the parameterization of $\hat x^\theta_t(x) \approx \mathbb E[G \cdot X | I_t = x]$, the loss function becomes
$$
\begin{align}
\mathcal L(\hat x^\theta) = \mathbb E_{t,z,x}\left[ \|\hat x^\theta_t(\alpha _t Z + \beta_t G \cdot X) - G \cdot X\|^2 \right]
\end{align}
$$
- If $\hat x^\theta$ is $\mathcal G$-equivariant, that is $\hat x^\theta(g \cdot x) = g \hat x^\theta(x)$
- And the base distribution is $\mathcal G$-invariant, $Z \overset{(d)}{=} G \cdot Z$  
- The norm $\| \cdot \|$ is $\mathcal  G$-invariant, $\|G \cdot X\| = \|X \|$

Then you can pull out the $G$ from the loss function
$$
\begin{align}
\mathcal L(\hat x^\theta) & = \mathbb E_{t,z,x,g} \left[ (\hat x^\theta_t(G(\alpha _t G^{-1} Z + \beta_t X)) - G \cdot X)^2 \right]\\
& = \mathbb E_{t,z,x} [(\hat x_t^\theta(\alpha_t Z + \beta_t X) - X)^2]
\end{align}
$$
Meaning you don't have to sample $\mathcal G$.
