#stochasticinterpolants 

Consider the MRA channel
$$
\begin{align}
F(x) = g\cdot x + w
\end{align}
$$
where $x \sim \pi$ (clean data), $g$ is a translation sampled from Haar measure, and $w \sim \mathcal N(0,\sigma^2 \mathbb I)$

I assume you have access to $\pi$, and want to sample $F(x) | x$, we can write the interpolant
$$
\begin{align}
I_t | x = \alpha_t z + \beta_t F(x) = \alpha_t z + \beta_t (g\cdot x + w)
\end{align}
$$

## Analytic Results
Consider the distribution on
$$
\begin{align}
I_t | x, g \sim \mathcal N \Big(\beta_t\, g\cdot x, (\alpha_t^2 + \beta_t^2 \sigma^2) \mathbb I \Big)
\end{align}
$$
Marginalizing over $g$ leaves us with
$$
\begin{align}
p_t(I_t | x) = \frac{1}{|G|} \sum_{g \in G} \mathcal N(I_t; \beta_t g\cdot x, s_t^2 \mathbb I), \qquad s_t^2 := \alpha_t^2 + \beta_t^2 \sigma^2
\end{align}
$$
## Drift Field
From Claude, so double check...
$$
\begin{align}
b_t(y | x) & = \mathbb E[\dot I_t | I_t = y, x]\\
& = b_t(y  |x) = \mathbb E_{g|y,x} \mathbb E[\dot I_t | I_t = y, x, g]
\end{align}
$$
$$
\begin{align}
\mathbb E[z | I_t = y,x,g] & = \frac{\alpha_t}{s_t^2}(y - \beta_t g \cdot x)\\
\mathbb E[g \cdot x + w | I_t = y,x,g] & = g \cdot x + \frac{\beta_t\sigma^2}{s_t^2} (y - \beta_t g \cdot x)
\end{align}
$$
Plugging into
$$
\begin{align}
\mathbb E[\dot I_t | y,x,g] =  \dot \beta_t g \cdot x + \frac{\dot \alpha_t \alpha + \dot \beta_t \beta_t \sigma^2}{s_t^2} (y - \beta_t g \cdot x)
\end{align}
$$
Let $\lambda_t := (\dot \alpha_t \alpha_t + \dot \beta_t \beta_t \sigma^2)/s_t^2$ and $\mu_t := \dot \beta_t - \lambda_t \beta_t$. Then
$$
\begin{align}
\mathbb E[\dot I_t | y,x,g] & = \lambda_t y + \mu_t \, g\cdot x\\
b_t(y | x) = \mathbb E_{g | y,x} \mathbb E[\dot I_t | y,x,g] & = \lambda_t y + \mu_t \sum_{g \in G} p_t(g | y,x) \, g\cdot x
\end{align}
$$
where $p_t(g|y,x)$ is given by $p(\theta | D) = p(D | \theta)p(\theta)$
$$
\begin{align}
p_t(g|y,x) & = \frac{p_t(y | g, x) p(g|x)}{\int p_t(y | g,x) p(g|x) \, dg}\\
& = \frac{\mathcal N(y; \ \beta_t g\cdot x , \ s_t^2 \mathbb I)}{\sum_{g' \in G} \mathcal N(y; \ \beta_t g' \cdot x , \ s_t^2 \mathbb I)} 
\end{align}
$$
In the 2nd move, we used $p_t(I_t | g, x)  = p_t(y | g,x) = \mathcal N(...)$, and $p(g|x) = 1/|G|$.

Rewritting this one more time, in a more ML flavor
$$
\begin{align}
b_t(y|x) = \lambda_t y + \mu_t (p_t \star x), \qquad p_t = \text{softmax}_g \left[\frac{\beta_t}{s_t^2}(y \star x) \right]
\end{align}
$$
where $y \star x$ is circular cross correlation.

## Neural Network Architecture
Input embeddings
- $Y_0 = \text{CircConv}_{k\times k}(y) \in \mathbb R^{C \times H \times W}$
- $X_0 = \text{CircConv}_{k'\times k'} (x) \in \mathbb R^{C' \times H \times W}$. Keep $x,y$ separate, don't concatenate at input.
- Embed Time: $\tau = \text{MLP}(\text{SinusoidalEmbed}(t)) \in \mathbb R^{d_{time}}$
where $\text{CircConv}_{k \times k}(x) =$ `nn.Conv2d(C_in, C_out, kernel_size=k, padding=k//2, padding_mode='circular')(x)` where $x \in \mathbb R^{C_{in} \times H \times W}$ and output is $\mathbb R^{C_{out} \times H \times W}$.





Output head
$$
\begin{align}
b_t^\theta(y | x) = \gamma(\tau)  \, y + \text{CircConv}_{1\times 1}(\delta (\tau) \odot A) + \epsilon(\tau)
\end{align}
$$where $\gamma(\tau) = \text{MLP}(\tau) \in \mathbb R$ , while $\delta,\epsilon : \mathbb R^{d_{time}} \to  \mathbb R^{\text{dim}(y)}$ are small MLPs as well.