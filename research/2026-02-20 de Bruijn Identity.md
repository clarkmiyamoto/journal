#math #informationtheory #probability
Reference: 
- https://benchugg.com/research_notes/entropy_and_clt/

In [[2025-01-09 Generative Modeling from Black-box Corruptions Self Consistent Stochastic Interpolants]], there is a nice application of the *de Bruijn Identity*.

Here's a quick exploration of it

**Fact** (KL Divergence of two stochastic process) Consider two stochastic processes $dX_t = a_t(X_t) dt + \sqrt{2 \sigma_t} dW_t$ and $dY_t = b_t(Y_t) dt + \sqrt{2 \sigma_t} dW_t$, where $\text{Law}(X_t) = \mu_t$ and $\text{Law}(Y_t) = \nu$. Then
$$
\begin{align}
\frac{d}{dt}\text{KL}(\mu_t \| \nu_t) = -\sigma I(\mu_t \| \nu_t) + \mathbb E_{\mu_t} \langle a-b, \nabla \log \mu_t - \nabla \log \nu_t\rangle
\end{align}
$$
where $I(\mu \| \nu) = \mathbb E_{\mu} [|\nabla \log \mu - \nabla \log \nu|^2]$ is the Fisher divergence.

*Proof:*  By Fokker Planck, we have
$$
\begin{align}
	\partial_t \mu_t & = \nabla \cdot [(-a_t + \sigma_t \nabla \log \mu_t)\mu_t], & \mu_{t=0} = \mu\\
	\partial_t \nu_t & = \nabla \cdot [(-b_t + \sigma_t \nabla \log \nu_t) \nu_t], & \nu_{t=0} = \nu
\end{align}
$$
Now we can compute how the KL divergence changes in time
$$
\begin{align}
\frac{d}{dt} \text{KL}(\mu_t\|\nu_t) & = \int \partial_t \mu_t \log  \frac{\mu_t}{\nu_t} + \int \mu_t\frac{\nu_t}{\mu_t} \frac{\nu_t \partial \mu_t - \mu_t \partial \nu_t}{\nu_t^2}\\
& = \int \partial_t \mu_t  \log \frac{\mu_t}{\nu_t} - \int \frac{\mu_t}{\nu_t} \, \partial_t \nu_t\\
& = \int \nabla \cdot [(-a_t + \sigma_t \nabla \log \mu_t)\mu_t] \log \frac{\mu_t}{\nu_t} - \int  \frac{\mu_t}{\nu_t} \nabla \cdot [(-b_t + \sigma_t \nabla \log \nu_t)\nu_t]\\
& = - \int \langle -a_t + \sigma_t \nabla \log \mu_t, \nabla \log \frac{\mu_t}{\nu_t} \rangle \mu_t + \int \langle \nabla  \left( \frac{\mu_t}{\nu_t} \right), -b_t + \sigma_t \nabla \log \nu_t \rangle \frac{\nu_t}{\mu_t}\mu_t\\
& = - \int [\langle -a_t + \sigma_t \nabla \log \mu_t , \nabla \log \frac{\mu_t}{\nu_t} \rangle + \int  \langle -b_t + \sigma_t \nabla \log \nu_t, \nabla \log \frac{\mu_t}{\nu_t} \rangle ] \mu_t\\
& = - \sigma_t \int \left| \nabla \log \frac{\mu_t}{\nu_t}\right|^2 \mu_t + \int \langle a_t - b_t, \nabla \log \frac{\mu_t}{\nu_t} \rangle \mu_t\\
& = - \sigma_t \, I(\mu_t \| \nu_t) + \mathbb E_{\mu_t} \langle a_t - b_t , \nabla \log \frac{\mu_t}{\nu_t} \rangle
\end{align}
$$

**Fact** (de Bruijn Identity)
$$
\frac{d}{dt} \text{KL}(\mu_t \| \nu_t) = - \sigma_t I(\mu_t \| \nu_t)
$$
*Proof:*  Take $a_t = b_t$.

A note on the original de Bruijn identity. Let $X_t = X+ \xi_t$ and $Y_t = Y + \xi_t$, where $\xi_t\sim \mathcal N(0, t^2)$ and  $X,Y \perp \xi_t$, you have
$$
\frac{d}{dt} \text{KL}(X + \xi_t \| Y + \xi_t) =- \frac{1}{2} I(X + \xi_t \| Y + \xi_t)
$$
In Euler discretization, $dX_s = b_s(X_s) ds + \sigma_s \, dW_s$ becomes $X_{s+ t} = X_t +  b_t(X_t) \, t + \sigma^2 \xi_t$, so you can see the original de Bruijn identity makes sense.