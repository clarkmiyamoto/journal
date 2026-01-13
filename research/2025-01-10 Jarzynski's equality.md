#math #sampling #sequentialmontecarlo #bayesianinference 


Say you're trying to compute $$\mathbb E_{x \sim p}[f(x)]$$where you have oracle access to $p$. The typical example is to use important sampling. What if $p$ is highly multimodal? Then you should use annealed important sampling. That is a discrete time algorithm, so I'm wondering if there is a continuous time formulation! (Similar to DDPM and SBD).

Say you have a family of distributions $(p_t)_{t \in [0,1]}$ where $$p_t(x) = e^{-V_t(x)} / Z$$You might naively say let's use Langevin to transport the samples $$ dX_t = - \nabla V_t(x) \, dt + \sqrt{2} \, dW_t
$$ where $W_t$ is a Wiener process. The corresponding Fokker Planck equation yields
$$\partial_t p_t = \nabla \cdot (\nabla V_t(x) \, p_t) + \nabla^2 p_t$$
The RHS trivially is zero. But the LHS is not always zero
$$
\begin{align}
\partial_t \log p_t & = - \partial_t V_t(x) + \frac{1}{Z} \int \partial_t V_t(x) e^{-V_t(x)}\, dx\\
\implies \partial_t p_t& = p_t(x) \left( -\partial_t V(x)  + \mathbb E_{x \sim p_t}[\partial_t V_t(x)]\right)
\end{align}
$$
Meaning using this SDE, you will find $\text{Law}(X_t) \neq p_t$. Jarzinsky's trick is to construct an ODE who's path can be used to get the correct marignal

## Theorem
Let $(X_t, A_t)$ be the solution of the SDE system $$
\begin{align}
dX_t & =  - \nabla V_t(X_t) \, dt + \sqrt{2} \, dW_t , & X_0 \sim p_0\\ 
dA_t & = (- \dot V_t(X_t) - \dot F_t(X_t)) A_t \, dt, & A_0 = 1 
\end{align}
$$ where $F_t \equiv \log Z_t$ (free energy of the system). Then for all test function $\varphi$
$$
\mathbb E[\varphi(X_t) W_t] = \int \varphi(x) \frac{e^{-V_t(x)}}{Z_t} \, dx
$$

## Proof
Usually you prove this using Feynman-Kac. Eric Vanden-Eijnden came up with this simpler proof, showed it to Simon Coste, who then posted it to his blog https://scoste.fr/posts/jarzynski/, which I am now writing about.

So first let's inspect the $A_t$ ODE
$$
\begin{align}
A_t = \exp \left(- \int (\dot V_t(X_t) - \dot F_t(X_t)) \, dt  \right)
\end{align}
$$ 


