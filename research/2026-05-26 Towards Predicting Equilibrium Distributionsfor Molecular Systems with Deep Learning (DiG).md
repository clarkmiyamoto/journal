#scsi #generative #paper 
URL: https://arxiv.org/pdf/2306.05445

# Notes
This paper develops **Distributional Graphormer (DiG)**. A deep learning approach to *approximately* predict the equilibrium distribution & efficiently sample diverse and chemically plausible structures of molecular systems.

Given: Discriptor $\mathcal D$ of target molecular system as input (e.g. amino acid sequence)

### Generative Modeling Framework
Consider $p_{simple} = \mathcal N(0, \mathbb I)$. You can reach this distribution from any distribution by running OU 
$$
\begin{align}
dR_t = - \frac{\beta_t}{2} R_t \, dt + \sqrt{\beta_t} dB_t
\end{align}
$$
where $B_t$ is standard Brownian motion. The reverse process is given by
$$
\begin{align}
dR_{\bar t} = \left[ \frac{\beta_{\bar t}}{2} R_{\bar t} + \beta_{\bar t} \nabla \log q_{\mathcal D, \bar t}(R_{\bar t
}) \right] \, d \bar{t} + \sqrt{\beta_{\bar t}}\,  dB_{\bar t}
\end{align}
$$
where $\bar t = \tau - t$ and 

We can write the forward OU process according to Fokker Planck
$$
\begin{align}
\partial_t q_t  & = \frac{\beta_t}{2}\nabla \cdot( R \, q_t)+ \frac{\beta_t}{2} \, \Delta q_t\\
\partial_t \log q_t & =
\end{align}
$$

### Physics Informed Pre-training using PINN Loss
In physics, we know when a system is at equilibrium (at temp $T$)
$$
\begin{align}
q_{\mathcal D, 0}(R) \propto \exp \left( - \frac{E_{\mathcal D}(\mathbf R)}{k_B T} \right)
\end{align}
$$
Using arctan time units, we can say this occurs at time $t=1$. Using Fokker Planck, we can write the dynamics on the score $s_{\mathcal D, t}(R) = \nabla \log q_{\mathcal D, t}(R)$, and setup it up as a PINN loss to Emulate Langevin Dynamics

# Appendix A
### A.2.1 Coarse-Grained Representation of Protein
Let $R$ be the coarse-grained representation of the protein. With residues treated as rigid bodies, proteins are represnted by the coordinates $C \in \mathbb R^3$ of alpha-carbon atoms, and orientations $Q \in \text{SO}(3)$.

Diffusion is traditionally defined on $\mathbb R^n$, so it must be adopted to lie on the Manifold (defined by $\text{SO}(3)$).
- You can generalize this to "Riemannian Flow Matching"
Mapping back to Voxel / object space from $R,C,Q$ space. One thing to google is "smear"
### Diffusion on Special Orthogonal Group
Consider the Lie group $\text{SO}(3)$, and it's associated Lie Algebra $\mathfrak{so}(3)$. Let $q = (x,y,z) \in \mathfrak{so}(3)$
$$
\begin{align}
Q = \exp \begin{pmatrix}
0 & z & -y \\
- z & 0 & x\\
 y & -x & 0
\end{pmatrix} \in \text{SO}(3), 
\quad q = (x,y,z) \in \mathfrak{so}(3)
\end{align}
$$
The forward process is defined the generalization of the Euclidian Fokker Planck
$$
\begin{align}
\partial_t p_t = \frac{1}{2} \Delta p_t
\end{align}
$$
to the covariant Fokker Planck defined on $(M,g)$
$$
\begin{align}
\partial_t p_t = \frac{1}{2} \Delta_g p_t
\end{align}
$$
where $\Delta_g$ is the Laplacian with metric $g$. When the manifold is compact, this defines a process which equilibrates (as $t \to \infty$) to the uniform measure over the manifold. For more comments see [[2026-06-10 Stochastic Interpolants for Distributions on Manifolds]]

# Appendix C: Training Pipeline & Dataset
1. Initialization
2. Physics Informed Diffusion Pre Training
3. Data-based training using simulation data
