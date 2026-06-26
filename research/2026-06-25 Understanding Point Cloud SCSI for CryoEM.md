**Notation**
- Let $\mathcal X$ be the space of point cloud configurations. Meaning $\mathcal X = \mathbb R^{N \times 3}$, where $N$ is the number of points in the cloud.
- Let $\mathcal Y$ be the space of observations from CryoET channel, therefore $\mathcal Y = \mathbb R^{K \times p^2}$, where $p$ is the number of pixels in the observation, $K$ is the number of observations from the CryoET channel.
- Let $\mathcal F : \mathcal X \to \mathcal Y$ denote the CryoET channel. This means $\mathcal F(X) = (P \circ R_{\theta + n \Delta \theta} \circ G \circ X + W)_{n=1}^K$, where $G$ places an 3D object at every point in the point cloud (e.g. this will place a solid ball at each point, or a 3D isotropic Gaussian), $R_{\theta + n \Delta \theta}$ is a random global rotation $\theta \sim \text{SO}(3)$ with a fixed rotation $\Delta \theta \in \text{SO}(2)$, $P$ is the CryoET projection, and $W \sim \mathcal N(0, \sigma^2 \mathbb I_{\text{dim}(\mathcal Y)})$ is additive white noise
- $\Phi_\theta(x_0|y)$ denotes the flow induced by the ODE stochastic interpolant. To be specific, I mean given the ODE $dX_t = b_t^\theta(X_t | y) dt$ with boundary condition $X_{t=0} = x_0$, I'll denote $X_{t=1} = \Phi(x_0, y)$.

**Given:**
- Query access to $\mathcal F$
- Samples of corrupted data $\mathcal D = \left\{ y^{(n)} \right\}_{n=1}^{N_{\text{data}}}$ , where $y^{(n)} = (y^{(n)}_1, ..., y^{(n)}_K) \in \mathcal Y$. The corrupted data is presumably sampled from $y^{(n)} \overset{iid}{\sim} \nu$

**Goal:**
- Sample the clean data distribution $x \sim \pi$
- Assuming no model misspecification, that is $\mathcal F_{\#} \pi = \nu$

**Method, Self Consistent Stochastic Interpolant (SCSI):**
The idea is to iteratively update the drift field in an EM-styled algorithm

1. E-Step $$
	\begin{align}
	I_t^{(k)} & = \begin{pmatrix}\alpha_t Z + \beta_t X^{(k)}\\
	Y^{(k)}
	\end{pmatrix}\\
	\tilde b_t(x|y) & = \begin{pmatrix} b_t(x|y) \\ 0\end{pmatrix} = \begin{pmatrix} \mathbb  E [\dot I_t | I_t=x, Y=y] \\ 0\end{pmatrix}\\
	\mathcal L[\hat b] & = \mathbb E_{t, (I_t, Y)} [| \hat b_t(I_t |Y ) -  \dot I_t |^2]
	\end{align}
	$$ 
2. M-Step. Using the new drift field, generate samples $$
	\begin{align}
	\forall y \in \mathcal D, \text{ generate }X^{(k+1)} := \Phi^{(k)}(Z | y)
	\end{align}
	$$ where $z \sim \mathcal N(0, \mathbb I)$.
 

