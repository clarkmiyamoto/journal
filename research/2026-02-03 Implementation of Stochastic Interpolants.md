#generative #stochasticinterpolants #coding 

Chatted w/ [[Eric Vanden-Eijnden]] about advisorship. Obviously to work with him, I should understand stochastic interpolant inside and out. I want to walk through the implementation aspect of [[Stochastic Interpolants]].

# Defining Interpolant
A stochastic interpolant $I_t$ is a stochastic process
$$
I_t = I(t, x_0, x_1) + \gamma_t z
$$
where
- $(x_0, x_1) \sim \nu(x_0, x_1)$ is drawn from a measure defined on a lifted space. Note it can simply be the independent measure $\nu(x_0, x_1) = \pi_0(x_0) \, \pi_1(x_1)$. 
- $z \sim \mathcal N(0, \mathbb I)$ and is independent from $x_0, x_1$. 
- The interpolant has boundary conditions $I(t=0,x_0, x_1) = x_0$ and $I(t=1, x_0, x_1) = x_1$. The noise strength $\gamma_{t=0} = \gamma_{t=1} = 0$.

We typically choose $I(t,x_0, x_1)$ to be the form
$$
I_t = \alpha_t x_0 + \beta_t x_1 + \gamma_t z
$$
where
- $\alpha_0 = \beta_1 = 1$ and $\alpha_1 = \beta_0 = 0$.

In practice you can implement
```python
from abc import ABC, abstractmethod

class Interpolant(ABC):
	"""
	Base class for defining a stochastic interpolant
	"""
	def __init__(
		self,
		base: Distribution,
		target: Distribution,
		):
		"""
		Args:
			base: Base distribution p_0 (typically noise).
			target: Target distribution p_1 (data).
		"""
		self.base = base
		self.target = target
	
	@abstractmethod
	def alpha(self, ts: Tensor) -> Tensor:
		"""
		Coefficient for base samples. α(0)=1, α(1)=0.
		Args:
			ts. Shape (batch)
		Returns:
			alpha evaluated at ts. Shape (batch).
		"""
		pass
		
	@abstractmethod
	def alpha_dot(self, ts: Tensor) -> Tensor:
		"""
		Time derivative of alpha: dα/dt.
		Args:
			ts. Shape (batch)
		Returns:
			dα/dt evaluated at ts. Shape (batch)."""
		pass
	
	@abstractmethod
	def beta(self, ts: Tensor) -> Tensor:
		"""
		Coefficient for target samples. β(0)=0, β(1)=1.
		Args:
			ts. Shape (batch)
		Returns:
			beta evaluated at ts. Shape (batch)."""
		pass
	
	@abstractmethod
	def beta_dot(self, ts: Tensor) -> Tensor:
		"""
		Time derivative of beta: dβ/dt.
		Args:
			ts. Shape (batch)
		Returns:
			dβ/dt evaluated at ts. Shape (batch).
		"""
		pass
	
	@abstractmethod
	def gamma(self, ts: Tensor) -> Tensor:
		"""
		Coefficient for noise. γ(0)=γ(1)=0.
		Args:
			ts. Shape (batch)
		Returns:
			gamma evaluated at ts. Shape (batch).
		"""
		pass

	@abstractmethod
	def gamma_dot(self, ts: Tensor) -> Tensor:
		"""
		Time derivative of gamma: dγ/dt.
		Args:
			ts. Shape (batch)
		Returns:
			dγ/dt evaluated at ts. Shape (batch).
		"""
		pass
	
	def interpolant(self, 
				    ts: Tensor, 
				    x0s: Tensor, 
				    x1s: Tensor,
				    zs: Tensor) -> Tensor:
		'''
		Evaluates the interpolant for various samples
			I|_{t,x_0,x_1,z} = alpha_t x_0 + beta_t x_1 + gamma_t z
			
		Args:
			ts. Shape (batch).
			x0s. Shape (batch, dim).
			x1s. Shape (batch, dim).
			zs. Shape (batch, dim)
		'''
		alphas = self.alpha(ts)
		betas = self.beta(ts)
		gammas = self.gamma(ts)
		
		return alphas * x0s + betas * x1s + gammas * zs
	
	def interpolant_dot(self, 
				    ts: Tensor, 
				    x0s: Tensor, 
				    x1s: Tensor,
				    zs: Tensor) -> Tensor:
		'''
		Evaluates the time-derivative of the interpolant for various samples
			\dot I|_{t,x_0,x_1,z} = \dot alpha_t x_0 + \dot beta_t x_1 + \dot gamma_t z
			
		Args:
			ts. Shape (batch).
			x0s. Shape (batch, dim).
			x1s. Shape (batch, dim).
			zs. Shape (batch, dim)
		'''
		alphas = self.alpha_dot(ts)
		betas = self.beta_dot(ts)
		gammas = self.gamma_dot(ts)
		
		return alphas * x0s + betas * x1s + gammas * zs
		
	def sample(self, batch_size) -> Tensor:
		ts = torch.rand(batch_size) # Shape (batch)
		x0s = self.base.sample(batch_size) # Shape (batch, dim)
		x1s = self.target.sample(batch_size) # Shape (batch, dim)
		zs = torch.randn_like(x0s) # Shape (batch, dim)
		
		return ts, x0s, x1s, zs
```

So for example consider the linear interpolant
$$
I_t = (1-t) x_0 + t\, x_1
$$

# Training
Via transport equation, to get $p_t = \text{Law}(I_t)$, you get
$$
\partial_t p_t = - \nabla \cdot (b_t \, p_t) \iff dX_t = b_t(X_t) \, dt
$$
where the drift field is given by
$$
b_t(x) = \mathbb E[\dot I_t | I_t = x]
$$
We will train a neural network to best approximate the target drift, I'll denote this parameterized drift as $\hat b_t(x)$.
$$
\begin{align}
\mathcal L(\hat b) & = \int_0^1  \mathbb E_{I_t}[|\hat b_t(I_t) - \dot I_t|^2]\\
& = \mathbb E_{t \sim [0,1]} \mathbb E_{(x_0, x_1) \sim \nu} \mathbb E_{z \sim \mathcal N(0,1)} \Big [ \Big|\hat b_t (\alpha_t x_0 + \beta_t x_1 + \gamma_t z ) - (\dot \alpha_t x_0 + \dot \beta_t x_1 + \dot \gamma z)  \Big|^2\Big]
\end{align}
$$
	Here's a derivation to show this gets the right answer. $$
	\begin{align}
	\int_0^1 \mathbb E_{x_t \sim I_t}[(\hat b_t(x_t) - b_t(x_t) )^2 ] dt & = \int_0^1 \mathbb E[\hat b_t(x_t)^2  + b_t(x_t)^2 - 2 \hat b_t(x_t) \cdot b_t(x_t)] \, dt\\
	& = \int_0^1 \mathbb E[\hat b_t(x_t)^2  - 2 \hat b_t(x_t) \cdot b_t(x_t)] + \text{Constant}\\
	& = \int_0^1 \mathbb E[ \hat b_t(x_t)^2 + \dot x_t^2 - 2\hat b_t(x_t) b_t(x_t) + 2 \hat b_t(x_t) \cdot \dot x_t - 2 \hat b_t(x_t) \cdot \dot x_t] + \text{Constant}\\
	& = \int_0^1 \mathbb E[(\hat b_t(x_t) - \dot x_t)^2 - 2 \, \hat b_t(x_t) b_t(x_t) + 2 \hat b_t(x_t)\cdot \dot x_t]
	\end{align}
	$$
	Notice by tower property $\mathbb E[f(X) \mathbb E[g(Y) | X]] = \mathbb E[f(X) g(Y)]$, therefore $\mathbb E[\hat b_t(x_t) \cdot \dot x_t] = \mathbb E[\hat b_t(x_t) \cdot \mathbb E[\dot x_t | x_t]] = \mathbb E[\hat b_t(x_t) \cdot  b_t(x_t)]$. So the 2nd & 3rd term cancel, leaving
	$$
	\int_0^1 \mathbb E[(\hat b_t(x_t) - b_t(x_t))^2] \, dt = \int_0^1 \mathbb E[(\hat b_t(x_t) - \dot x_t)^2] \, dt + \text{Constant}
	$$
	So the loss function has a unique minimizer which coincides with $\hat b_t(x) = b_t(x)$.

Here's an example implementation
```python
model: torch.nn.Module
interpolant: Interpolant
optimizer: torch.Optimizer
epochs: int
batch_size: int

for _ in range(epochs):
	# Remove gradients
	optimizer.zero_grad()
	
	# Sample data
	ts, x0s, x1s, zs = interpolant.sample(batch_size)
	I = interpolant.interpolant(ts, x0s, x1s, zs) # (batch_size, dim)
	dIdt = interpolant.interpolant_dot(ts, x0s, x1s, zs) # (batch_size, dim)
	
	# Estimator for loss function
	loss = ((model(I) - dIdt)**2).mean()
	loss.backwards() # Get gradients
	optimizer.step() # Propagate gradients to model
	
	# Logging, e.g. WandB
	...

print(model) # Trained velocity field!
```

# Inference
After using training to construct an approximate velocity field $\hat b_t(x)$. We can use inference via the samples level of the Fokker Planck, and then use Euler discretization
$$
X_{t+h} = X_t + h \, \hat b_t(X_t)
$$
```python
step_size: float
batch_size: int

ts = torch.arange(0, 1, step=step_size)
xt = base.sample(batch_size)
for t in ts:
	xt = xt + step_size * model(xt)

print(xt) # Final result!
```

___
Yep that's it!