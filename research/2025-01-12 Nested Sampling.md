#bayesianinference #sampling #evidence

You are tasked w/ computing the model evidence (or marginal likelihood)
$$
Z = \int p(\mathcal D | \theta) \, p(\theta) \, d\theta
$$
where $p(\mathcal D | \theta)$ is the likelihood, and $p(\theta)$ is a prior (in a Bayesian framework). Also note this can an integral of many dimensions $\theta \in \mathbb R^d$. 

Nested sampling transforms this integral into an integral of one dimension
$$
X(L_{min}) = \mathbb P(L(\mathcal D|\theta) > L_{min}) = \int_{L(\mathcal D | \theta) > L_{min}} p(\theta) d\theta
$$
Then $Z$ can be written as
$$
Z = \int_0^1 L_{min}(X)\, dX
$$
where $L_{min}(X)$ is the inverse of $X(L_{min})$. Because it's a CDF, it's monotonic, and thus invertible.

*Proof:*
	We want to prove
	$$
	Z = \int L(\theta) \, p(\theta) \, d\theta = \int_0^1 Y(X) \, dX
	$$
	 where $Y(\cdot)$ is the inverse of $X(\lambda) = \int_{L(\theta) > \lambda} p(\theta) \, d\theta$, that is $Y(X(\lambda)) = \lambda$.
	 First notice that $$
	\begin{align}
	X(\lambda) & = \int_{L(\theta) > \lambda} p(\theta) d\theta\\
	& = \int \Theta(L(\theta) - \lambda) \, p(\theta)\, d\theta
	\end{align}
	$$ where $\Theta$ is the Heaviside function. You can also note that $\frac{d}{dx} \Theta(x) =  \delta(x)$, where $\delta$ is the dirac delta function.  This means you can write
	$$
	\begin{align}
	\frac{dX}{d\lambda} & = - \int \delta(L(\theta) - \lambda) p(\theta) \, d\theta\
	\end{align}
	$$
	$$
	\begin{align}
	\int_0^1Y\left( X(\lambda) \right) \,dX & = \int_{Y(0)}^{Y(1)} Y \left( X(\lambda)  \right) \frac{dX}{d\lambda} \, d\lambda 
	\end{align}
	$$
	I'll note the $X(-\infty) = 1$ because $L$ will always be greater than negative infinity, and $X(+\infty) = 0$ because $L$ will never be greater than positive infinity. This leads to the bounds
	$$
	\begin{align}
	& = \int_{+\infty}^{-\infty} \lambda \frac{dX}{dY} \, d\lambda  & \text{Def of inverse}\\
	& = -\int_{+\infty}^{-\infty} \lambda \int  \delta(L(\theta) - \lambda) p(\theta) \, d\theta \, d\lambda\\
	& = \int^{+\infty}_{-\infty} L(\theta) \, p(\theta) \, d\theta
	\end{align}
	$$
	 QED.