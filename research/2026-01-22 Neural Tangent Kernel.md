#math #architecture #optimization 

A couple years ago, there was a super fundmental result when weights are initalized in a certain way, when trained using SGD, the neural network's parameter move less and less (in the high number of weights limit), and it recovers a Gaussian process.

# Neural Tangent Kernel
Consider natural gradient descent acting on a model $f_\theta(x)$ (w/ model parameter $\theta$), via some loss function $\mathcal L(\theta) = \frac{1}{N} \sum_{i=1}^N \ell(f_\theta(x_i), y_i)$ w/ batch size $N$. 
$$
\theta_{t+h} \leftarrow \theta_t - h \, \nabla_\theta \mathcal L(\theta)
$$
Taking the limit $h \to 0$, we recover an ODE for the continuous time dynamics
$$
\begin{align}
\frac{d\theta}{dt} & = - \nabla_\theta \mathcal L(\theta)\\
& = - \frac{1}{N} \sum_{i=1}^N \frac{\partial \ell}{\partial f} \,\nabla_\theta f_\theta(x_i)
\end{align}
$$
where $\theta = \theta(t)$.

We can now inspect how the neural network, itself, evolves in time
$$
\begin{align}
\frac{df_\theta}{dt}(x) = \nabla_\theta f \cdot \frac{d\theta}{dt} = - \frac{1}{N} \sum_{i=1}^N \frac{\partial \ell}{\partial f} \underbrace{(\nabla_\theta f(x))^T (\nabla_\theta f(x_i))}_{\text{Neural Tangent Kernel}}
\end{align}
$$
The underlined component is identified as the **neural tangent kernel**
$$
H(t) = H(x, x'; \theta(t)) = (\nabla_\theta f(x))^T (\nabla_\theta f(x_i))
$$
# Neural Network Gaussian Process
## Simple: 2 Layer Network
Consider a  two layer neural network
$$
f_\theta(x) = \frac{1}{\sqrt{m}} \sum_{i=1}^m a_i \sigma(\mathbf w_i^T \, \mathbf x)
$$
where the activation is bounded $\sigma : \mathbb R^d \to [-1,+1]$ (for example soft-plus). You can also do this analysis for ReLU but the proof is structured slightly differently.
# Perturbation of Initalization
You can show that the neural tangent kernel tends to not change as $m \to \infty$. In particular
$$
\| H(t) - H(0)\|_2 \leq \mathcal O\left(\frac{t N^3}{\sqrt{m}} \right)
$$
where $N$ is the batch size, $m$ is the width of the hidden layer, is the length of the continuous time optimization.


Let $\epsilon > 0$. If $m = \Omega(\epsilon^{-2} n^2 \log(n/\delta))$, then 
$$
\mathbb P(\| H(0) - H^*\| \leq \epsilon) = 1-\delta
$$


# Resources
-  https://pages.cs.wisc.edu/~yliang/cs839_spring22/documents/0301-lec11-NTK-1.pdf