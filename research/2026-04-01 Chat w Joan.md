#meeting 

- Let's keep in mind the question
	- "What is optimal ODE or SDE vs EM" 
	- Right now we have a good answer for Gaussian
	- Now we want to do this in the context of a GMM + AWGN

Consider a data space $\mathcal X$ and a measurement space $\mathcal Y$, and we have a corruption channel $\mathcal F : \mathcal X \to \mathcal Y$.

To do this when $\mathcal X \neq \mathcal Y$, we can define an embedding $\mathcal Z = \mathcal X \otimes \mathcal Y$. From here you can lift the channel into
$$
\begin{align}
\tilde {\mathcal F}(x,y) = \begin{cases} (w, \mathcal F(x))\end{cases}
\end{align}
$$
This is one choice, (where $w \sim \text{Noise}$). You can also do another version
$$
\begin{align}
(x , F(x)) \mapsto (w, F(x))
\end{align}
$$
Before you could have thought of $w$ as some latent variable, aka you learn the marginal (not the conditional). But on the 2nd version learns $x | \mathcal F(x)$. This is the EM version!

**Joan**: It's quite natural to focus on the AWGN + GMM.

- Going back to application
	- There's a application in cryoEM called **multi reference alignment (MRA)**
	  You have $x = x(u)$ (where $u \in \text{2d grid}$), so this is an image / field. Assume boundary conditios are periodic, so the spacetime lives on a torus.
	  Your channel $(\mathcal Fx)(x) = x(u - u_0) + w(u)$, where $u_0 \sim \text{Unif}(T)$ (where $T$ is the translation group) and $w(u) \sim^{iid} \gamma$ (white noise at every point in space iid). 
	  
	  In this case, we have $\mathcal K$ is not injective.
	  
	  Heuristically, you can think of this as only being able to recover some distribution within an orbit. So imagine a bunch of iso-lines, where each line is a different photo of a MNIST digit, and along the line gives various translation of that digit.
	  
	  Consider a very simple interpolant. Assume you have access to the original distirbution $\pi$. And assume you use a linear interpolant $I_t = (1-t)x + t F(x)$, the drift becomes $\hat b_t(x) = \mathbb E[F(x) - x | I_t]$. Notice that $F(x) - x$ is a very weird image, it's like the digit with an inverted digit else where...
	  
	  But when you do the embedding (when $\mathcal F : \mathcal X \to \mathcal Y$). It seems to look a lot easier
	  $$\begin{align} I_t^x =  (1-t) x + t w \\ I_t^y = (1-y) \tilde w + t g.x +. tw\end{align}$$
	  
	  So when you're learning $x | \mathcal F(x)$ you want the following properties
	  
	  1. $f \approx b$ (where $f$ is your neural network) and $b$ is the optimal drift
	  2. $f(x | y) = f(x | g.y)$ (where $g$ is the group operator)
	  3. $g^{-1} f(x|y) = f(g.x | y)$
	  
	  This may seem a bit crazy, but we can make a neural network architecture which has this property
	  
	  Encode the $x$ into a UNet (which is equivariant), and encode the $y$'s into a ResNet (which is invariant). The $\text{ResNet}(y)$ can be treated similar to a time-embedding for a UNet / conditioner, so it'll leave the $\text{UNet}$ invariant. Wow!
	  
	  
	  
	- The forward process $\mathcal F$, which defines the integral operator $\mathcal K$ is no longer injective
	- Now $\mathcal K \pi$ is invariant to $G$ (in cryoEM $G = \text{SO}(3)$)
		- $\mathcal K \pi = \mathcal K G \pi$, where $\mathcal G \pi$ is the pushforward on the distribution (not at the samples!)

### Another Direction (not top priority)
At inference time, when you run
$$
\begin{align}
dY_\tau^{(k+1)} = \frac{1 + \epsilon_\tau}{2} \nabla \log \pi_\tau^{(k)}(Y_\tau) d\tau + \sqrt{\epsilon_\tau} dW_\tau
\end{align}
$$
You initialize from $\text{Law}((1-r) \tilde y + r y) = \tilde \mu$ and I set $\text{Law}(Y_{\tau = 0}) = \tilde \mu$.


