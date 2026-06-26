#meeting 

Talk: Qi Liu

# Notes
Setup
- Let physics be a markov transition kernel
- Ergodic, so it unique stationary measure

**Example**
- Consider markov chain $P \in \mathbb R^{d \times d}$, with invariant measure $\mu$.
- Given a long observation of states $\left\{ x_t \right\}_t$ 
- Consider estimator $\hat P^{(s)} = (1-s) \hat P + s 1^T \mu$, $s \in [0,1]$
	- Look at path KL between true and the estimated. You'll find the path KL (1) depends on the dimension of $P$, (2) and to minimize it, as $d$ gets bigger, you'll need a bigger $s$.

