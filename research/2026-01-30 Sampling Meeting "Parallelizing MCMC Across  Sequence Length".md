
Presenting: Bob Carpenter

TLDR: Parallelizes mcmc in series, that is generates a sequence of length $N$ in $\mathcal O(\log N)$ time.

# Introductory Example: All Prefix Sums

Consider a sequence $S = \{a,b,c,...\}$ and you want to construct the cumulative sum
$$
f(S) = \{0, a, a+b, a+b+c,... \}
$$
The trick to represent it as a tree, where each node is an entry in the sequence. And then you propagate up using $+$. At the top of the tree you have the final entry in $f(S)$, going back down restores the intermediate values.

Stating the problem like this allows you to generalize the algorithm beyond just regular addition.
- All you need is an identity
- And the operation $a+b$ needs to associative

Some examples of problems:
- String concatenation
- Matrix multiplication

# Sequential Processing
Consider a recursion, for $f :\mathbb R^D \to \mathbb R^D$. 
$$
\mathbf s_t = f_t(\mathbf s_{t-1})
$$
with an initial state $\mathbf s_0$. This generates a sequence $\mathbf s_{1:T} \subseteq \mathbb R^D$  If $f_t$ is an affine function, then the sequence can be computed in $\mathcal O(\log T)$ time w/ $\mathcal O(T)$ processors 

**The new trick:**
Consider the transition function $f_t$ with an initial guess $\mathbf s_0$. Linearize the system around a guess for the state trajectory $\mathbf s_{1:T}^{(i)}$ , then then evaluate the resulting linear systme to obtain a new guess $\mathbf s^{(i+1)}_{1:T}$. This is equivalent to using Newton's method on an approprixately defined residual
$$
\mathbf r(\mathbf s_{1:T}) = \text{vec}([\mathbf s_1  - f_1(\mathbf s_0), ..., \mathbf s_T - f_T(\mathbf s_{T-1})])
$$
and the Newton's update is given by Taylor's expansion
$$
\mathbf s_t^{(i+1)} = f_t(\mathbf s_{t-1}^{(i)}) + J_t(\mathbf s_{t-1}^{(i+1)} - \mathbf s_{t-1}^{(i)})
$$
where $J_t = \partial _s f (\mathbf s_{t-1}^{(i)})$ . 

Some discussion notes
- In the Markov transition kernel, we fix the seeds before hand making $f$ a deterministic function.
- You can show (formally) that Newton's method (which is optimizing you to the true trajectory) convergences in at most $T$ iterations.

By rearanging terms
$$
s_i^{(i+1)} = J_t s_{t-1}^{(i+1)} + \underbrace{f_t(s_{t-1}^{(i)}) - J_t s_{t-1}^{(i)}}_{u_t}
$$
This might get very expensive if $J_t$ is a full matrix (your compute will cost $\mathcal O(D^3)$), but you can instead reduce it to  $\mathcal O(D)$ by finding a $\text{diag}(J_t)$.
