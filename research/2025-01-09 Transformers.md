#architecture 

I'm trying to figure out what a Transformer is, because I actually have never used them.

# Attention
## Simplified Version
The attention vector $\mathbf a_i \in \mathbb R^d$  at token position $i$, is given by the matrix multiplication
$$
\begin{align}
\mathbf a_i = \sum_{j \leq i} \alpha_{ij} \mathbf x_j
\end{align}
$$
You can think of this as a linear combination of $\mathbf x_j \in \mathbb R^d$ weighted by $\alpha_{ij}$. The question is how do we choose the weighting?

What if we use a score (similarity metric), like a dot product? And then to keep $\mathbf a_i \sim \mathcal O(1)$ we can use softmax
$$
\begin{align}
\text{score}(\mathbf x_i, \mathbf x_j) & = \mathbf x_i \cdot \mathbf x_j\\
\alpha_{ij} & = \text{softmax}(\text{score}(\mathbf x_i, \mathbf x_j))
\end{align}
$$
This is just to spiritually prepare you for the read attention head.

## Single Attention Head
Here is the attention head used in actual transformers (head refers to specific structured layers in the transformer). 

The idea is to make a layer which finds which inputs (different entries in a markov chain sequence $\{\mathbf x_i\}_i$) are of most importance to each other. The attention head attempts to restructure the input $\mathbf x_i \in \mathbb R^d$ into three different embeddings
- Query: The current position
- Key: What each entry offers about it's input.
- Value: The actual content being retrieved

To capture these three roles, we'll notate them as $W^Q, W^K, W^V \in \mathbb R^{d \times d_k}$. These will project an input $\mathbf x_i$ into its representation of a query, key, or value. 
$$\mathbf q_i = \mathbf x_i W^Q , \ \mathbf k_i =\mathbf x_i  W^K  , \ \mathbf v_i = \mathbf x_i W^V $$
We want capture the similarity between a current element "query", and its previous element "key". So the "score" will be a dot product between the query and value vector. To prevent things from numerically blowing up, we'll normalize by the $\sqrt{d_k}$ dimension of the query & key vector $\mathbf q_i \in \mathbb R^{d_k}$ 
$$
\begin{align}
\text{score}(\mathbf x_i , \mathbf x_j) & = \frac{\mathbf q_i \cdot \mathbf k_j}{\sqrt{d_k}} \\
\alpha_{ij} & = \text{softmax}(\text{score}(\mathbf x_i, \mathbf x_j))\\
\textbf{head}_i & = \sum_{j \leq i} \alpha_{ij} \mathbf v_j\\
\mathbf a_i & = \textbf{head}_i W^O
\end{align}
$$
The input $\mathbf x_i$ and output $\mathbf a_i$ both live in $\mathbf R^d$ (this is called the model dimensionality). So the output weight $W^{O} \in \mathbb R^{d_k \times d}$.

Justifying the naming
- Query: What position $i$ is asking for. It gets compared to every key for reference.
- Key: What position $j$ "advertises" about it self.
- Value: what position $j$ actually contributes to the output (as it bypassse the softmax).

## Multiheaded Attention
Say you have $A$ sequences of attention, each weight matrix (query, key, value, output) will become different (per attention block)
$$
\begin{align}
\mathbf a_i & = (\textbf{head}^1_i \oplus ... \oplus \textbf{head}^A_i) W^{O}\\
\text{MultiHeadedAttention}(\mathbf x_i, [\mathbf x_{1}, ..., \mathbf x_{i-1}]) & = \mathbf a_i
\end{align}
$$ where $\oplus$ operator is concatenation.

# Transformer Block

## Layer Norm

This just means normalize the input $\text{LayerNorm} : \mathbb R^d  \to \mathbb R^d$ 
$$
\text{LayerNorm}(\mathbf x) = \gamma (\mathbf x - \mu ) / \sigma + \beta
$$
where $\gamma, \beta$ are learned. $\mu = \frac{1}{d} \sum_{i=1}^d [\mathbf x]_i$ is the mean over the entries in the vector, and $\sigma = \sqrt{\frac{1}{d}\sum_{i=1}^d ([\mathbf x]_i - \mu)^2}$  is the std over the entries in the vector.

## Transformer
The full transformer block ends up being an input of $\mathbf x_i$
$$
\begin{align}
\mathbf t_i^1 & = \text{LayerNorm}(\mathbf x_i)\\
\mathbf t_i^2 & = \text{MultiHeadedAttention}(\mathbf x_i, [\mathbf x_1, ..., \mathbf x_{i-1}])\\
\mathbf t_i^3 & = \mathbf t^2_i + \mathbf x_i\\
\mathbf t_i^4 & = \text{LayerNorm}(\mathbf t_i^3)\\
\mathbf t_i^5 & = \text{MLP}(\mathbf t_i^4)\\
\mathbf h_i & = \mathbf t_i^5 + \mathbf t_i^3
\end{align}
$$
and getting an output of $\mathbf h_i$.

Note there are residual connection across the layer norm + attention, and then across the layer norm + MLP.

# Tricks for Training
https://rbcborealis.com/research-blogs/tutorial-17-transformers-iii-training/

So apparently, training Transformers from scratch is difficult. The people who made it gave some recommendations
1. The transformer block has the resdiual connections + layer norm
2. Learning rate warm up
3. Use adaptive optimizer (ADAM) w/ large batch size $\geq 1000$

## Why?
There's a couple reasons
1. Gradient shrinkage through layer norm
	$$\frac{\partial}{\partial \mathbf x} \text{LayerNorm}(\mathbf x) = \mathcal O\left( \frac{\sqrt{D}}{\|\mathbf x\|} \right)$$
	where $\mathbf x \in \mathbb R^D$. Meaning as you compose Transformer blocks, (where $\|\mathbf x \| \gg \sqrt{D}$) you'll slowly loose all gradient information. This means your model's output will depend a lot more on later layers (closer to output), than earlier layers (closer to input).
2. ADAM is high (or unbounded) variance at the beginning of training

Here's a good figure from the website that shows how the problem cascade.
![[2025-01-09 problems.png]]