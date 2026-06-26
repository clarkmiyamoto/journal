#scsi 

1. Talked with Jiequn. He had a nice result on CryoET on MNIST (full dataset).
	- CryoET is each object has $K$ projections, where the projection is $$F(X) = P\,  G \cdot X + \epsilon$$
	  where $G \in \{G_0 + n \Delta G\}_{n=1}^K$, where $G_0 \sim \text{Haar}(SO(3))$ 

2. I tried the point cloud representation $x \in \mathbb R^{N \times 3}$. 
	1. Supervised setting works
	2. Unsupervsied setting seems to suffer from collapse. But I think I need to toy around with it more before we say it doesn't work...

> Eri

> Joan: Sanity check is start at a perturbation from ground truth, and see if model and converge to GT


1. Use some principles that you don't break when using LLMs
	1. Be in control of the conversation
		1. That is always be able to understand what its telling you.
		2. 