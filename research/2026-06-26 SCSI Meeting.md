#scsi 

- Working in CryoET setting. Finally got it to work on $\mu = \delta_x$!
	- Previously the ground truth was point sampled from the *surface* of a shape, now I'm sampling the volume of a shape.
	- There are also some more tricks to get the algorithm to work better
		- GVP interpolant $I_t = \cos(...) Z + \sin(...) X$ increases performance a lot
			- Joan had an intuition using a variance preserving interpolant will give better performance, but he wasn't sure why...
		- Use $\gamma = 0.999$. This is $\Theta_{\text{EMA}}$ 

My pseudocode code
![[2026-06-26 scsi algo.png]]
Previously, 
- I wasn't using EMA
- I was constructing a data $\mathcal D^{(k)} = \left\{ (X^{(k)}, \mathcal F(X^{(k)})) \right\}$ as the $M$ step, and then for the E-step I run multiple epochs on this dataset.