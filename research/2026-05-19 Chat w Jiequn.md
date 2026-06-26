#meeting 
# Notes

Types of problems
- Homogenous problems: Assume each picture is the same
	- Classical example: Micky mouse hat
- Heterogenous problems: Assume each picture is different
	- Discrete heterogeneity, e.g. there are $K$ different proteins in your dataset. **Easier to work in this setting first**
		- Do $K$ homogenous problems in a single dataset.
		- **Reasonable toy problem**
			- Do two rings. One class of images is interlocked rings, and one is separated rings
	- Gaussian mixtures.
	- Continuous heterogeneity.

How much data do I have access to?
- They usually have thousands to millions of samples.

What kind of variation do we want to allow for clean distribution $\pi$?
- What choice of homotopy do you allow?

Next steps, get the inverse problem working in a simple setting
1. Do the discrete heterogeneity. Two rings. Any modifications should have it s.t. the projection looks very different under rotations, that is why the rings are good
2. Try to find the right homotopy / initialization
3. How does the amount of data affect the algorithm.

