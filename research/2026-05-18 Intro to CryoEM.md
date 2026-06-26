#scsi #cryoem

References
- (AM26) [A primer on machine learning in cryo-electron microscopy (cryo-EM)](https://www.owlposting.com/p/a-primer-on-ml-in-cryo-electron-microscopy)
- [Getting started in CryoEM](youtube.com/watch?v=gDgFbAqdM_c&list=PL8_xPU5epJdctoHdQjpfHmd_z9WvGxK8-&ab_channel=caltech)


# Notes
## (AM26) [A primer on machine learning in cryo-electron microscopy (cryo-EM)](https://www.owlposting.com/p/a-primer-on-ml-in-cryo-electron-microscopy)
CryoEM is a method to understand the 3D structure of extremely small structure (e.g. proteins, small molecules, etc.). It shares this categorization with two other methods of note
- X-Ray Crystallography
- NMR Imaging
Let's briefly go over those

**X-Ray Crystallography**
X-rays are sent to a crystal, where they deflect due interactions with electrons (Compton scattering), producing a diffraction pattern. You then ask the question, can I infer the crystal structure from the diffractions.

It has a resolution (often) below $1 \mathring{\mathrm{A}}$. 

However you need to crystalize the proteins. That is they need to construct to be frozen into a crystal structure. You can imagine for large protein complexes this would be quite difficult.

**NMR spectroscopy**
Recall the Zeeman effect, that is atoms in a magnetic field receive a perturbative correction. The spectral line splits (that is in a graph $\text{Energy}$ vs $\text{Magnetic Field Strength}$, calculate this function w/ perturbation theory). By controlling the magnetic field, and sending timed pulses of EM fields (usually at radio-frequency), you can measure how atoms interact with each other.

It is limited to small proteins.

**Why CryoEM is better**
You flash-freeze a thin layer of vitreous ice. Now you can study massive protein complexes. Due to advances in electron diffraction, we can now achive resolution at $3 \mathring{\mathrm{A}}$ (and some cases $1 \mathring{\mathrm{A}}$).

The problem is CryoEM is not easy to perform. Apart from running the experimental apparatus, the data is challenging.

**CryoEM Experimental Workflow**
1. Assume you have purified a solution of proteins, suspended in aqueous solution. Now squeeze the protein onto a grid.
2. Upon apply protein solution to grid, it creates a thin film across the holes. Like soap bubbles spanning a bubble wand. 
3. Most biological molecules are made up of light elements (e.g. Carbon, Nitrogen, Oxygen, Hydrogen). These elements don't strongly interact with electrons, 
4. Imaging
	1. Grid view: low magnification of entire grid. Shows us if we've got usable ice.
	2. Square view: medium magnification. Once again double checks if ice is good.
	3. Hole view (called **micrographs**): high magnification. We hope to se a even distribution of proteins, not too crowded, not too sparse.

At the end of this, you hope have a bunch of projections of molecules.

**Reconstruction**
We have been assuming the corruption channel has been
$$
\begin{align}
\mathcal F(X) = P  G  X + \sigma W
\end{align}
$$
where $P$ is a randon (esq) transform, and $G \sim \text{Haar}(\text{SO}(3))$. However the uniform sampling of rotations is a bit naive.
- For example, these particles are under the influence of gravity. So if they have non-isotropic density, then they will have a preferred direction 
- Or if the particles are packed tightly, they can influence each other.
This is called **orientation bias / preferred orientation**.

Previous methods make a guess for $X^{(k)}$, apply forward model $Y^{(k)} = \mathcal F(X^{(k)})$, check if it aligns with data, then iterate in $k$.

**Tackling Conformation Heterogeneity**
The previous methods assume configurations can be averaged into a single meaningful structure. E.g. our 2D projects are different views of slightly different 3D objects.

An method which attempts to poke at this problem is **CryoDRGN**. It's a variational autoencoder which learns a continuous generative model of 3D structure $X$ via the 2D projections $Y$ . E.g. what is the $p(X)$ which could have produced observations $\{Y_i\}_i$?

Downsides
- Must be retrained per micrograph
- Need good initialization of structure

This particular method struggles when heterogeneity is too strong (obviously to be expected for any method).

*In our methodology:* This is natively handled. Because we're learning $p(X|Y)$

**Ab initio reconstruction**
One of the previous methods was needing a good initialization. It'd be nice if we could do this w/o the need for a prior. "Ab initio" means "from first principles" which is what we'll talk about here.

Enter the Method **[CryoDRGN2](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhong_CryoDRGN2_Ab_Initio_Neural_Reconstruction_of_3D_Protein_Structures_From_ICCV_2021_paper.pdf)** (same author as first method)

We start with a random initialization (set by initialization of neural network's weights). 

**Compositional Heterogeneity**
Imagine you have a micrograph that has multiple proteins. Now you can do it.

**What's left?**
- Infer preferred orientation problem
- Infer  