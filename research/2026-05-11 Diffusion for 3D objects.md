#scsi #stochasticinterpolants 

- https://www.sciopen.com/article/10.26599/CVM.2025.9450452. Gives overview of diffusion models in 3D data.
-  https://arxiv.org/pdf/2103.01458. Formalization of diffusion models on point clouds.
- https://holodiffusion.github.io/static/docs/holo_diffusion_fullres.pdf. Training 3d diffusion model using 2d images.
- https://www.biorxiv.org/content/10.64898/2025.12.29.696802v1. Generative foundational model for CryoEM. Learn $p(x | x')$ in voxel space, meaning $x,x' \in \mathbb R^{64^3}$.
- https://github.com/ml-struct-bio/cryodrgn. CryoDRGN
- https://arxiv.org/pdf/2307.01831. DiT-3D
- https://arxiv.org/pdf/2405.14832. Direct3d: Image to 3D generation

# Overview

Styles of diffusion models for 3D vision tasks
- Point clouds
	- Each image becomes a collection of points. Let $x_i \in \mathbb R^3$, so a point cloud (which is one object) is $\left\{ x_i \right\}_{i=1}^p$. So a dataset becomes a collection of point clouds $\mathcal D = \left\{ \left\{ x_i^{(n)} \right\}_{i=1}^p \right\}_{n=1}^N$ . To store additional information (color, material, etc.), embed this as additional $d$ parameters in point, $x_i \in \mathbb R^{3 + d}$.
- Meshes
	- Each image is stored as a graph. $G = (V, E)$, where $V \subseteq \mathcal P(\mathbb R^{3 + d})$ and $E$ are edges connecting veriticies. 
- Voxel grids
	- Promote state space to from 2d signals (pixels on 2d lattice) $\phi : \mathbb Z^2 \to \mathbb R$ to 3d signals (pixels on 3d lattice) $\phi: \mathbb Z^3 \to \mathbb R$.


# Notes: DiT-3D
https://arxiv.org/pdf/2307.01831

For each point cloud $p_i \in \mathbb R^{N \times 3}$ with $N$ points for $(x,y,z)$ coordinates, we voxelize it as input $v_i \in \mathbb R^{V \times V \times V \times 3}$

With the voxel input, you use the patichify operator $\textsf{Patchify}_p : \mathbb R^{V \times V \times V \times 3} \to \mathbb R^{L \times 3}$, where you use patch sizes $p \times p \times p$, meaning $L = (V / p)^3$. 

with size $p \times p \times p$ to construct a sequence of tokens $t \in \mathbb R^{L \times 3}$

# Notes: Direct3D
Learns conditional model $p(x | y)$, where $x \in \mathbb R^{32^3}$ and $y \in \mathbb R^{32^2}$. The $y$'s get converted into tokens by using the DINO architecture-- they pretrain  the encoder, and then leave it fixed during training the generative model.

