#meeting 

Had a brief chat w/ Florentin after my meeting w/ Joan.

I told him about the lifted SCSI problem, he stopped me before I could get to the lifting and said
```
Why are you using MRA channel: F(x) = T.x + w
If you only have injectivity conditioned on sampling a Group Orbit,
then should you just learn the AWGN channel F(x) = x + w, and then
just randomly translate at the end?
```

He also thought the MRA channel could be decomposed into a symmetric & rotation component
- https://onlinelibrary.wiley.com/doi/10.1002/cpa.3160440402
For example consider decomposing a linear operator w/ SVD
$$
\begin{align}
M & = U S V^T\\
& = (U V^T) (V S V^T)\\
& = (US U^T)(UV^T)
\end{align}
$$
So $UV^T$ is unitary, and $USU^T$ is symmetric.

Additionally he recommended to look at HuggingFace for well known diffusion architectures. As a first place to try, look at https://github.com/openai/improved-diffusion