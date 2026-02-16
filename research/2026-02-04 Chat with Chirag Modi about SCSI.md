#generative #stochasticinterpolants 

Had a chat w/ [[Chirag Modi]] about his paper: [[2025-01-09 Generative Modeling from Black-box Corruptions Self Consistent Stochastic Interpolants]]. Here some summarized notes.

# Notes

He got inspiration for the project by reading: https://arxiv.org/pdf/2405.13712

Similar papers
- https://arxiv.org/pdf/2510.12691
- https://arxiv.org/pdf/2510.05205



Extra reading
- https://arxiv.org/abs/2410.05646

Applications: Hogg was interested, ask him for ideas...

The SDE is more difficult to train, perhaps there's interesting theoretical explorations to go through.
- I found a paper talking about the noise scheduling of the SDE https://arxiv.org/pdf/2502.09130

Joan and Eric said they want to apply it to **Cryo-electron microscopy**

Maybe there's some way to do reward tilting, so you can do posterior sampling. Can you actually do posterior sampling, instead of just the inverse problem.

Perhaps there's a way to improve simulation based inference, as they're similar problems... 

Solving inverse problem where you have constraints. I.e. sum of pixels = a number.

At each training step you need to make the forward model  cheap. Are there tricks we can do to make this run on cheap-approximates to expensive models.
