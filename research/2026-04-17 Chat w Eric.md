#scsi #meeting 

I'll denote
- $I_t = \alpha_t x + \beta_t \mathcal F(x)$ as the supervised setting
- $I_t^{(k+1)} = \alpha_t \Phi_{\Theta^{(k)}}(y) + \beta_t y$ as the unsupervised (or inverse problem) setting

# Notes
On the convergence analysis of the AWGN
- The $I_t = (1-\sqrt{\alpha_t}) x + \sqrt{\alpha_t} \mathcal F(x)$ is a bad interpolant in practice because you have divergences at $t=0,1$. You can either do this analysis using $I_t = \alpha_t x + \beta_t \mathcal F(x)$; perhaps as a starting point you can do variance preserving $I_t = \cos(\pi t / 2) x + \sin(\pi t / 2) \mathcal F(x)$.

On experimental study of MRA + MNIST
- I originally thought it was architecture problems preventing convergence. Eric asked if it worked on the supervised problem, and I said yes. He believes if it works on the supervised problem, but it gets stuck/doesn't improve on the inverse problem, it must be a problem of the fixed point. So I need to play w/ the SCSI algorithm
- Method for debugging neural networks & SCSI
	1. See if it's possible to learn forward problem $I_t | X =x \overset{(d)}{=} \alpha_t z + \beta_t \mathcal F(x)$ 
		1. If it can't learn this, it's probably an architecture problem. See https://github.com/google-research/tuning_playbook
	2. Once forward is working, try the inverse problem $I_t | Y= y  
		1. If inverse problem seems to not work, yet the forward problem works, try to determine if it's a problem of the fixed point. Play with SCSI algorithm parameters (e.g. inner loop vs outer loop, let it cheat, etc.)