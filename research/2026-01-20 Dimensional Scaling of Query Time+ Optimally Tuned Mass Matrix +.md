#hemcee 

[[Yifan Chen]] had a note that I need to access the acceptance probability using the leapfrog + metroplist hasting step. So there a now a couple of new todo's based on [[2026-01-16 Dimensional Scaling & Optimally Tuned Matricies]] and some feedback from [[2026-01-17 Autocorrelation of Discrete Dynamics HMCs]].

Recap of the feedback.
- In 1 dimension / statistics on the marginal, the mass matrix is equivalent to tuning the step size & integration length. 
- To get dimensional scaling, the M-H accept/reject step causes correlations yields dimensional scaling. In particular if you want the acceptance ratio remain $\mathcal O(1)$, then the step size needs to scale $h \sim d^{-1/4}$. This is a well known result, see http://stuart.caltech.edu/publications/pdf/stuart106.pdf. In contrast the stretch move has scaling $h \sim d^{-1/2}$.

A small rabbit hole I went on was the effects of the leapfrog discretization. The Hamiltonian is not exactly preserved. You can show that $H^{ODE} - H^{Leapfrog} = \epsilon^2 H_\epsilon + \mathcal O(\epsilon^3)$. 





# To Do

Check the dimensional scaling of a Gaussian
      ![[2026-01-16 dimensional scaling.png]]See this plot from Yifan's paper https://arxiv.org/pdf/2505.02987
- [ ] Do the same plot as above, but on the y-axis do autocorrelation per leapfrog step.
- [ ] Compare the performance of the auto-tuned sampler on a properly conditioned regular HMC ((1)make the mass matrix the Hessian at the extrema, (2) make the mass matrix the true covariance of the distribution), and compare it to ours.


