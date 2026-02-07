Code up todo's in [[2026-01-16 Dimensional Scaling & Optimally Tuned Matricies]]

Reading list
- Ensemble Based Annealed Importance Sampling: https://arxiv.org/pdf/2401.15645
	- There has to be a easy generalization of this to affine invariant methods.
	- Same w/ Jarzynski's equality (continuous time AIS) which uses Langevin dynamics...
- NETS: https://arxiv.org/pdf/2410.02711
- Diffusion Models as Priors: https://proceedings.neurips.cc/paper_files/paper/2024/file/d65c4ce22241138c1784ff753d4c746c-Paper-Conference.pdf
- Bouncy Particle Sampler: https://arxiv.org/pdf/1510.02451
- Firefly Monte Carlo: https://arxiv.org/pdf/1403.5693

- Read through this: https://metaphor.ethz.ch/x/2025/fs/401-3382-25L/

_____
# 1-Month Study Plan: Mathematical Proofs in MCMC, Sampling & Generative Models

## Week 1: Foundational Convergence Theory

| Done  | Day | Topic                          | What to Learn                                                                                          | Resources                                                       |
| :---: | :-: | ------------------------------ | ------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------- |
| [ x ] |  1  | Markov Chain Fundamentals I    | Prove Perron-Frobenius theorem for finite state spaces                                                 | Levin, Peres & Wilmer, _Markov Chains and Mixing Times_ Ch. 1-2 |
| [ x ] |  2  | Markov Chain Fundamentals II   | Prove detailed balance ⟹ stationarity; show converse fails without extra conditions                    | Levin, Peres & Wilmer Ch. 1-2                                   |
|   ☐   |  3  | Ergodic Theorems I             | Prove ergodic theorem for finite irreducible aperiodic chains                                          | Meyn & Tweedie, _Markov Chains and Stochastic Stability_ Ch. 13 |
|   ☐   |  4  | Ergodic Theorems II            | Extend to general state spaces via Harris recurrence; construct stationary but non-ergodic chain       | Meyn & Tweedie Ch. 17                                           |
|   ☐   |  5  | Total Variation Convergence I  | Prove coupling inequality: ‖μPⁿ - π‖_TV ≤ P(τ_couple > n)                                              | Levin, Peres & Wilmer Ch. 4-5                                   |
|   ☐   |  6  | Total Variation Convergence II | Coupling proof of convergence for Metropolis-Hastings                                                  | Levin, Peres & Wilmer Ch. 5                                     |
|   ☐   |  7  | Geometric Ergodicity           | Prove geometric ergodicity under drift + minorization; derive mixing time for random walk on hypercube | Meyn & Tweedie Ch. 15; Levin et al. Ch. 5                       |

---

## Week 2: MCMC Correctness & Convergence Rates

|Done|Day|Topic|What to Learn|Resources|
|:-:|:-:|---|---|---|
|☐|8|MH Correctness I|Prove MH satisfies detailed balance|Tierney (1994), _Annals of Statistics_|
|☐|9|MH Correctness II|Prove reversibility sufficient but not necessary; Tierney's general state space validity proof|Tierney (1994); Robert & Casella Ch. 6|
|☐|10|Gibbs Sampler I|Prove Gibbs as MH special case with acceptance = 1; Hammersley-Clifford connection|Robert & Casella Ch. 7|
|☐|11|Gibbs Sampler II|Systematic vs random scan convergence; data augmentation preserves target (Tanner & Wong)|Tanner & Wong (1987); Liu, _Monte Carlo Strategies_ Ch. 4|
|☐|12|Spectral Methods I|Prove Cheeger's inequality: conductance ↔ spectral gap|Levin, Peres & Wilmer Ch. 13|
|☐|13|Spectral Methods II|Poincaré inequality for variance bounds|Levin, Peres & Wilmer Ch. 13|
|☐|14|Spectral Methods III|Log-Sobolev implies faster concentration; spectral gap for Glauber dynamics (high temp Ising)|Levin et al. Ch. 13; Martinelli lecture notes|

---

## Week 3: Continuous-Time & Diffusion-Based Methods

|Done|Day|Topic|What to Learn|Resources|
|:-:|:-:|---|---|---|
|☐|15|Langevin Dynamics I|Prove Fokker-Planck equation for overdamped Langevin|Pavliotis, _Stochastic Processes and Applications_ Ch. 4|
|☐|16|Langevin Dynamics II|Show stationary dist ∝ exp(-U); prove ULA bias in non-asymptotic regime|Durmus & Moulines (2017), _Annals of Applied Probability_|
|☐|17|MALA I|Prove MALA acceptance corrects discretization bias|Roberts & Tweedie (1996), _Bernoulli_|
|☐|18|MALA II|Optimal scaling: why step size ~ d^(-1/3); coupling vs Wasserstein techniques|Roberts & Rosenthal (1998), _Statistical Science_|
|☐|19|Score-Based Diffusions I|Prove forward SDE has known transition kernel (OU process)|Song et al. (2021), _ICLR_ "Score-Based Generative Modeling through SDEs"|
|☐|20|Score-Based Diffusions II|Derive score matching from KL minimization; Vincent's denoising = implicit score matching|Vincent (2011), _Neural Computation_; Yang Song's blog|
|☐|21|Score-Based Diffusions III|Prove reverse-time SDE (Anderson's theorem); why score function appears|Anderson (1982), _Stochastic Processes and Applications_|

---

## Week 4: Modern Generative Model Theory

|Done|Day|Topic|What to Learn|Resources|
|:-:|:-:|---|---|---|
|☐|22|Normalizing Flows I|Prove change of variables formula (Jacobian term); invertibility requirements|Papamakarios et al. (2021), _JMLR_ "Normalizing Flows"|
|☐|23|Normalizing Flows II|Universal approximation for flows under mild conditions|Huang et al. (2018); Papamakarios et al. (2021)|
|☐|24|VAE Theory I|Derive ELBO via Jensen's inequality; tightness conditions|Kingma & Welling (2014), _ICLR_|
|☐|25|VAE Theory II|IWAE bound is tighter (proof); reparameterization gradient is unbiased|Burda et al. (2016), _ICLR_|
|☐|26|Diffusion Guarantees I|Score estimation error ⟹ sampling error bounds|Chen et al. (2023), "Sampling is as easy as learning the score"|
|☐|27|Diffusion Guarantees II|Convergence guarantees under smoothness assumptions|De Bortoli et al. (2021), _NeurIPS_|
|☐|28|Diffusion Guarantees III|Manifold hypothesis issues; connections to optimal transport|Pidstrigach (2022); Chen et al. (2023)|

---

## Daily & Weekly Practices

### Daily (30-60 min)

|Done|Practice|
|:-:|---|
|☐|After reading each proof, close the book and reconstruct it from memory|
|☐|Identify the "crux move" — what key insight makes the proof work?|
|☐|Note which inequalities/lemmas you import vs prove yourself|

### Weekly

|Done|Week|Practice|
|:-:|:-:|---|
|☐|1|Find alternative proof of one theorem (often in different papers)|
|☐|2|Find alternative proof of one theorem|
|☐|3|Find alternative proof of one theorem|
|☐|4|Find alternative proof of one theorem|
|☐|All|Maintain a "proof patterns" document: coupling, drift conditions, spectral methods, Lyapunov functions|

---

## Key References (Full Citations)

### Textbooks

|Resource|Use For|
|---|---|
|Levin, Peres & Wilmer, _Markov Chains and Mixing Times_ (2nd ed, 2017)|Weeks 1-2: Finite chains, spectral methods, mixing times|
|Meyn & Tweedie, _Markov Chains and Stochastic Stability_ (2nd ed, 2009)|Weeks 1-2: General state space theory, drift conditions|
|Robert & Casella, _Monte Carlo Statistical Methods_ (2nd ed, 2004)|Week 2: MCMC algorithms, Gibbs sampler|
|Pavliotis, _Stochastic Processes and Applications_ (2014)|Week 3: SDEs, Fokker-Planck, Langevin|

### Key Papers

|Paper|Topic|
|---|---|
|Tierney (1994), "Markov Chains for Exploring Posterior Distributions"|MH general state space validity|
|Roberts & Tweedie (1996), "Exponential Convergence of Langevin Distributions"|MALA convergence|
|Roberts & Rosenthal (1998), "Optimal Scaling of Discrete Approx to Langevin Diffusions"|Step size scaling|
|Durmus & Moulines (2017), "Nonasymptotic convergence analysis for ULA"|ULA non-asymptotic bounds|
|Song et al. (2021), "Score-Based Generative Modeling through SDEs"|Modern diffusion framework|
|Chen et al. (2023), "Sampling is as easy as learning the score"|Diffusion sampling complexity|

### Surveys

|Survey|Use For|
|---|---|
|Vempala, "Geometric Random Walks: A Survey"|Sampling complexity, log-concave distributions|
|Papamakarios et al. (2021), "Normalizing Flows for Probabilistic Modeling"|Flows comprehensive overview|
|Wibisono, "Sampling as optimization in the space of measures"|Modern Wasserstein/optimization perspective|

---

## End-of-Month Calibration

By day 28, you should be able to:

|☐|Skill|
|:-:|---|
|☐|Prove MH correctness from scratch in ~10 minutes|
|☐|Explain why detailed balance is sufficient but not necessary for correct stationary distribution|
|☐|State and sketch proof of at least one quantitative mixing time bound|
|☐|Derive the score-based diffusion reverse-time SDE|
|☐|Identify which proof technique (coupling, spectral, drift) applies to a new sampler|
|☐|Reconstruct ELBO derivation and explain when it's tight|

---

## Notes & Progress Log

_Use this space to track insights, questions, and proof patterns you discover:_

### Proof Patterns Identified

### Questions to Revisit

### Key Insights