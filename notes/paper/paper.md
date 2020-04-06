# Model selection using deep learning and sparse bayesian learning

## Introduction
[Model selection]

[DeepMod intro, can include any fitting]

[SBL for rigorous selection of variables]


In the sparse regression framework, model discovery can be separated into two mostly separate tasks: constructing the function library $\Theta$ and finding the sparse coefficient vector corresponding to problem xxx. Following this separation, we first compare the sparsity of four different algorithms on a known, precomputed library, corrupted by a set level of white noise. These algorithms are:
  1. Lasso + thresholding, a classic sparsity promoting algorithm.
  2. Sparse bayesian learning as introduced by tipping et al.
  3. A noise-robust version of SBL using Bayesian lasso, introduced by Helgoy.
  4. STridge, the sparsity algorithm introduced by Kutz et al.

We then turn to the full task of model discovery, in which the library is computed from noisy observations. We consider two different approaches of computing the library, static and dynamic. In the static case, the library is computed from observations and a sparsity pattern is extracted from this library, PDE-find style. In the dynamic case, the a representation of the data is made from which the library is computed and constrained as the optimization takes place, deepmod style. We thus compare the following six cases.
  1. Static library
     1. PDE-find
     2. SBL
     3. Noise robust SBL
  2. Dynamic library
     1. DeepMod
     2. DeepMod + SBL
     3. DeepMod + Noise robust SBL.

The main contributions of this article are two-fold:
    1. We show that sparse bayesian learning is an efficient and rigorous method of variable selection, outperforming other measures such as pde-find or lasso.
    2. We show how to include sparse bayesian learning in the deepmod framework, allowing it to benefit from high-accuracy derivatives, outperforming other algorithms when the library has to be determined from noisy data.

The rest of the article is organized as follows:
  * **Section 2** reviews related work on model disovery, with a focus on sparsity methods.
  * In **section 3** we review model discovery using sparse fitting as introduced by PDE-find and how to implement such techniques using DeepMoD. We reiterate the DeepMoD as a more general framework.
  * **Section 4** reviews sparse bayesian learning (SBL) and how to include the Bayesian lasso.
  * We show the efficient implementation of SBL in deepmod in **Section 5**. We include several new results about how to initialize and pick the optimal vector.
  * In **section 6** we show the results of using SBL for model discovery. We present results on a precomputed library to show how SBL outperforms other methods and generally study the sparsity properties and also show the results when such a library is computed a la deepmod and pde find.
  * **Section 7** reviews our results and considers possible extensions.

## Related work


## Model discovery: DeepMoD and PDE-find

[Lasso, sparsity etc.]

### PDE find

[Review of PDE-find]

### DeepMod

[Review on DeepMod]

## Sparse Bayesian learning

We define our hierarchical Bayesian model as follows,
$$
p(\mathbf{y}\mid \mathbf{w}, \beta) = \mathcal{N}(\mathbf{w} \mid \mathbf{\Phi}\mathbf{w}, \beta^{-1})
$$

$$
p(\mathbf{w} \mid \gamma, \beta) = \prod_{i=0}^{N}\mathcal{N}(w_i \mid 0, \gamma_i\beta^{-1})
$$

with an inverse gamma prior on $\beta$:

$$
p(\beta) = \frac{b^{a}}{\Gamma(a)}({\beta})^{a-1}e^{-b\beta} \quad (a >0, b >0)
$$




### Bayesian Lasso

[How to add lasso to SBL]


## Including SBL in DeepMoD

[How to include SBL in the deepmod framework]

### Picking initial vector

### Picking next vector


## Results

[Baselines of datasets]

### Precomputed library

[Precomputing the library and adding noise on the library]

### Dynamic library

[Using deepmod to dynamically calculate the library and pde find]

## Discussion
