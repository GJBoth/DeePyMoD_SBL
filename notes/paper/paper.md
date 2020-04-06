# Model selection using deep learning and sparse bayesian learning

## Introduction
[Model selection]

[DeepMod intro, can include any fitting]

[SBL for rigorous selection of variables]

[Baselines and Contributions]

The rest of the article is organized as follows:
  * In **section 2** we review model discovery using sparse fitting as introduced by PDE-find and how to implement such techniques using DeepMoD. We reiterate the DeepMoD as a more general framework.
  * **Section 3** reviews sparse bayesian learning (SBL) and how to include the Bayesian lasso.
  * We show the efficient implementation of SBL in deepmod in **Section 4**. We include several new results about how to initialize and pick the optimal vector.
  * In **section 5** we show the results of using SBL for model discovery. We present results on a precomputed library to show how SBL outperforms other methods and generally study the sparsity properties and also show the results when such a library is computed a la deepmod and pde find.
  * **Section 6** reviews our results and considers possible extensions.

## Model discovery: DeepMoD and PDE-find

[Lasso, sparsity etc.]

### PDE find

[Review of PDE-find]

### DeepMod

[Review on DeepMod]

## Sparse Bayesian learning

### Sparse Bayesian Learning

[Review on sparse bayesian learning]


### Bayesian Lasso

[How to add lasso to SBL]


## Including SBL in DeepMoD

[How to include SBL in the deepmod framework]

## Results

[Baselines of datasets]

### Precomputed library

[Precomputing the library and adding noise on the library]

### Dynamic library

[Using deepmod to dynamically calculate the library and pde find]

## Discussion
