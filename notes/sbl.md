# Derivation of formulas for Sparse Bayesian Lasso Learning

In this document we derive all the necessary equations for the sparse bayesian learning with bayesian lasso and fast maximum likelihood optimisation.

## Model

We define our hierarchical Bayesian model as follows,
$$
p(\mathbf{y}\mid \mathbf{w}, \beta) = \mathcal{N}(\mathbf{w} \mid \mathbf{\Phi}\mathbf{w}, \beta^{-1})
$$

$$
p(\mathbf{w} \mid \gamma, \beta) = \prod_{i=0}^{N}\mathcal{N}(w_i \mid 0, \gamma_i\beta^{-1})
$$

$$
p(\mathbf{\gamma} \mid \lambda) = \prod_{i=0}^{N}\frac{\lambda}{2}e^{-\frac{\lambda\gamma_i}{2}}
$$

$$
p(\lambda) = \frac{\delta^{\nu}}{\Gamma(\nu)}\lambda^{\nu-1}e^{-\delta\lambda} \quad (\nu >0, \delta >0)
$$

$$
p(\beta) = \frac{b^{a}}{\Gamma(a)}({\beta})^{a-1}e^{-b\beta} \quad (a >0, b >0)
$$

where $\beta = \sigma^{-2}$ is the precision (note that if using $\sigma^2$, the prior should be inverse gamma and not gamma), $\mathbf{\Phi}$ the design matrix with of size $(N \times M)$, where $N$ is the number of samples in the dataset and $M$ the number of terms in the library matrix. [ TO DO: EXPLAIN EFFECT OF EVERY TERM]

We can obtain the conditional prior distribution by integrating out the hyperparameters $\mathbf{\gamma}$ [TO DO: CHECK CALCULATION]:
$$
p(\mathbf{w} \mid \beta) = \prod_{i=0}^{N}\frac{\sqrt{\beta\lambda}}{2}e^{-\sqrt{\beta\lambda}|w_i|}
$$
which is the conditional prior corresponding to the Bayesian Lasso. 

### Posterior

The posterior of the model, i.e. $p(\mathbf{w}, \mathbf{\gamma}, \sigma^2, \lambda \mid \mathbf{y})$ is undetermined so we decompose like Tipping as follows:
$$
p(\mathbf{w}, \mathbf{\gamma}, \beta, \lambda \mid \mathbf{y}) = p(\mathbf{w}\mid \mathbf{y}, \mathbf{\gamma}, \beta, \lambda)p(\mathbf{\gamma}, \beta, \lambda \mid \mathbf{y})
$$
The first term on the right is a convolution of two Gaussians and is hence also a Gaussian with mean and covariance
$$
\mathbf{\Sigma} = [\beta\mathbf{\Phi}^T\mathbf{\Phi} + \Lambda^{-1}]^{-1}
$$

$$
\mu = \beta\mathbf{\Sigma}\mathbf{\Phi}^T\mathbf{y}
$$

where $\Lambda = \text{diag}(\mathbf{\gamma}\beta^{-1})$. We can factor out the noise and write things slightly easier:
$$
\mathbf{\Sigma} = \beta^{-1}[\mathbf{\Phi}^T\mathbf{\Phi} + \text{diag}(\mathbf{\gamma}^{-1})]^{-1} = \beta^{-1}\hat{\mathbf{\Sigma}}
$$

$$
\mu = \hat{\mathbf{\Sigma}}\mathbf{\Phi}^T\mathbf{y}
$$

which is independent from the noise [TO DO: WHY WHAT DOES THAT MEAN?]. We estimate the term on the right by type-II maximum likelihood optimisation, where we replace $\lambda$ and $\beta$ by their point estimates. Note that by Bayes' rule, $p(\mathbf{\gamma}, \beta, \lambda \mid \mathbf{y}) = p(\mathbf{y}, \mathbf{\gamma}, \beta, \lambda)p(\mathbf{y}) \propto p(\mathbf{y}, \mathbf{\gamma}, \beta, \lambda)$, so we can find their point estimates by optimizing over the joint distribution. The joint distribution can be found by integrating out $\mathbf{w}$
$$
p(\mathbf{y}, \mathbf{\gamma}, \beta, \lambda) = \int p(\mathbf{y}\mid \mathbf{w}, \beta)p(\mathbf{w}\mid \mathbf{\gamma})p(\mathbf{\gamma}\mid\lambda)p(\lambda)p(\beta)d\mathbf{w}
$$
where the last three terms are independent of $\mathbf{w}$, so that the integral is a convolution of the first two Gaussians leading to another Gaussian, finally giving the joint distribution as
$$
p(\mathbf{y}, \mathbf{\gamma}, \beta, \lambda) =\left(\frac{1}{2\pi}\right)^{N/2}|\mathbf{C}|^{-1/2}e^{-\frac{1}{2}\mathbf{y}^T\mathbf{C}^{-1}\mathbf{y}}p(\mathbf{\gamma}\mid\lambda)p(\lambda)p(\beta)
$$


with the covariance $C$ defined as
$$
C = \beta^{-1}\mathbf{I}_N + \mathbf{\Phi}\mathbf{\Lambda}\mathbf{\Phi}^T = \beta^{-1}(\mathbf{I}_N + \mathbf{\Phi}\text{diag}(\mathbf{\gamma})\mathbf{\Phi}^T) = \beta^{-1}\hat{C}
$$
Working with probabilities leads to small numbers, so it's numerically easier to optimise over the logarithm of the evidence, 
$$
\mathcal{L} = -\frac{1}{2}\log|\mathbf{C}| - \frac{1}{2}\mathbf{y}^T\mathbf{C}^{-1}\mathbf{y}\\
+ N \log \frac{\lambda}{2} - \frac{\lambda}{2}\sum_i\gamma_i\\
+ \nu \log \delta - \log \Gamma(\nu) + (\nu-1)\log\lambda - \delta\lambda\\
+ a \log b - \log \Gamma(a) + (a-1)\log\beta - b\beta
$$
where the first line is due to likelihood and its prior, the  following due to the hyperprior ( and hence corresponds to lasso-like terms), and the two to the gamma (hyper-) priors over $\lambda$ and $\beta$ . We can rewrite it in terms of the noise-invariant covariance $\hat{\mathbf{C}}$ as
$$
\mathcal{L} = \frac{N}{2}\log\beta-\frac{1}{2}\log|\hat{\mathbf{C}}| - \frac{\beta}{2}\mathbf{y}^T\hat{\mathbf{C}}^{-1}\mathbf{y}\\
+ N \log \frac{\lambda}{2} - \frac{\lambda}{2}\sum_i\gamma_i\\
+ \nu \log \delta - \log \Gamma(\nu) + (\nu-1)\log\lambda - \delta\lambda\\
+ a \log b - \log \Gamma(a) + (a-1)\log\beta - b\beta
$$


 In the next section we derive a fast optimalisation algorithm and show that many $\gamma_i$ will be zero, corresponding to a sparse solution.

## Optimalisation

We follow Tipsen for the fast optimisation of the likelihood. The basic idea is to split optimise every hyperparameter $\gamma_i$ *separately*, which allows for closed for solutions. Iteratively updating all the hyper parameters converges on the true solution. The advantage of the closed solutions is that it allows us to determine analytically whether a term should stay in the model or not. 

To optimise every $\gamma_i$ separately, we need to decompose the loss function in a part belonging to $\gamma_i$ and $\gamma_{-i}$, where $-i$ denotes all components except $i$. Considering the loss function above, we start by writing $\hat{\mathbf{C}}$ as
$$
\hat{\mathbf{C}} = \mathbf{I}_N + \sum_{m\neq i}\gamma_m \phi_m\phi_m^T+ \gamma_i \phi_i\phi_i^T = \hat{\mathbf{C}}_{-i} + \gamma_i \phi_i\phi_i^T
$$
Applying the Woodbury Indentity decomposes the inversion as
$$
\hat{\mathbf{C}}^{-1}=\hat{\mathbf{C}}^{-1}_{-i} - \frac{\hat{\mathbf{C}}^{-1}_{-i}\phi_i\phi_i^T\hat{\mathbf{C}}^{-1}_{-i}}{\gamma_i^{-1}+ \phi_i^T\hat{\mathbf{C}}^{-1}_{-i}\phi_i}
$$
For the determinant, we first factorise $\hat{\mathbf{C}}$ as
$$
\hat{\mathbf{C}} = \hat{\mathbf{C}}_{-i}\left(\mathbf{I}+\gamma_i\phi_i^T\hat{\mathbf{C}}^{-1}_{-i}\phi_i\right) = \hat{\mathbf{C}}_{-i}\left(1+\gamma_i\phi_i^T\hat{\mathbf{C}}^{-1}_{-i}\phi_i\right)
$$
and upon applying the matrix identity $det(AB) = det(A)\times det(B)$ we obtain
$$
|\hat{\mathbf{C}}| = |\hat{\mathbf{C}}_{-i}|\left|1+\gamma_i\phi_i^T\hat{\mathbf{C}}^{-1}_{-i}\phi_i\right| = |\hat{\mathbf{C}}_{-i}|\left(1+\gamma_i\phi_i^T\hat{\mathbf{C}}^{-1}_{-i}\phi_i\right)
$$
Putting these expressions into the derived log likelihood and ignoring every term independent of $\gamma$ we obtain
$$
\mathcal{L}(\gamma) = -\frac{1}{2}\left\{\log|\hat{\mathbf{C}}_{-i}|+\beta\mathbf{y}^T\hat{\mathbf{C}}_{-i}^{-1}\mathbf{y} + \lambda\sum_{m\neq i}\gamma_m\right\}\\
-\frac{1}{2}\left\{\log\left(1+\gamma_i\phi_i^T\hat{\mathbf{C}}^{-1}_{-i}\phi_i\right) -\beta\frac{\mathbf{y}^T\hat{\mathbf{C}}^{-1}_{-i}\phi_i\phi_i^T\hat{\mathbf{C}}^{-1}_{-i}\mathbf{y}}{\gamma_i^{-1}+ \phi_i^T\hat{\mathbf{C}}^{-1}_{-i}\phi_i} + \lambda\gamma_i \right\}\\
= \mathcal{L}(\gamma_{-i}) + \mathcal{L}(\gamma_{i})
$$

To make this more understandable, we introduce [TO DO, DEFINE HAT{S} and HAT{Q}] 
$$
s_i = \phi_i^T\hat{\mathbf{C}}^{-1}_{-i}\phi_i \quad \text{and} \quad q_i =\phi_i^T\hat{\mathbf{C}}^{-1}_{-i}\mathbf{y}
$$
where $s_i$ is a sparsity factor and $q_i$ is a quality factor [TO DO: EXPAND ON EXPLANATION WHY]. With these definitions, we rewrite $\mathcal{L}(\gamma_i)$ as 
$$
\mathcal{L}(\gamma_i) = \log(1+\gamma_is_i) - \frac{\beta\gamma_iq_i^2}{1 + \gamma_is_i} + \lambda\gamma_i
$$

### Minimizing $\gamma$

The derivative w.r.t $\gamma_i$ is
$$
\frac{\partial\mathcal{L}(\gamma_i)}{\partial\gamma_i} = \frac{s_i(1 + \gamma_is_i)- \beta q^2 + \lambda(1 + \gamma_is_i)^2}{(1 + \gamma_is_i)^2} \\ = \frac{\gamma_i^2(\lambda s_i^2) + \gamma_i(2\lambda s_i + s_i^2)+(\lambda+s_i-\beta q_i^2)}{(1+\gamma_i s_i)^2}
$$


the denominator is always positive while the numerator is of a quadatric form, thus giving the solutions
$$
\gamma_i = \frac{-(2\lambda s_i+ s_i^2)\pm\sqrt{(2\lambda s_i + s_i^2)^2-4\lambda s_i^2(\lambda + s_i - \beta q_i^2)}}{2\lambda s_i^2}
$$
In this form it's easier to analyze. Note that for $\beta q_i^2 - s_i < \lambda$, both solutions are negative.  We require  that $\gamma_i > 0$, and the derivative at $\gamma_i = 0$ is smaller than zero, so for this case the minimum is $\gamma_i = 0$. For $\beta q_i^2 - s_i > \lambda$, we can shorten the previous equation so that we finally obtain
$$
\gamma_i = 
\begin{cases}
	\frac{-2\lambda - s_i + \sqrt{4\beta\lambda q_i^2+s_i^2}}{2\lambda s_i}, & \text{if } \beta q_i^2 - s_i > \lambda \\
	0, & \text{if } \beta q_i^2 - s_i \leq \lambda
\end{cases}
$$
It's also more convenient to define 
$$
\hat{S}_i = \phi_i^T\hat{\mathbf{C}}^{-1}\phi_i, \quad \hat{Q}_i = \phi_i^T\hat{\mathbf{C}}^{-1}\mathbf{y}
$$
with which we can calculate $s_i$ and $q_i$ as
$$
\hat{s_i} = \frac{\hat{S}_i}{1-\gamma_i \hat{S}_i}, \quad \hat{q_i} = \frac{\hat{Q}_i}{1-\gamma_i \hat{S}_i}.
$$
Since $\hat{\mathbf{C}}$ is an $N \times N$ matrix, inverting it is very expensive. Using the Woodbury Identity, we rewrite it in terms of the inverse of an $M \times M$ matrix, where $ M \ll N$ as $M$ is typically less than 5:
$$
\hat{S}_i = \phi_i^T\phi_i - \phi_i^T\Phi\hat{\mathbf{\Sigma}}\Phi^T\phi_i\\
\hat{Q}_i = \phi_i^T\mathbf{y} - \phi_i^T\Phi\hat{\mathbf{\Sigma}}\Phi^T\mathbf{y}
$$

### Minimizing $\lambda$

We also need to minimize w.r.t $\lambda$. The derivative is
$$
\frac{\partial \mathcal{L}}{\partial \lambda} = \frac{N}{\lambda}-\frac{1}{2}\sum_i\gamma_i + \frac{\nu -1}{\lambda} - \delta
$$
which gives as the optimal value for $\lambda$ as,
$$
\lambda = \frac{2(N+\nu-1)}{\sum_i \gamma_i + 2\delta}.
$$
We thus require knowledge of $\nu$ and $\delta$ to estimate $\lambda$. Minimizing for $\nu$ and $\delta$ gives,
$$
\delta = \frac{\nu}{\lambda}\\
\log(\nu) = \psi(\nu)
$$
where $\psi$ is the digamma function. This problem is solved by $\nu =0, \delta =0$ giving for $\lambda$
$$
\lambda = \frac{2(N-1)}{\sum_i \gamma_i}.
$$
Note that this is different from Helgoy et al, who update $\delta$ and $\nu$. However, Babacaun have performed a comparison between $\nu=0$ and $\nu$ and $\delta$ automatically set, but at the same value and observed slightly better performance. In effect, we're applying a non-informative prior on $\lambda$. 

## Noise update

Rather than following Babaucan, we do wish to update the noise. Starting from the log-likelihood, it's derivative w.r.t $\beta$ is 
$$
\frac{\partial \mathcal{L}}{\partial \beta} = \frac{N}{2\beta} - \frac{1}{2}\mathbf{y}^T\hat{\mathbf{C}}^{-1}\mathbf{y}
$$
Solving for $\beta$ gives 
$$
\beta = \frac{N}{\mathbf{y}^T\hat{\mathbf{C}}^{-1}\mathbf{y}}
$$
This again involves the inverse of $\hat{\mathbf{C}}$ which is inefficient to calculate. Using the Woodbury inversion identity on $\hat{\mathbf{C}}$ gives
$$
\mathbf{y}^T\hat{\mathbf{C}}^{-1}\mathbf{y} = \mathbf{y}^T(\mathbf{y}-\mathbf{\Phi}\mathbf{\mu})
$$
so that we obtain for the final new estimate of $\beta$
$$
\beta = \frac{N}{\mathbf{y}^T(\mathbf{y}-\mathbf{\Phi}\mathbf{\mu})}
$$
which is easy to calculate since we know all the terms already.



## Starting vector and initial $\lambda$

Tipping recommends the starting vector to be the one with largest normalized projection on the target vector, i.e the $\phi_i$  which maximises
$$
\frac{||\phi_i^T\mathbf{y}||^2}{||\phi_i||^2}.
$$
This vector maximizes the largest initial likelihood as well [TO DO: CHECK]. In the following two sections we calculate what $\gamma_i$ would be in case of $\lambda=0$ initially or develop a method to self-consistently determine $\lambda$ and $\gamma_i$. Addiotionaly, if we normalize $\Theta$ the have norm-1 columns, $||\phi_i||=1$, so that the vector initial vector is found by 
$$
||\phi_i^T\mathbf{y}||^2.
$$

### Using $\lambda =0 $

The limit $\lim_{\lambda \to 0}\gamma_i$ doesn't converge due to a $\lambda^{-1}$ term. Nonetheless, by solving $\lim_{\lambda\to0}\partial\mathcal{L}/\partial\gamma_i$ we obtain 
$$
\gamma_i = \frac{\beta \hat{q}_i^2-\hat{s}_i}{\hat{s}_i^2}
$$
which is similar to Tippings result. Note that for the initial vector, all $\gamma_{-i}$ are 0 so that $\hat{\mathbf{C}}^{-1}_{-i} = \mathbf{I}_N$. Consequently, we see that 
$$
\hat{s}_i = \phi_i^T\phi_i, \quad \hat{q}_i = \phi_i^T\mathbf{y}.
$$
Putting this back into our expression for $\gamma$ we get for the initial value,
$$
\gamma_i = \frac{\beta||\phi_i^T\mathbf{y}||^2/||\phi_i||^2-1}{||\phi_i||^2}
$$
This equation greatly simplifies when $\Theta$ has been normalized to have norm-1 columns,
$$
\gamma_i = \beta||\phi_i^T\mathbf{y}||^2-1
$$

### Estimating $\lambda$

We can also self-consistently find $\gamma_i$ and $\lambda$ . For the initial term, we have for $\lambda$
$$
\lambda = \frac{2(N-1)}{\gamma_i}
$$

## Updating optimal vector

## Bringing it all together









## 