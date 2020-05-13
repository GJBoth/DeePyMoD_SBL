'''Sparsity estimators which can be plugged into deepmod.
We keep the API in line with scikit learn (mostly), so scikit learn can also be plugged in.
See scikitlearn.linear_models for applicable estimators.'''

import numpy as np
from sklearn.cluster import KMeans
from pysindy.optimizers import STLSQ
from sklearn.linear_models import LassoCV

import warnings
warnings.filterwarnings("ignore", category=UserWarning)  # To silence annoying pysindy warnings


class Threshold:
    '''Performs additional thresholding on coefficient result from estimator. Basically
    a thin wrapper around the given estimator. '''
    def __init__(self, threshold=0.1, estimator=LassoCV):
        self.estimator = estimator
        self.threshold = threshold

        # Library contains offset so turn off the intercept
        self.estimator.set_params(fit_intercept=False)

    def fit(self, X, y):
        coeff = self.estimator.fit(X, y).coef_
        coeff[np.abs(coeff) < self.threshold] = 0.0

        self.coef_ = coeff  # to keep in line width sklearn
        return self


class Clustering():
    ''' Sparsity estimator based on k-means clusternig.'''
    def __init__(self, estimator=LassoCV):
        self.estimator = estimator
        self.kmeans = KMeans(n_clusters=2)

        # Library contains offset so turn off the intercept
        self.estimator.set_params(fit_intercept=False)

    def fit(self, X, y):
        coeff = self.estimator.fit(X, y).coef_
        clusters = self.kmeans.fit_predict(np.abs(coeff)).astype(np.bool)

        # make sure terms to keep are 1 and to remove are 0
        max_idx = np.argmax(np.abs(coeff))
        if clusters[max_idx] != 1:
            clusters = ~clusters

        self.coef_ = clusters.astype(np.float32)  #to keep in line with sklearn
        return self


class PDEFIND():
    def __init__(self, lam=1e-5, dtol=1, **kwargs):
        self.lam = lam
        self.dtol = dtol
        self.kwargs = kwargs

    def fit(self, X, y):
        coeff = PDEFIND.TrainSTRidge(X, y[:, None], self.lam, self.dtol, **self.kwargs)
        self.coef_ = coeff.squeeze()
        return self

    @staticmethod
    def TrainSTRidge(R, Ut, lam, d_tol, maxit = 25, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8, print_best_tol = False):
        """
        This function trains a predictor using STRidge.

        It runs over different values of tolerance and trains predictors on a training set, then evaluates them
        using a loss function on a holdout set.

        Please note published article has typo.  Loss function used here for model selection evaluates fidelity using 2-norm,
        not squared 2-norm.
        """

        # Split data into 80% training and 20% test, then search for the best tolderance.
        np.random.seed(0) # for consistancy
        n,_ = R.shape
        train = np.random.choice(n, int(n*split), replace = False)
        test = [i for i in np.arange(n) if i not in train]
        TrainR = R[train,:]
        TestR = R[test,:]
        TrainY = Ut[train,:]
        TestY = Ut[test,:]
        D = TrainR.shape[1]

        # Set up the initial tolerance and l0 penalty
        d_tol = float(d_tol)
        tol = d_tol
        if l0_penalty == None: l0_penalty = 0.001*np.linalg.cond(R)

        # Get the standard least squares estimator
        w = np.zeros((D,1))
        w_best = np.linalg.lstsq(TrainR, TrainY, rcond=None)[0]
        err_best = np.linalg.norm(TestY - TestR.dot(w_best), 2) + l0_penalty*np.count_nonzero(w_best)
        tol_best = 0

        # Now increase tolerance until test performance decreases
        for iter in range(maxit):

            # Get a set of coefficients and error
            opt = STLSQ(threshold=tol, alpha=lam, fit_intercept=False)
            w = opt.fit(R, Ut).coef_.T
            err = np.linalg.norm(TestY - TestR.dot(w), 2) + l0_penalty * np.count_nonzero(w)
            # Has the accuracy improved?
            if (err <= err_best) and (not np.all(w == 0)):
                err_best = err
                w_best = w
                tol_best = tol
                tol = tol + d_tol

            else:
                tol = max([0,tol - 2*d_tol])
                d_tol  = 2*d_tol / (maxit - iter)
                tol = tol + d_tol
        return w_best
