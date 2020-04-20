import torch
import torch.nn as nn


class Library(nn.Module):
    def __init__(self, library_func, library_args={}):
        super().__init__()
        self.library_func = library_func
        self.library_args = library_args

    def forward(self, input):
        time_deriv_list, theta = self.library_func(input, **self.library_args)
        return time_deriv_list, theta


class Fitting(nn.Module):
    def __init__(self, n_terms, n_out):
        super().__init__()
        self.coeff_vector = nn.ParameterList([torch.nn.Parameter(torch.rand((n_terms, 1), dtype=torch.float32)) for _ in torch.arange(n_out)])
        self.sparsity_mask = [torch.ones(n_terms, dtype=torch.bool) for _ in torch.arange(n_out)]

    def forward(self, input):
        thetas, time_derivs = input
        sparse_thetas = self.apply_mask(thetas)
        self.coeff_vector = self.fit_coefficient(sparse_thetas, time_derivs)
        return sparse_thetas, self.coeff_vector

    def apply_mask(self, theta):
        sparse_theta = [theta[:, sparsity_mask] for sparsity_mask in self.sparsity_mask]
        return sparse_theta
    
    def fit_coefficient(self, thetas, time_derivs):
        return self.coeff_vector

    
class FittingDynamic(nn.Module):
    def __init__(self, n_terms, n_out):
        super().__init__()
        self.coeff_vector = [torch.rand((n_terms, 1), dtype=torch.float32) for _ in torch.arange(n_out)] # initialize randomly cause otherwise tensorboard will complain
        self.sparsity_mask = [torch.ones(n_terms, dtype=torch.bool) for _ in torch.arange(n_out)]

    def forward(self, input):
        thetas, time_derivs = input
        sparse_thetas = self.apply_mask(thetas)
        self.coeff_vector = self.fit_coefficient(sparse_thetas, time_derivs)
        return sparse_thetas, self.coeff_vector

    def apply_mask(self, theta):
        sparse_theta = [theta[:, sparsity_mask] for sparsity_mask in self.sparsity_mask]
        return sparse_theta
    
    def fit_coefficient(self, thetas, time_derivs):
        opt_coeff = [torch.inverse(theta.T @ theta) @ (theta.T @ dt) for theta, dt in zip(thetas, time_derivs)] # normal equation for least squares
        return opt_coeff
