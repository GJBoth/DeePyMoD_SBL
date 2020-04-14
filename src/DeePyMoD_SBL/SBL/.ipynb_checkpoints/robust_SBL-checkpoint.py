import numpy as np
from copy import copy


def robust_SBL(theta, y, nu=1.0, delta=5.0):
    beta, gamma, Phi, theta_normed, y_normed = initialize(theta, y)

    l = 0.0
    n_samples = theta_normed.shape[0]
    converged = False

    mu, Sigma = posterior(Phi, gamma, y_normed)
    si, qi = sparse_quality_factor(theta_normed, Phi, Sigma, gamma, y_normed)
    beta = update_noise(n_samples, y_normed, Phi, mu)
    l = update_filter(gamma, nu, n_samples, beta, qi, si)
    while not converged:
        Phi, gamma = update_design_matrix(Phi, theta_normed, gamma, si, qi, beta, l)
        mu, Sigma = posterior(Phi, gamma, y_normed)
        si, qi = sparse_quality_factor(theta_normed, Phi, Sigma, gamma, y_normed)
        beta = update_noise(n_samples, y_normed, Phi, mu)
        l = update_filter(gamma, nu, n_samples, beta, qi, si)
        converged = convergence(si, qi, l, beta, gamma)
    return gamma


def posterior(Phi, gamma, y):
    Sigma = np.linalg.inv(Phi.T @ Phi + np.diag(gamma[gamma != 0.0]**-1))
    mu = Sigma @ Phi.T @ y

    return mu, Sigma


def sparse_quality_factor(theta, Phi, Sigma, gamma, y):
    precalc = Phi @ Sigma @ Phi.T

    Sm = np.concatenate([phi_i[:, None].T @ phi_i[:, None] - phi_i[:, None].T @ precalc @ phi_i[:, None] for phi_i in theta.T])
    Qm = np.concatenate([phi_i[:, None].T @ y - phi_i[:, None].T @ precalc @ y for phi_i in theta.T])

    si = Sm / (1 - gamma[:, None] * Sm)
    qi = Qm / (1 - gamma[:, None] * Sm)

    return si, qi


def update_design_matrix(Phi, theta, gamma, si, qi, beta, l):
    gamma_current = copy(gamma)
    gamma_new = ((-2 * l - si + np.sqrt(4*beta*l*qi**2+si**2)) / (2*l*si)).squeeze()

    in_basis_idx = (gamma_current != 0.0)
    out_basis_idx = (gamma_current == 0.0)
    allowed = beta * qi**2 - si >= l

    dL_adding = dL(gamma_new, si, qi, beta, l)
    dL_adding[in_basis_idx] = 0.0
    dL_adding[~allowed] = 0.0

    dL_removing = -dL(gamma_current, si, qi, beta, l)
    dL_removing[out_basis_idx] = 0.0
    dL_removing[allowed] = 0.0

    dL_updating = dL(gamma_new, si, qi, beta, l) -dL(gamma_current, si, qi, beta, l)
    dL_updating[out_basis_idx] = 0.0
    dL_updating[~allowed] = 0.0

    dL_complete = np.concatenate((dL_adding, dL_updating, dL_removing), axis=1)
    idx, action = np.unravel_index(np.argmax(dL_complete), dL_complete.shape)

    if action == 0:
        gamma_current[idx] = gamma_new[idx]
    elif action == 1:
        gamma_current[idx] = gamma_new[idx]
    elif action == 2:
        gamma_current[idx] = 0.0

    Phi = theta[:, gamma_current != 0.0]
    return Phi, gamma_current

def update_noise(N, y, Phi, mu):
    beta = N * np.linalg.inv(y.T @ (y - Phi @ mu))
    return beta

def update_filter(gamma, nu, n_samples, beta, qi, si):
    delta = (n_samples - 1 + nu) / np.max(beta * qi**2 - si)
    l = 2 * (n_samples - 1 + nu) / (np.sum(gamma) + 2 * delta)

    return l

def initialize(theta, y):
    theta_normed = theta / np.linalg.norm(theta, axis=0, keepdims=True)
    y_normed = y / np.linalg.norm(y)
    beta = 1 / (0.1 * np.var(y_normed))

    ini_idx = np.argmax((theta_normed.T @ y_normed)**2)
    Phi = theta_normed[:, [ini_idx]]
    n_terms = theta.shape[1]
    gamma = np.zeros(n_terms)
    gamma[ini_idx] = beta * (Phi.T @ y_normed)**2 - 1

    return beta, gamma, Phi, theta_normed, y_normed

def convergence(si, qi, l, beta, gamma):
    if np.sum(gamma != 0.0):
        if np.all((beta * qi**2 - si)[gamma == 0.0] <=l): # if all elements not in model shouldnt be in model
            dgamma = gamma - ((-2 * l - si + np.sqrt(4*beta*l*qi**2+si**2)) / (2*l*si)).squeeze()
            if np.max(np.abs(dgamma[gamma != 0.0])) < 1e-6:
                converged = True
            else:
                converged= False
        else:
            converged = False
    else:
        converged = False
    return converged

def dL(gamma, si, qi, beta, l):
    delta_l = -np.log(1 + gamma[:, None] * si) + beta * gamma[:, None] * qi**2/(1+gamma[:, None]*si) - l * gamma[:, None]
    return delta_l
