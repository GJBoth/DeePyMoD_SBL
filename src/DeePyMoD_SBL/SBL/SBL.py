import numpy as np


def SBL(theta, t):
    # Initializing
    Phi, alfa, beta = initialize(theta, t)
    Sigma, mu = posterior(Phi, t, alfa, beta)
    sm, qm, Sm, Qm = sparse_quality_factor(theta, t, Phi, Sigma, alfa, beta)

    # Running
    converged = False
    while converged is False:
        Phi, alfa = update_design_matrix(theta, sm, qm, alfa, Sm, Qm)
        Sigma, mu = posterior(Phi, t, alfa, beta)
        sm, qm, Sm, Qm = sparse_quality_factor(theta, t, Phi, Sigma, alfa, beta)
        beta = update_noise(Phi, t, mu, Sigma, alfa)
        converged = convergence(sm, qm, alfa)

    return alfa, mu, Sigma, 1/beta


def initialize(theta, t):
    # Noise level
    beta = 1 / (np.var(t) * 0.1)  # beta = 1/sigma^2

    # Finding best initial vector
    projection = np.concatenate([((phi_i[:, None].T @ t).T @ (phi_i[:, None].T @ t)) / (phi_i[:, None].T @ phi_i[:, None]) for phi_i in theta.T])
    start_idx = np.argmax(projection)

    # Initializing alphas
    alfa = np.ones((theta.shape[1], 1)) * np.inf
    alfa[start_idx] = theta[:, start_idx:start_idx+1].T @ theta[:, start_idx:start_idx+1] / (projection[start_idx] - 1/beta)
    Phi = theta[:, [start_idx]]

    return Phi, alfa, beta


def posterior(Phi, t, alfa, beta):
    Sigma = np.linalg.inv(alfa[alfa != np.inf] * np.eye(Phi.shape[1]) + beta * Phi.T @ Phi)  # posterior covariance
    mu = beta * Sigma @ Phi.T @ t  # posterior mean

    return Sigma, mu


def sparse_quality_factor(theta, t, Phi, Sigma, alfa, beta):
    B = beta * np.eye(Phi.shape[0])
    precalc = B @ Phi @ Sigma @ Phi.T @ B

    Sm = np.concatenate([phi_i[:, None].T @ B @ phi_i[:, None] - phi_i[:, None].T @ precalc @ phi_i[:, None] for phi_i in theta.T])
    Qm = np.concatenate([phi_i[:, None].T @ B @ t - phi_i[:, None].T @ precalc @ t for phi_i in theta.T])

    sm = Sm/(1 - Sm/alfa)
    qm = Qm/(1 - Sm/alfa)

    return sm, qm, Sm, Qm


def update_design_matrix(theta, sm, qm, alfa, Sm, Qm):
    idx = optimal_vec(sm, qm, Sm, Qm, alfa)
    theta_i = qm[idx, 0]**2 - sm[idx, 0]

    # Decididing what to do
    if (theta_i > 0) & (alfa[idx, 0] != np.inf):
        alfa[idx, 0] = sm[idx, 0]**2 / theta_i  # reestimating
    elif (theta_i > 0) & (alfa[idx, 0] == np.inf):
        alfa[idx, 0] = sm[idx, 0]**2 / theta_i  # adding alpha
    elif (theta_i< 0) & (alfa[idx, 0] != np.inf):
        alfa[idx, 0] = np.inf #removing alpha

    Phi = theta[:, alfa[:, 0] != np.inf]  # rebuilding phi

    return Phi, alfa


def update_noise(Phi, t, mu, Sigma, alfa):
    beta = (Phi.shape[0] - Phi.shape[1] + np.sum(alfa[alfa != np.inf] * np.diag(Sigma))) / ((t - Phi @ mu).T @ (t - Phi @ mu))

    return beta


def convergence(sm, qm, alfa):
    dt = qm**2 - sm
    delta_alfa = sm**2 / dt - alfa  #check a_new - a
    converged = np.max(np.abs(delta_alfa[dt > 0])) < 10**-6  # if max delta_a < 10^-6 and all other dt < 0, it has converged

    return converged


def optimal_vec(sm, qm, Sm, Qm, alfa):
    basis_idx = alfa != np.inf  # idx of bases in model
    set_idx = alfa == np.inf  # idx of bases not in model

    add_basis = (Qm**2 - Sm)/Sm + np.log(Sm/Qm**2)
    del_basis = Qm**2/(Sm - alfa) - np.log(1-Sm/alfa)
    alfa_new = sm**2/(qm**2 - sm)
    redo_basis = Qm**2/(Sm + (1/alfa_new-1/alfa)**-1) - np.log(1 + Sm * (1/alfa_new-1/alfa))

    #Making everything into nice matrix
    add_basis[basis_idx] = np.nan
    dt = qm**2 - sm
    add_basis[dt <= 0] = np.nan #stuff above assumes dt > 0
    del_basis[set_idx] = np.nan
    redo_basis[set_idx] = np.nan

    # Deciding update
    possible_update = np.concatenate((add_basis, redo_basis, del_basis), axis=1)
    idx = np.unravel_index(np.nanargmax(possible_update), possible_update.shape)[0]

    return idx
