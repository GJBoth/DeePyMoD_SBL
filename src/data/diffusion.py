import numpy as np


def library(x, t, D=2.0, x0=0.0, sigma=0.5):
    '''Calculates library and time derivative for diffusion process. x and t must be Nx1.
    Polynomial up to second order, derivative up to third.'''

    # Calculating derivatives
    p = (2 * np.pi * sigma**2 + 4 * np.pi * D * t)**(-1/2) * np.exp(-(x - x0)**2/(2*sigma**2+4 * D * t))
    p_x = -2 * (x - x0) / (2 * sigma**2 + 4 * D * t) * p
    p_xx = ((x-x0)**2 - sigma**2 - 2 * D * t) / (sigma**2 + 2 * D * t)**2 * p
    p_xxx = -(x - x0) * (-3 * sigma**2 - 6 * D * t + (x - x0)**2) / (sigma**2 + 2 * D * t)**3 * p

    p_t = D * p_xx

    # Creating theta
    derivs = np.concatenate([np.ones_like(p), p_x, p_xx, p_xxx], axis=1)
    theta = np.concatenate([derivs, p * derivs, p**2 * derivs], axis=1)

    return p_t, theta

if __name__ == '__main__':
    # testing if it works
    x_points = 100
    t_points = 25
    D = 2.0
    x_grid, t_grid = np.meshgrid(np.linspace(-10, 10, x_points), np.linspace(0, 1, t_points), indexing='ij')
    time_deriv, theta = library(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1), D=D)

    assert time_deriv.shape == (x_points * t_points, 1), "Time deriv has wrong shape."
    assert theta.shape == (x_points * t_points, 12) , "Theta has wrong shape."

    correct_coeffs = np.zeros((12, 1))
    correct_coeffs[2, 0] = D
    found_coeffs = np.linalg.lstsq(theta, time_deriv, rcond=None)[0]
    assert np.allclose(correct_coeff, found_coeffs), "Columns of theta arent correct."
