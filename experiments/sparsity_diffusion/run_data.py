import numpy as np
from DeePyMoD_SBL.data.diffusion import library

np.random.seed(42)


x_points = 100
t_points = 25
x_grid, t_grid = np.meshgrid(np.linspace(-10, 10, x_points), np.linspace(0, 1, t_points), indexing='ij')

time_deriv, theta = library(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1), D=2.0, x0=0.0, sigma=0.5)

data = {'time_deriv': time_deriv, 'theta': theta}
np.save('data.npy', data)
