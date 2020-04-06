from DeePyMoD_SBL.data.diffusion import library
from DeePyMoD_SBL.SBL import SBL
import numpy as np

x_points = 100
t_points = 25
x_grid, t_grid = np.meshgrid(np.linspace(-10, 10, x_points), np.linspace(0, 1, t_points), indexing='ij')
time_deriv, theta = library(x_grid.reshape(-1, 1), t_grid.reshape(-1, 1))

noise = np.var(time_deriv) * 0.01
t = time_deriv + np.random.normal(scale= np.sqrt(noise), size=time_deriv.shape)

alfa, mu, Sigma, noise = SBL(theta, t)

mu

alfa
