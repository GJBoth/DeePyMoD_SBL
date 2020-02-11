from deepmod_l1.analytical import theta_analytical
from deepmod_l1.SBL import SBL
import numpy as np

D = 0.5
a = 0.25

x = np.linspace(-5, 5, 50, dtype=np.float32)
t = np.linspace(0, 5, 50, dtype=np.float32)
x_grid, t_grid = np.meshgrid(x, t, indexing='ij')
    
# Analytical
time_deriv, theta = theta_analytical(x_grid, t_grid, D, a)

noise = np.var(time_deriv) * 0.1
t = time_deriv + np.random.normal(scale= np.sqrt(noise), size=time_deriv.shape)

alfa, mu, Sigma, noise_estimate = SBL(theta, t)
print(alfa)
print(mu)
print(Sigma)
print(noise, noise_estimate)