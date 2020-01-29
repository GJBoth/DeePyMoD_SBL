import numpy as np

def c(x, t, D, a):
    c = 1 / np.sqrt(2*np.pi*(2*D*t + a**2))*np.exp(-x**2/(2*a**2+4*D*t))
    
    return c

def c_x(x, t, D, a):
    c_x = -x/(a**2+2*D*t) * c(x, t, D, a)
    return c_x


def c_xx(x, t, D, a):
    c_xx = -1/(a**2+2*D*t) * c(x, t, D, a) - x/(a**2+2*D*t) * c_x(x, t, D, a)
    return c_xx
    
def c_t(x, t, D, a):
    c_t = D * c_xx(x, t, D, a)
    
    return c_t

def theta_analytical(x_grid, t_grid, D, a):
    u = c(x_grid, t_grid, D, a).reshape(-1, 1) # Because numerical derivatives don't do at edge
    u2 = u**2

    u_x = c_x(x_grid, t_grid, D, a).reshape(-1, 1)
    u_xx = c_xx(x_grid, t_grid, D, a).reshape(-1, 1)

    u_t = c_t(x_grid, t_grid, D, a).reshape(-1, 1)
    
    theta_analytical = np.concatenate([np.ones_like(u), u_x, u_xx, u, u*u_x, u*u_xx, u**2, u**2*u_x, u**2*u_xx], axis=1)
    time_deriv_analytical = u_t
    
    return time_deriv_analytical, theta_analytical