import numpy as np
from scipy.special import erfc

class Burgers:
    ''' Class to generate analytical solutions of Burgers equation with delta peak initial condition. 
    
    Good source: https://www.iist.ac.in/sites/default/files/people/IN08026/Burgers_equation_viscous.pdf'''
    
    def __init__(self, viscosity, A):
        self.v = viscosity
        self.A = A
    
    def solution(self, x, t):
        '''Generation solution.'''
        return self.u(x, t, self.v, self.A)
    
    def library(self, x, t):
        ''' Returns library with 3rd order derivs and 2nd order polynomial'''
        u = self.u(x, t, self.v, self.A)
        u_x = self.u_x(x, t, self.v, self.A)
        u_xx = self.u_xx(x, t, self.v, self.A)
        u_xxx = self.u_xxx(x, t, self.v, self.A)
  
        derivs = np.concatenate([np.ones_like(u), u_x, u_xx, u_xxx], axis=1)
        theta = np.concatenate([derivs, u * derivs, u**2 * derivs], axis=1)

        return theta
    
    def time_deriv(self, x, t):
        ''' Return time derivative'''
        u_t = self.u_t(x, t, self.v, self.A)
        return u_t
    
    @staticmethod
    def u(x, t, v, A):
        '''Calculates solution.'''
        R = A/(2*v)
        z = x/np.sqrt(4*v*t)
        
        solution = np.sqrt(v/t) * ((np.exp(R) - 1) * np.exp(-z**2)) / (np.sqrt(np.pi) + (np.exp(R) - 1)*np.sqrt(np.pi/2)*erfc(z))
        return solution
    
    @staticmethod
    def u_x(x, t, v, A):
        '''Calculates first order spatial derivative of solution.'''
        R = A/(2*v)
        z = x/np.sqrt(4*v*t)
        
        u = Burgers.u(x, t, v, A)
        u_x = 1/np.sqrt(4*v*t) * (np.sqrt(2*t/v)*u**2-2*z*u)
        return u_x
    
    @staticmethod
    def u_xx(x, t, v, A):
        '''Calculates second order spatial derivative of solution.'''
        R = A/(2*v)
        z = x/np.sqrt(4*v*t)
        
        u = Burgers.u(x, t, v, A)
        u_x = Burgers.u_x(x, t, v, A)
        u_xx = 1/np.sqrt(4*v*t) * (-2*u/np.sqrt(4*v*t) - 2*z*u_x + 2*np.sqrt(2*t/v)*u*u_x) # could be written shorter, but then get NaNs due to inversions
        return u_xx
    
    @staticmethod
    def u_xxx(x, t, v, A):
        '''Calculates third order spatial derivative of solution.'''
        R = A/(2*v)
        z = x/np.sqrt(4*v*t)
        
        u = Burgers.u(x, t, v, A)
        u_x = Burgers.u_x(x, t, v, A)
        u_xx = Burgers.u_xx(x, t, v, A)
        u_xxx = 1/np.sqrt(4*v*t) * (-4/np.sqrt(4*v*t) * u_x + 2 *np.sqrt(2*t/v)*u_x**2 + u_xx*(-2*z+2*np.sqrt(2*t/v)*u)) # could be written shorter, but then get NaNs due to inversions
        return u_xxx
    
    @staticmethod
    def u_t(x, t, v, A):
        '''Calculates first order temporal derivative of solution.'''
        R = A/(2*v)
        z = x/np.sqrt(4*v*t)
        
        u = Burgers.u(x, t, v, A)
        u_x = Burgers.u_x(x, t, v, A)
        u_xx = Burgers.u_xx(x, t, v, A)
        u_t = v * u_xx - u *u_x
        return u_t