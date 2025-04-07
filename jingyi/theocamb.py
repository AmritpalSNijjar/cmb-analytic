import numpy as np
import scipy
import astropy
import scipy.linalg
import os
import sys
from scipy.stats import multivariate_normal
import itertools
import scipy.integrate as integrate
from scipy.integrate import quad


class TheoCamb:
    def __init__(self, ombh2, omch2, H0, As):
        self.aeq    = 1/3400
        self.ombh2  = ombh2
        self.omch2  = omch2
        self.H0     = H0
        self.h      = H0/100
        self.As     = As
        self.f_nu   = 0.405
        self.a0     = 1
        self.omm    = (self.ombh2 + self.omch2)/ self.h**2
        self.omr    = self.aeq * self.omm
        self.omlamb = 1 - self.omm - self.omr
        self.omlh2  = (self.h)**2 - ombh2 - omch2
        self.k_eq   = np.sqrt(2 * self.omm * a0) * H0
        self.a      = np.arange(1e-5,1+0.9e-7,1e-7) #can change steps

    def H(self):
        self.H = self.H0 * np.sqrt(self.omr * self.a**(-4) + self.omm * self.a**(-3) + self.omlamb)
        #return self.H
         #Implement Hubble parameter

    def etaintegrate(self,a):
        H = self.H0 * np.sqrt(self.omr * a**(-4) + self.omm * a**(-3) + self.omlamb)
        return 1/H

    def eta(self, a0):
        return quad(self.etaintegrate, 0, a0)[0] 
         #Implement how to integrate to get comformal time

    def listeta(self):
        eta = []
        for a_val in self.a:
           itemeta = self.eta(a_val)
           eta.append(itemeta)
        return np.array(eta)

    def ug(self, a):
        return (a**3 + 2*a**2/9 - 8*a/9 - 16/9 + 16*np.sqrt(a+1)/9)/(a*(a+1))

    def ud(self, a):
        return 1/(a*np.sqrt(a+1))

    def A(self, k):
        return np.sqrt(As*(k**(n-1))((k_eq/k)**4)((6/(5+2*f_nu))**2))

    def Delta_T(self, a):
        return (1 + (2 * f_nu * (1 - (0.333*a/(a+1))))/5) * self.A(k) * self.ug(a)

    def N_2(a, k):
        Nitem1 = -((20*a + 19)*self.A(k)*self.ug(a))/(10*(3*a + 4))
        Nitem2 = -(8*a*self.A(k))/(3*(3*a + 4))
        Nitem3 = (8*self.A(k)*np.log((3*a + 4)/4))/9
        return Nitem1 + Nitem2 + Nitem3

    def Phi(self, a, k):
        #Implement G potential
        return (3/4)*((k_eq/k)**2)*((a + 1)/a**2)*self.Delta_T(a)

    def Psi(self, a, k):
        #Implement G potential
        return (3/4)*((k_eq/k)**2)*((a + 1)/a**2)*(self.Delta_T(a) + (8*f_nu*self.N_2(a, k))/(5*(a+1)))
"""    

