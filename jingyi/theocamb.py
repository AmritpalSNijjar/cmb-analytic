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
        self.a_eq   = 1/3400
        self.ombh2  = ombh2
        self.omch2  = omch2
        self.H0     = H0
        self.h      = H0/100
        self.As     = As
        self.f_nu   = 0.405
        self.a0     = 1
        self.omb    = self.ombh2/self.h**2
        self.omm    = (self.ombh2 + self.omch2)/ self.h**2
        self.omr    = self.a_eq * self.omm
        self.omlamb = 1 - self.omm - self.omr
        self.omlh2  = (self.h)**2 - self.ombh2 - self.omch2
        self.k_eq   = np.sqrt(2 * self.omm * self.a0) * self.H0
        self.R_eq   = self.R(self.a_eq)
        self.a      = np.arange(1e-5,1+0.9e-7,1e-7) #can change steps

    def H(self, a):
        self.H = self.H0 * np.sqrt(self.omr * (a**(-4)) + self.omm * (a**(-3)) + self.omlamb)
        #return self.H
         #Implement Hubble parameter

    def etaintegral(self, a):
        H = self.H0 * np.sqrt(self.omr * a**(-4) + self.omm * a**(-3) + self.omlamb)
        return 1/(H*(a**2))

    def eta(self, a0):
        return quad(self.etaintegral, 0, a0)[0] 
         #Implement how to integrate to get comformal time

    def listeta(self):
        eta = []
        for a_val in self.a:
           itemeta = self.eta(a_val)
           eta.append(itemeta)
        return np.array(eta)

#    def get_ks(self, k_lower = 1e-5, k_upper = 1, n_ks = 1e5, point_spacing = "log"):
        """
        Function for generating an array of k mode points to be used.
        """
    
#        if point_spacing == "lin":
#            return np.linspace(start = a_lower, stop = a_upper, num = n_as)
#        else: # log spaced a values
#            return np.geomspace(start = a_lower, stop = a_upper, num = n_as)():


    def ug(self, a):
        return (a**3 + 2*a**2/9 - 8*a/9 - 16/9 + 16*np.sqrt(a+1)/9)/(a*(a+1))

    def ud(self, a):
        return 1/(a*np.sqrt(a+1))

    def A(self, k):
        #return np.sqrt(self.As*(k**(n-1))((k_eq/k)**4)((6/(5+2*self.f_nu))**2))
        return np.sqrt(self.As*((self.k_eq/k)**4)*((6/(5+2*self.f_nu))**2))

    def Delta_T(self, a, k):
        return (1 + (2 * self.f_nu * (1 - (0.333*a/(a+1))))/5) * self.A(k) * self.ug(a)

    def N_2(self, a, k):
        Nitem1 = -((20*a + 19)*self.A(k)*self.ug(a))/(10*(3*a + 4))
        Nitem2 = -(8*a*self.A(k))/(3*(3*a + 4))
        Nitem3 = (8*self.A(k)*np.log((3*a + 4)/4))/9
        return Nitem1 + Nitem2 + Nitem3
 
    def Phi(self, a, k):
        #Implement G potential
        return (3/4)*((self.k_eq/k)**2)*((a + 1)/a**2)*self.Delta_T(a, k)

    def Psi(self, a, k):
        #Implement G potential
        return (3/4)*((self.k_eq/k)**2)*((a + 1)/a**2)*(self.Delta_T(a, k) + (8*self.f_nu*self.N_2(a, k))/(5*(a+1)))  

    def R(self, a):
        return (3 * self.omb * a)/((1 - self.f_nu) * 4 * self.omm)

    def R_dot(self, a):
        return self.k_eq * self.R_eq * np.sqrt((1 + a)/2)

    def R_ddot(self, a):
        return ((self.k_eq**2) * self.R_eq)/4

    def c_s(self, a):
        return np.sqrt(1/(3 * (1 + self.R(a))))

    def rs_intergral(self, a):
        # Eq 7
        rs_intergral = self.c_s(a) * self.etaintegrate(a)
        return rs_intergral

    def r_s(self, a0):
        # Eq 7
        rs_result = quad(self.rs_intergral, 0, a0)[0] 

        return rs_result 

    def J_eta(self, a, k):
        # Eq D-2
        J_result = (np.sqrt(3) * self.R_dot(a))/(4 * k * np.sqrt(1 + self.R(a)))

        return J_result

    def G_eta(self, a, k):
        # Eq D-4
        G_part1 = (1 + self.R(a))**(-1/4)
        G_part2 = 1 - ((1 + self.R(a))* self.Psi(a, k))/self.Phi(a, k)
        G_part3 = (3 * self.R_ddot(a))/(4 * (k**2)) - self.J_eta(a, k)**2

        G_result = G_part1 * (G_part2 + G_part3)

        return G_result

    def I_eta(self, a0, k):
        # Eq D-3
        
        I_integrate = lambda a:self.etaintegrate(a) * self.G_eta(a, k) * self.Phi(a, k) * np.sin(k * self.r_s(a0) - k * self.r_s(a))

        I_part1 = k/np.sqrt(3)
        I_part2 = quad(I_integrate, 0, a0)[0]

        I_result = I_part1 * I_part2

        return I_result

        



