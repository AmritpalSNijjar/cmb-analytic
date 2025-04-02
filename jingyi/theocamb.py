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
        self.ombh2 = ombh2
        self.omch2 = omch2
        self.H0    = H0
        self.h     = H0/100
        self.As    = As
        self.omlh2 = (self.h)**2 - ombh2 - omch2

        self.a     = np.arange(1e-5,1+0.9e-7,1e-7) #can change steps

    def H(self):
        self.aeq    = 1/3400
        self.omm    = (self.ombh2 + self.omch2)/ self.h**2
        self.omr    = self.aeq * self.omm
        self.omlamb = 1 - self.omm - self.omr
        self.H = self.H0 * np.sqrt(self.omr * self.a**(-4) + self.omm * self.a**(-3) + self.omlamb)
        #return self.H
         #Implement Hubble parameter



    def etaintegrate(self,a):
        self.aeq    = 1/3400
        self.omm    = (self.ombh2 + self.omch2)/ self.h**2
        self.omr    = self.aeq * self.omm
        self.omlamb = 1 - self.omm - self.omr
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

"""
    def Phi(self):
        #Implement G potential

    def Psi(self):
        #Implement G potential
"""    

