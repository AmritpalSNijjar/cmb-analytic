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
from tqdm import tqdm

class TheoCamb:
    def __init__(self, ombh2, omch2, H0, As, ns, tau, noL_CDM = False):
        self.ombh2  = ombh2
        self.H0     = H0
        self.As     = As
        self.ns    = ns
        self.tau   = tau

        self.h      = H0/100        
        # self.a_eq   = 4.15e-5/self.ommh2 from Dodelson Eq.(2.87)
        # change a from a=1/3400 to this
        # Because a_eq in the paper is 1 while a defined here in from 1e-5 to 1, real a and a_eq used in the code below should be a/a_eq where a_eq is defined above
        
        if noL_CDM:
            self.om0h2 = self.h**2
            self.a_eq  = 4.15e-5/self.om0h2 # Dodelson Eq. (2.87)
            self.omrh2 = self.a_eq * (self.om0h2/self.h**2)
            self.omch2 = self.h**2 - self.ombh2 - self.omrh2
            self.omLh2 = 0
        else:
            self.omch2 = omch2
            self.om0h2 = self.ombh2 + self.omch2
            
            self.a_eq  = 4.15e-5/self.om0h2
            self.omrh2 = self.a_eq * (self.om0h2/self.h**2)
            
            self.omLh2 = self.h**2 - self.om0h2 - self.omrh2 


        self.f_nu   = 0.405
        self.a0     = 1
        self.omm    = self.om0h2/(self.h**2)
        self.omr    = self.omrh2/(self.h**2)
        self.omb    = self.ombh2/(self.h**2)
        self.omL    = self.omLh2/(self.h**2)
        
        #self.k_eq   = np.sqrt(2 * self.omm * self.a0 * 3400) * self.H0
        self.k_eq   = 0.073*(self.ombh2 + self.omch2)
        # dodelson 7.39, also in HU/S in the paragraph right after Eqn. D9

        self.R_eq   = self.R(1) #self.R(self.a_eq * 3400) 
        #R here is a function of a, R_eq = R(a_eq) = R(1)

        #set a
        #self.a_step = 1e-5
        #self.a      = np.arange(1e-5,1+0.9e-7,self.a_step) #can change steps
        self.n_as    = int(1e3+2)
        self.a_lower = 1e-5
        self.a_upper = 1

        self.n_ks    = int(1e3)
        self.k_lower = 1e-3
        self.k_upper = 1

        z_star       = 1000*(self.ombh2/self.h**2)**(-0.027/(1+0.11*np.log(self.ombh2/self.h**2))) # C2
        self.a_star  = 1/(1 + z_star)

        as_1     = self.get_scales(a_lower = self.a_lower, a_upper = self.a_star, n_as = self.n_as//2, point_spacing = "log", endpoint = False) 
        as_2     = self.get_scales(a_lower = self.a_star, a_upper = self.a_upper, n_as = self.n_as//2, point_spacing = "log")
        self.a_s = np.concatenate((as_1, as_2))

        self.a_s_rescaled = self.a_s/self.a_eq

        self.ks   = self.get_ks(k_lower = self.k_lower, k_upper = self.k_upper, n_ks = self.n_ks, point_spacing = "lin")

        self.etas = np.zeros_like(self.a_s)
        
        #if not the next line... errors when integrating!
        self.etas[0] = 0
        
        for i in range(1, self.n_as):
            self.etas[i] = self.etas[i - 1] + self.eta(self.a_s[i - 1], self.a_s[i])
        # Fill the array of eta

        self.recomb_ind = self.n_as//2
        self.eta_star = self.a_s[self.recomb_ind]

        # Dictionaries for switching between from eta to a
        self.a_of_eta = {self.etas[i]: self.a_s[i] for i in range(self.n_as)}
        
    def get_scales(self, a_lower = 1e-5, a_upper = 1, n_as = 1e5, point_spacing = "log", endpoint = True):
        """
        Function for generating an array of scale factor points to be used.
        """
        if point_spacing == "lin":
            return np.linspace(start = a_lower, stop = a_upper, num = int(n_as), endpoint = endpoint)
        else: # log spaced a values
            return np.geomspace(start = a_lower, stop = a_upper, num = int(n_as), endpoint = endpoint)

    def get_ks(self, k_lower = 1e-5, k_upper = 1, n_ks = 1e5, point_spacing = "log"):
        """
        Function for generating an array of k mode points to be used.
        """
    
        if point_spacing == "lin":
            return np.linspace(start = k_lower, stop = k_upper, num = int(n_ks))
        else: # log spaced a values
            return np.geomspace(start = k_lower, stop = k_upper, num = int(n_ks))

    def H(self, a):
        H = self.H0 * np.sqrt(self.omr * (a**(-4)) + self.omm * (a**(-3)) + self.omL)
        return H
         #Implement Hubble parameter

    def eta_integral(self, a):
        H = self.H0 * np.sqrt(self.omr * a**(-4) + self.omm * a**(-3) + self.omL)
        return 1/(H*(a**2))

    def eta(self, a_lower, a_upper):
        return quad(self.eta_integral, a_lower, a_upper)[0] 
         #Implement how to integrate to get comformal time

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
        return np.sqrt(self.As*((k/self.k_eq)**4)*((6/(5+2*self.f_nu))**2)*(k**(-3)))

    def Delta_T(self, a, k):
        return (1 + (2 * self.f_nu * (1 - (0.333*a/(a+1))))/5) * self.A(k) * self.ug(a)

    def N_2(self, a, k):
        Nitem1 = -((20*a + 19)*self.A(k)*self.ug(a))/(10*(3*a + 4))
        Nitem2 = -(8*a*self.A(k))/(3*(3*a + 4))
        Nitem3 = (8*self.A(k)*np.log((3*a + 4)/4))/9
        return Nitem1 + Nitem2 + Nitem3
 
    def Phi_bar(self, a, k):
        #Implement G potential
        return (3/4)*((self.k_eq/k)**2)*((a + 1)/a**2)*self.Delta_T(a, k)

    def Psi_bar(self, a, k):
        #Implement G potential
        return -(3/4)*((self.k_eq/k)**2)*((a + 1)/a**2)*(self.Delta_T(a, k) + (8*self.f_nu*self.N_2(a, k))/(5*(a+1)))

    def T_k(self, k):
        # Eq A-21
        q       = k/(self.omm * (self.h**2) * np.exp(-2 * self.omb))
        T_item1 = (np.log(1 + 2.34 * q))/(2.34 * q)
        T_item2 = 1 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4
        T_k     = T_item1 * (T_item2**(-1/4))
        return T_k

    def Phi(self, a, k):
        #Eq A-22
        alpha_1   = 0.11
        beta      = 1.6
        Phi_T     = self.T_k(k)
        Phi_item  = (1 - Phi_T)*np.exp(-alpha_1 * ((a * k/self.k_eq)**beta)) + Phi_T
        Phi       = self.Phi_bar(a, k) * Phi_item
        return Phi

    def Psi(self, a, k):
        #Eq A-22
        alpha_2   = 0.097
        beta      = 1.6
        Psi_T     = self.T_k(k)
        Psi_item  = (1 - Psi_T)*np.exp(-alpha_2 * ((a * k/self.k_eq)**beta)) + Psi_T
        Psi       = self.Psi_bar(a, k) * Psi_item
        return Psi

    def R(self, a):
        return (3 * self.omb * a)/((1 - self.f_nu) * 4 * self.omm)

    def R_dot(self, a):
        return self.k_eq * self.R_eq * np.sqrt((1 + a)/2)

    def R_ddot(self, a):
        return ((self.k_eq**2) * self.R_eq)/4

    def c_s(self, a):
        return np.sqrt(1/(3 * (1 + self.R(a))))

    def r_s(self, eta):
        # Because we have an eta array before, please pay attention that here the input is eta_array
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
        # a0 is an array

        n = len(self.a)
        if n % 2 == 0:
            n = n
        else:
            n = n + 1
        
        #I_integrate = lambda a:self.etaintegral(a) * self.G_eta(a, k) * self.Phi(a, k) * np.sin(k * self.r_s(a0[-1]) - k * self.r_s(a))

        #I_part1 = k/np.sqrt(3)
        #I_part2 = np.array([quad(I_integrate, 0, a)[0] for a in a0])

        #I_result = I_part1 * I_part2

        #return I_result

    def Theta_0_hat(self, a, k):
        # Eq D-1
        Theta_0_0 = 0.43 * self.Phi_bar(1e-5, k)#Theta_0_0 is defined by A19  
        r_s       = np.array([self.r_s(a) for a in self.a])
        J0        = self.J_eta(0, k)
        Theta_0_right1 = np.cos(k*r_s) + J0*np.sin(k*r_s)
        Theta_0_right2 = Theta_0_0 + self.Phi(1e-5, k)
        Theta_0_right  = Theta_0_right1 * Theta_0_right2 +self.I_eta(a, k)
        return Theta_0_right/((1 + self.R(a))**(1/4)) - self.Phi(a, k)

    #def 




