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
from scipy.integrate import simpson
#from scipy.integrate import cumulative_simpson

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

        self.R_eq   = self.R_a(1) #self.R(self.a_eq * 3400) 
        #R here is a function of a, R_eq = R(a_eq) = R(1)

        #set a
        #self.a_step = 1e-5
        #self.a      = np.arange(1e-5,1+0.9e-7,self.a_step) #can change steps
        self.n_as    = int(2e3)
        self.a_lower = 1e-6
        self.a_upper = 1

        self.n_before_recomb_as = 1000
        self.n_after_recomb_as  = self.n_as - self.n_before_recomb_as

        self.n_ks    = int(1e3)
        self.k_lower = 1e-3
        self.k_upper = 1

        self.n_small_ks = 250
        self.n_large_ks = self.n_ks - self.n_small_ks

        # Scale factor for recombination.
        
        # https://arxiv.org/abs/astro-ph/9407093: Equation (C-2)

        z_star       = 1000*(self.ombh2/self.h**2)**(-0.027/(1+0.11*np.log(self.ombh2/self.h**2))) # C2
        self.a_star  = 1/(1 + z_star)

        # Matching scale for large-scale and small-scale solutions.
        
        # https://arxiv.org/abs/astro-ph/9407093: in the second paragraph following Equation (8) 
        self.k_matching = 0.08*self.h**3

        # self.a_s = [a_lower, ...........,   a*,  ............, a_upper]
        #
        #            |---n_before_recomb_as---||----n_after_recomb_as----|
        #            |-----------------------n_as------------------------|
                         
        
        # self.n_before_recomb_as number of values of scale factor upto a*
        as_1   = self.get_scales(a_lower = self.a_lower, 
                                 a_upper = self.a_star, 
                                 n_as = self.n_before_recomb_as, 
                                 point_spacing = "log", 
                                 endpoint = False)
        
        self.recomb_ind = self.n_before_recomb_as
        
        # self.n_after_recomb_as number of values of scale factor from a* upto a_upper
        as_2   = self.get_scales(a_lower = self.a_star, 
                                 a_upper = self.a_upper,
                                 n_as = self.n_after_recomb_as, 
                                 point_spacing = "log")
        
        self.a_s = np.concatenate((as_1, as_2))

        # self.ks = [k_lower, .........,  k_matching, ........., k_upper]
        #
        #           |-------n_small_ks-------||--------n_large_ks-------|
        #           |-----------------------n_ks------------------------|

        # self.n_small_ks number of values of scale factor upto k_matching
        ks_1 = self.get_ks(k_lower = self.k_lower, 
                           k_upper = self.k_matching, 
                           n_ks = self.n_small_ks, 
                           point_spacing = "log", 
                           endpoint = False)

        self.k_matching_ind = self.n_small_ks
        
        # self.n_large_ks number of values of scale factor from k_matching upto k_upper
        ks_2 = self.get_ks(k_lower = self.k_matching, 
                           k_upper = self.k_upper, 
                           n_ks = self.n_large_ks, 
                           point_spacing = "log")
        
        self.ks   = np.concatenate((ks_1, ks_2))

        ##########################
        ##### Conformal Time #####
        ##########################

        self.a = self.a_s/self.a_eq
        # Rescaled a_s

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

    def get_ks(self, k_lower = 1e-5, k_upper = 1, n_ks = 1e5, point_spacing = "log", endpoint = True):
        """
        Function for generating an array of k mode points to be used.
        """
    
        if point_spacing == "lin":
            return np.linspace(start = k_lower, stop = k_upper, num = int(n_ks), endpoint = endpoint)
        else: # log spaced a values
            return np.geomspace(start = k_lower, stop = k_upper, num = int(n_ks), endpoint = endpoint)

    def H_a(self, a):
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


    def ug_a(self, a):
        #Eq A-6
        return (a**3 + 2*a**2/9 - 8*a/9 - 16/9 + 16*np.sqrt(a+1)/9)/(a*(a+1))

    def ud_a(self, a):
        #Eq A-7
        return 1/(a*np.sqrt(a+1))

    def A_k(self, k):
        #return np.sqrt(self.As*(k**(n-1))((k_eq/k)**4)((6/(5+2*self.f_nu))**2))
        return np.sqrt(self.As*(k**(self.ns-1))*((k/self.k_eq)**4)*((6/(5+2*self.f_nu))**2)*(k**(-3)))

    def Delta_T(self, a, k):
        return (1 + (2 * self.f_nu * (1 - (0.333*a/(a+1))))/5) * self.A_k(k) * self.ug_a(a)

    def N2_ak(self, a, k):
        Nitem1 = -((20*a + 19)*self.A_k(k)*self.ug_a(a))/(10*(3*a + 4))
        Nitem2 = -(8*a*self.A_k(k))/(3*(3*a + 4))
        Nitem3 = (8*self.A_k(k)*np.log((3*a + 4)/4))/9
        return Nitem1 + Nitem2 + Nitem3
 
    def Phi_bar_ak(self, a, k):
        #Implement G potential
        return (3/4)*((self.k_eq/k)**2)*((a + 1)/a**2)*self.Delta_T(a, k)

    def Psi_bar_ak(self, a, k):
        #Implement G potential
        return -(3/4)*((self.k_eq/k)**2)*((a + 1)/a**2)*(self.Delta_T(a, k) + (8*self.f_nu*self.N2_ak(a, k))/(5*(a+1)))

    def T_k(self, k):
        # Eq A-21
        q       = k/(self.omm * (self.h**2) * np.exp(-2 * self.omb))
        T_item1 = (np.log(1 + 2.34 * q))/(2.34 * q)
        T_item2 = 1 + 3.89 * q + (16.1 * q)**2 + (5.46 * q)**3 + (6.71 * q)**4
        T_k     = T_item1 * (T_item2**(-1/4))
        return T_k

    def Phi_ak(self, a, k):
        #Eq A-22
        alpha_1   = 0.11
        beta      = 1.6
        Phi_T     = self.T_k(k)
        Phi_item  = (1 - Phi_T)*np.exp(-alpha_1 * ((a * k/self.k_eq)**beta)) + Phi_T
        Phi       = self.Phi_bar_ak(a, k) * Phi_item
        return Phi

    def Psi_ak(self, a, k):
        #Eq A-22
        alpha_2   = 0.097
        beta      = 1.6
        Psi_T     = self.T_k(k)
        Psi_item  = (1 - Psi_T)*np.exp(-alpha_2 * ((a * k/self.k_eq)**beta)) + Psi_T
        Psi       = self.Psi_bar_ak(a, k) * Psi_item
        return Psi

    def R_a(self, a):
        return (3 * self.omb * a)/((1 - self.f_nu) * 4 * self.omm)

    def Rdot_a(self, a):
        return self.k_eq * self.R_eq * np.sqrt((1 + a)/2)

    def Rddot_a(self, a):
        return ((self.k_eq**2) * self.R_eq)/4

    def cs_a(self, a):
        return np.sqrt(1/(3 * (1 + self.R_a(a))))

    def rs_a(self, a):
        # Because we have an eta array before, please pay attention that here the input is eta_array
        # Eq 7  
        # But using Eq D-9, the result is what we expected
        R_eq = self.R_a(self.a_eq)
        R    = self.R_a(a) 

        rs_result = (1/self.k_eq) * (2/3) * np.sqrt(6/R_eq) * np.log((np.sqrt(1+R)+np.sqrt(R+R_eq))/(1+np.sqrt(R_eq)))

        return rs_result 

    def Phi0_k(self, k):
        # Eq A-20
        Phi_0 = np.sqrt(self.As * (k**(n-4)))
        
        return Phi_0   

    def J_ak(self, a, k):
        # Eq D-2
        J_result = (np.sqrt(3) * self.Rdot_a(a))/(4 * k * np.sqrt(1 + self.R_a(a)))

        return J_result

    def G_ak(self, a, k):
        # Eq D-4
        G_part1 = (1 + self.R_a(a))**(-1/4)
        G_part2 = 1 - ((1 + self.R_a(a))* self.Psi_ak(a, k))/self.Phi_ak(a, k)
        G_part3 = (3 * self.Rddot_a(a))/(4 * (k**2)) - self.J_ak(a, k)**2

        G_result = G_part1 * (G_part2 + G_part3)

        return G_result

    def I_etak(self, eta_ind, k_ind, Phi, G, rs):
        """
        Equation (D-3) in https://arxiv.org/abs/astro-ph/9407093
        """

        k = self.ks[k_ind]
        
        integrand = Phi[:eta_ind + 1, k_ind]*G[:eta_ind + 1, k_ind]*np.sin(k*rs[eta_ind] - k*rs[:eta_ind + 1])
        
        integral_term = simpson(integrand, self.etas[:eta_ind + 1])
        
        factor = k/np.sqrt(3)

        return factor*integral_term


    def theta_1_hat_large_k_integral_term(self, eta_ind, k_ind, Phi, G, rs):
        """
        Integral term appearing in Equation (D-5) in https://arxiv.org/abs/astro-ph/9407093
        """

        k = self.ks[k_ind]        
        
        integrand = Phi[:eta_ind + 1, k_ind]*G[:eta_ind + 1, k_ind]*np.cos(k*rs[eta_ind] - k*rs[:eta_ind + 1])

        integral_term = simpson(integrand, self.etas[:eta_ind + 1])
        
        factor = k/np.sqrt(3)

        return factor*integral_term

    def theta_0_hat_small_k_integral_term(self, eta_ind, k_ind, Phi, Psi, rs):
        """
        Integral term appearing in Equation (D-6) in https://arxiv.org/abs/astro-ph/9407093
        """

        k = self.ks[k_ind]        
        
        integrand = (Phi[:eta_ind + 1, k_ind] - Psi[:eta_ind + 1, k_ind])*np.sin(k*rs[eta_ind] - k*rs[:eta_ind + 1])

        integral_term = simpson(integrand, self.etas[:eta_ind + 1])
        
        factor = k/np.sqrt(3)

        return factor*integral_term

    def theta_1_hat_small_k_integral_term(self, eta_ind, k_ind, Phi, Psi, rs):
        """
        Integral term appearing in Equation (D-6) in https://arxiv.org/abs/astro-ph/9407093
        """
        #it should be D-7

        k = self.ks[k_ind]        
        
        integrand = (Phi[:eta_ind + 1, k_ind] - Psi[:eta_ind + 1, k_ind])*np.cos(k*rs[eta_ind] - k*rs[:eta_ind + 1])

        integral_term = simpson(integrand, self.etas[:eta_ind + 1])
        
        factor = k/np.sqrt(3)

        return factor*integral_term
    
    
    def compute_Cls(self):

        # These two outputs are the goal of this compute routine!
        ells = 0 # dummy value for now
        cls  = 0 # dummy value for now
        
        ###########################################
        ##### Construct Arrays for Potentials #####
        ###########################################

        self.Phi = np.zeros((self.n_as, self.n_ks))
        self.Psi = np.zeros((self.n_as, self.n_ks))
        
        for a_ind, a in enumerate(self.a_s):
            for k_ind, k in enumerate(self.ks):
                # The a's inputed are NOT rescaled, the rescaling happens within the functions
                self.Phi[a_ind, k_ind] = self.Phi_ak(a, k) 
                self.Psi[a_ind, k_ind] = self.Psi_ak(a, k)

        #########################################
        ##### Construct Array for G(eta, k) #####
        #########################################
        
        G = np.zeros((self.n_as, self.n_ks))
        
        for a_ind, a in enumerate(self.a_s):
            
            R = self.R_a(a)
            Rddot = self.Rddot_a(a)
            
            for k_ind, k in enumerate(self.ks):
                # The a's inputed are NOT rescaled, the rescaling happens within the functions

                J = self.J_ak(a, k)
                
                G[a_ind, k_ind] = ((1+R)**(-0.25))*(1 - (1 + R)*(self.Psi[a_ind, k_ind]/self.Phi[a_ind, k_ind]) + (3/(4*k**2))*Rddot - J**2)

        #########################################
        ###### Construct Array for r_s(eta) #####
        #########################################

        # Double check if cs is even needed/used anywhere? It might not be.
        
        cs = self.cs_a(self.a_s)
        self.rs = self.rs_a(self.a_s)
        
        cs[0] = np.sqrt(1/3)
        self.rs[0] = 0

        #####################################################################
        ###### Construct Monopole and Dipole Solutions at Recombination #####
        #####################################################################

        # Step 1. Compute all the integrals appearing in Eqns. D-1, D-5, D-6, and D-7
        #         Construct integrals as arrays for each value of k-mode

        I_eta_star = np.zeros(self.n_large_ks)
        
        J_eta_star = self.J_ak(self.a_star, self.ks)
        J_0 = self.J_ak(self.a_s[0], self.ks)

        theta_1_hat_large_k_integral = np.zeros(self.n_large_ks)
        
        theta_0_hat_small_k_integral = np.zeros(self.n_small_ks)
        theta_1_hat_small_k_integral = np.zeros(self.n_small_ks)

        for k_ind, k in enumerate(self.ks):

            if k_ind < self.k_matching_ind:
                # k < 0.08*h^3
                theta_0_hat_small_k_integral[k_ind] = self.theta_0_hat_small_k_integral_term(
                    eta_ind = self.recomb_ind, 
                    k_ind   = k_ind, 
                    Phi     = self.Phi, 
                    Psi     = self.Psi,
                    rs      = self.rs)
                
                theta_1_hat_small_k_integral[k_ind] = self.theta_1_hat_small_k_integral_term(
                    eta_ind = self.recomb_ind, 
                    k_ind   = k_ind, 
                    Phi     = self.Phi, 
                    Psi     = self.Psi,
                    rs      = self.rs)

            else:
                k_ind_large = k_ind - self.k_matching_ind - 1
                # k >= 0.08*h^3
                I_eta_star[k_ind_large] = self.I_etak(
                    eta_ind = self.recomb_ind, 
                    k_ind   = k_ind, 
                    Phi     = self.Phi, 
                    G       = G,
                    rs      = self.rs)
                
                theta_1_hat_large_k_integral[k_ind_large] = self.theta_1_hat_large_k_integral_term(
                    eta_ind = self.recomb_ind, 
                    k_ind   = k_ind, 
                    Phi     = self.Phi, 
                    G       = G,
                    rs      = self.rs)
        
        print(I_eta_star)
        # Monopole and Dipole Moments at recombination , for large and small scales, as given by Eqns. D-1, D-5, D-6, and D-7.
        
        # ^Θ0(η*) as appears in Equation (D-1) in https://arxiv.org/abs/astro-ph/9407093
        # FOR SMALL SCALES (LARGE K) ===> all k indices are indexed [self.k_matching_ind:]
        
        theta_0_hat_recomb_large_k = (
            (np.cos(self.ks[self.k_matching_ind:] * self.rs_a(self.a_star)) + J_0[self.k_matching_ind:] * np.sin(self.ks[self.k_matching_ind:] * self.rs_a(self.a_star))) * (0.43*self.Phi_ak(self.a_s[0], self.ks[self.k_matching_ind:]) + self.Phi_ak(self.a_s[0], self.ks[self.k_matching_ind:]))
            + I_eta_star
        ) * (1 + self.R_a(self.a_star))**(-0.25) - self.Phi[self.recomb_ind, self.k_matching_ind:]

        
        # ^Θ1(η*) as appears in Equation (D-5) in https://arxiv.org/abs/astro-ph/9407093
        # FOR SMALL SCALES (LARGE K) ===> all k indices are indexed [self.k_matching_ind:]
        
        theta_1_hat_recomb_large_k = (
            (1 + J_eta_star[self.k_matching_ind:]*J_0[self.k_matching_ind:])*(0.43*self.Phi_ak(self.a_s[0], self.ks[self.k_matching_ind:]) + self.Phi_ak(self.a_s[0], self.ks[self.k_matching_ind:]))*np.sin(self.ks[self.k_matching_ind:] * self.rs_a(self.a_star))
            + (J_eta_star[self.k_matching_ind:] - J_0[self.k_matching_ind:])*(0.43*self.Phi_ak(self.a_s[0], self.ks[self.k_matching_ind:]) + self.Phi_ak(self.a_s[0], self.ks[self.k_matching_ind:]))*np.cos(self.ks[self.k_matching_ind:] * self.rs_a(self.a_star))
            + J_eta_star[self.k_matching_ind:]*I_eta_star
            - theta_1_hat_large_k_integral
        ) * np.sqrt(3) * ((1 + self.R_a(self.a_star))**(-0.75))

        # ^Θ0(η*) as appears in Equation (D-6) in https://arxiv.org/abs/astro-ph/9407093
        # FOR LARGE SCALES (SMALL K) ===> all k indices are indexed [:self.k_matching_ind]
        
        theta_0_hat_recomb_small_k = (
            (0.43*self.Phi_ak(self.a_s[0], self.ks[:self.k_matching_ind]) + self.Phi_ak(self.a_s[0], self.ks[:self.k_matching_ind]))*np.cos(self.ks[:self.k_matching_ind] * self.rs_a(self.a_star))
            + theta_0_hat_small_k_integral
        ) - self.Phi[self.recomb_ind, :self.k_matching_ind]

        # ^Θ1(η*) as appears in Equation (D-1) in https://arxiv.org/abs/astro-ph/9407093
        # FOR LARGE SCALES (SMALL K) ===> all k indices are indexed [:self.k_matching_ind]
        
        theta_1_hat_recomb_small_k = (
            (0.43*self.Phi_ak(self.a_s[0], self.ks[:self.k_matching_ind]) + self.Phi_ak(self.a_s[0], self.ks[:self.k_matching_ind]))*np.sin(self.ks[:self.k_matching_ind] * self.rs_a(self.a_star))
            - theta_1_hat_small_k_integral
        ) * np.sqrt(3) * (1 + self.R_a(self.a_star))**(-0.5)


        # Joining Large and Small Scales at k_matching = 0.08*h^3
        theta_0_hat_recomb = np.concatenate((theta_0_hat_recomb_small_k, theta_0_hat_recomb_large_k))
        theta_1_hat_recomb = np.concatenate((theta_1_hat_recomb_small_k, theta_1_hat_recomb_large_k))

        # Returning  ^Θ0(η*) and  ^Θ1(η*) for now, for testing purposes.
        # Thus far, Steps 1 and 2 of Appendix D in https://arxiv.org/abs/astro-ph/9407093.
        
        return theta_0_hat_recomb, theta_1_hat_recomb



