import numpy as np
from scipy.integrate import quad
from scipy.integrate import simpson
from scipy.integrate import cumulative_simpson

class analytic_CMB:

    def __init__(self, ombh2, omch2, H0, As, ns, tau, noL_CDM = False):
        
        ##############################
        ##### Setting parameters #####
        ##############################
        
        # LCDM parameters
        self.ombh2 = ombh2
        self.H0    = H0
        self.As    = As
        self.ns    = ns
        self.tau   = tau
        
        self.h     = H0/100
        
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


        self.f_nu  = 0.405

        # Dodelson Equation (7.39), also in https://arxiv.org/abs/astro-ph/9407093 in the paragraph right after Equation (D-9)
        self.k_eq = 0.073*self.om0h2

        #########################################################
        ##### Setting scale factor values and k mode values #####
        #########################################################
        
        # TODO: allow for tuning of these parameters? Currently, for testing, they are hard-coded at some arbitrary values.
        
        self.n_as    = int(2e3)      # Number of scale factor values.
        self.a_lower = 1e-6          # Lower limit of scale factor.
        self.a_upper = 1             # Upper limit of scale factor.
        
        self.n_ks    = int(1e3)      # Number of k mode values.
        self.k_lower = 1e-3          # Lower limit of k mode.
        self.k_upper = 1             # Upper limit of k mode.

        # Scale factor for recombination.
        
        # https://arxiv.org/abs/astro-ph/9407093: Equation (C-2)
        z_star    = 1000*(self.ombh2/self.h**2)**(-0.027/(1+0.11*np.log(self.ombh2/self.h**2))) 
        self.a_star = 1/(1 + z_star)

        # Matching scale for large-scale and small-scale solutions.
        
        # https://arxiv.org/abs/astro-ph/9407093: in the second paragraph following Equation (8) 
        self.k_matching = 0.08*self.h**3

        # self.a_s = [a_lower, ..., a*, ..., a_upper]
        #
        #            |---n_as//2---||----n_as//2----|  (n_as//2 arbitrarily chosen as the cutoff)
        #            |-------------n_as-------------|
                         
        self.n_as = 2*self.n_as//2
        
        # n_as//2 values of scale factor upto a*
        as_1   = self.get_scales(a_lower = self.a_lower, 
                                 a_upper = self.a_star, 
                                 n_as = self.n_as//2, 
                                 point_spacing = "log", 
                                 endpoint = False)
        
        self.recomb_ind = self.n_as//2
        
        # n_as//2 values of scale factor from a* upto a_upper
        as_2   = self.get_scales(a_lower = self.a_star, 
                                 a_upper = self.a_upper,
                                 n_as = self.n_as//2, 
                                 point_spacing = "log")
        
        self.a_s = np.concatenate((as_1, as_2))

        # self.ks = [k_lower, ..., k_matching, ..., ..., k_upper]
        #
        #           |-----n_ks//4-----||--------3*n_ks//4-------| (n_ks//4 arbitrarily chosen as the cutoff)
        #           |-------------------n_ks--------------------|

        self.n_as = 4*self.n_as//4

        # n_ks//4 values of scale factor upto k_matching
        ks_1 = self.get_ks(k_lower = self.k_lower, 
                           k_upper = self.k_matching, 
                           n_ks = self.n_ks//4, 
                           point_spacing = "log", 
                           endpoint = False)

        self.k_matching_ind = self.n_ks//4
        
        # 3*n_ks//4 values of scale factor from k_matching upto k_upper
        ks_2 = self.get_ks(k_lower = self.k_matching, 
                           k_upper = self.k_upper, 
                           n_ks = 3*self.n_ks//4, 
                           point_spacing = "log")
        
        self.ks   = np.concatenate((ks_1, ks_2))

        ##########################
        ##### Conformal Time #####
        ##########################

        self.etas = np.zeros_like(self.a_s)
        
        #if not the next line... errors when integrating!
        self.etas[0] = 0
        
        for i in range(1, self.n_as):
            self.etas[i] = self.etas[i - 1] + self.eta(self.a_s[i - 1], self.a_s[i])

        self.eta_star = self.etas[self.recomb_ind]

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

    def eta_integrand(self, a):
        """
        Integrand for calculating conformal time eta of a given scale factor a_input: 1/(H*a_input^2).
        """
        
        H = (self.H0)*np.sqrt((self.om0h2/self.h**2)*a**-3 + (self.omrh2/self.h**2)*a**-4 + (self.omLh2/self.h**2))
        return 1/(H*a**2)

    def eta(self, a_lower, a_upper):
        """
        Conformal time eta of a given scale factor a_input: The ingegral of da/(H*a^2) from 0 -> a_input.
        """
        # Lower bound set to very small number instead of zero.
        return quad(self.eta_integrand, a_lower, a_upper)[0]

    def UG_a(self, a):
        """
        Equation (A-6) in https://arxiv.org/abs/astro-ph/9407093
        """
        
        a_rescale = a/self.a_eq 
        
        return (a_rescale**3 + (2/9)*a_rescale**2 - (8/9)*a_rescale - 16/9 + (16/9)*np.sqrt(a_rescale + 1) )/(a_rescale*(a_rescale + 1))

    def A_k(self, k):
        """
        A(k) found in Equation (A-20) in https://arxiv.org/abs/astro-ph/9407093
        """

        val = np.sqrt(self.As*(k**(self.ns - 1))*((k/self.k_eq)**4)*(((5/6)*(1+(2/5)*self.f_nu))**-2)*k**-3)
        return val

    def N2_ak(self, a, k, mode = "approximate"):
        """
        Equation (A-12) in https://arxiv.org/abs/astro-ph/9407093
        """
        
        a_rescale = a/self.a_eq 

        A = self.A_k(k)
        UG = self.UG_a(a) # NOTE: Input is a_scaled to today as we have currently implemented things.

        if mode == "approximate":
            val = -(1/10)*((20*a_rescale+19)/(3*a_rescale+4))*A*UG - (8/3)*(a_rescale/(3*a_rescale+4))*A + (8/9)*np.log((3*a_rescale+4)/4)*A
        elif mode == "exact":
            val = 0 # In case we want to allow for Equation (A-11) to be used instead in the future.

        H = (self.H0)*np.sqrt((self.om0h2/self.h**2)*a**-3 + (self.omrh2/self.h**2)*a**-4 + (self.omLh2/self.h**2))
        
        val *= np.cos((0.5*k)/(a*H))

        return val

    def Delta_T_ak(self, a, k):
        """
        Equation (A-16) in https://arxiv.org/abs/astro-ph/9407093
        """
        
        a_rescale = a/self.a_eq 

        A = self.A_k(k)
        UG = self.UG_a(a) # NOTE: Input is a_scaled to today as we have currently implemented things.
        
        val = (1+(2/5)*self.f_nu*(1-0.333*a_rescale/(a_rescale + 1)))*A*UG
        
        return val

    def Phi_bar_ak(self, a, k):
        """
        Equation (A-17a) in https://arxiv.org/abs/astro-ph/9407093
        """
        
        a_rescale = a/self.a_eq 

        Delta_T = self.Delta_T_ak(a, k) # NOTE: Input is a_scaled to today as we have currently implemented things.
        
        val = (3/4)*((self.k_eq/k)**2)*((a_rescale + 1)/a_rescale**2)*Delta_T
        return val

    def Psi_bar_ak(self, a, k):
        """
        Equation (A-17b) in https://arxiv.org/abs/astro-ph/9407093
        """
        
        a_rescale = a/self.a_eq 

        Delta_T = self.Delta_T_ak(a, k) # NOTE: Input is a_scaled to today as we have currently implemented things.
        N2      = self.N2_ak(a, k, mode = "approximate") # NOTE: Input is a_scaled to today as we have currently implemented things.
        
        val = -(3/4)*((self.k_eq/k)**2)*((a_rescale + 1)/a_rescale**2)*(Delta_T + (8/5)*self.f_nu*N2/(a_rescale + 1))
        return val

    def T_k(self, k):
        """
        Equation (A-21) in https://arxiv.org/abs/astro-ph/9407093
        """

        q = k/(self.om0h2*np.exp(-2*self.ombh2/self.h**2))

        val = np.log(1+2.34*q)/(2.34*q)*(1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)
        
        return val
    
    def Phi_ak(self, a, k):
        """
        Equation (A-22a) in https://arxiv.org/abs/astro-ph/9407093
        """

        a_rescale = a/self.a_eq

        alpha_1 = 0.11
        beta    = 1.6
        
        Phi_bar = self.Phi_bar_ak(a, k) # NOTE: Input is a_scaled to today as we have currently implemented things.
        Tk      = self.T_k(k)

        val = Phi_bar*(Tk + (1 - Tk)*np.exp(-alpha_1*(a_rescale*k/self.k_eq)**beta))
        
        return val

    def Psi_ak(self, a, k):
        """
        Equation (A-22b) in https://arxiv.org/abs/astro-ph/9407093
        """

        a_rescale = a/self.a_eq

        alpha_2 = 0.097
        beta    = 1.6
        
        Psi_bar = self.Psi_bar_ak(a, k) # NOTE: Input is a_scaled to today as we have currently implemented things.
        Tk      = self.T_k(k)

        val = Psi_bar*(Tk + (1 - Tk)*np.exp(-alpha_2*(a_rescale*k/self.k_eq)**beta))
        
        return val

    def R_a(self, a):
        """
        Equation (D-8a) in https://arxiv.org/abs/astro-ph/9407093
        """

        a_rescale = a/self.a_eq 
        
        val = (1/(1-self.f_nu))*(3/4)*(self.ombh2/self.om0h2)*a_rescale
        return val

    def Rdot_a(self, a):
        """
        Equation (D-8b) in https://arxiv.org/abs/astro-ph/9407093
        """

        a_rescale = a/self.a_eq 

        R_eq = self.R_a(self.a_eq) # NOTE: Input is a_scaled to today as we have currently implemented things.

        val = self.k_eq*np.sqrt(1 + a_rescale)*R_eq/np.sqrt(2)

        return val

    def Rddot_a(self):
        """
        Equation (D-8c) in https://arxiv.org/abs/astro-ph/9407093
        """
        
        # Note: This is a constant.
        
        R_eq = self.R_a(self.a_eq)
        val = (self.k_eq**2)*(R_eq)/4
        
        return val

    def J_ak(self, a, k):
        """
        Equation (D-2) in https://arxiv.org/abs/astro-ph/9407093
        """

        a_rescale = a/self.a_eq

        val = (np.sqrt(3)/(4*k))*(self.Rdot_a(a)/np.sqrt(1+self.R_a(a)))

        return val
        
    def cs_a(self, a):
        """
        Equation (D-6) in https://arxiv.org/abs/astro-ph/9407093
        """
        
        R = self.R_a(a) # NOTE: Input is a_scaled to today as we have currently implemented things.
        
        val = np.sqrt((1/3)*(1/(1+R)))
        
        return val

    def rs_a(self, a):
        """
        Equation (D-9) in https://arxiv.org/abs/astro-ph/9407093
        """
    
        R_eq = self.R_a(self.a_eq)
        R = self.R_a(a)
        
        val = (2/3) * np.sqrt(6/R_eq) * np.log((np.sqrt(1+R)+np.sqrt(R+R_eq))/(1+np.sqrt(R_eq))) * (1/self.k_eq)
        
        return val
      
    def Phi_0k(self, k): # Might not need this
        """
        Phi(0, k) found in Equation (A-20) in https://arxiv.org/abs/astro-ph/9407093
        """

        rhs = self.As*k**(self.ns-1) # Right Hand Side of the equation
        val = np.sqrt(rhs/(k**3))
        
        return val

    def I_etak(self, eta_ind, k_ind, Phi, G):
        """
        Equation (D-3) in https://arxiv.org/abs/astro-ph/9407093
        """

        k = self.ks[k_ind]
        
        integrand = Phi[:eta_ind + 1, k_ind]*G[:eta_ind + 1, k_ind]*np.sin(k*self.rs_a(self.a_s[eta_ind]) - k*self.rs_a(self.a_s[:eta_ind + 1]))
        
        integral_term = simpson(integrand, self.etas[:eta_ind + 1])
        
        factor = k/np.sqrt(3)

        return factor*integral_term

    def theta_1_hat_large_k_integral_term(self, eta_ind, k_ind, Phi, G):
        """
        Integral term appearing in Equation (D-5) in https://arxiv.org/abs/astro-ph/9407093
        """

        k = self.ks[k_ind]        
        
        integrand = Phi[:eta_ind + 1, k_ind]*G[:eta_ind + 1, k_ind]*np.cos(k*self.rs_a(self.a_s[eta_ind]) - k*self.rs_a(self.a_s[:eta_ind + 1]))

        integral_term = simpson(integrand, self.etas[:eta_ind + 1])
        
        factor = k/np.sqrt(3)

        return factor*integral_term

    def theta_0_hat_small_k_integral_term(self, eta_ind, k_ind, Phi, Psi):
        """
        Integral term appearing in Equation (D-6) in https://arxiv.org/abs/astro-ph/9407093
        """

        k = self.ks[k_ind]        
        
        integrand = (Phi[:eta_ind + 1, k_ind] - Psi[:eta_ind + 1, k_ind])*np.sin(k*self.rs_a(self.a_s[eta_ind]) - k*self.rs_a(self.a_s[:eta_ind + 1]))

        integral_term = simpson(integrand, self.etas[:eta_ind + 1])
        
        factor = k/np.sqrt(3)

        return factor*integral_term

    def theta_1_hat_small_k_integral_term(self, eta_ind, k_ind, Phi, Psi):
        """
        Integral term appearing in Equation (D-6) in https://arxiv.org/abs/astro-ph/9407093
        """

        k = self.ks[k_ind]        
        
        integrand = (Phi[:eta_ind + 1, k_ind] - Psi[:eta_ind + 1, k_ind])*np.cos(k*self.rs_a(self.a_s[eta_ind]) - k*self.rs_a(self.a_s[:eta_ind + 1]))

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
            Rddot = self.Rddot_a()
            
            for k_ind, k in enumerate(self.ks):
                # The a's inputed are NOT rescaled, the rescaling happens within the functions

                J = self.J_ak(a, k)
                
                G[a_ind, k_ind] = ((1+R)**(-0.25))*(1 - (1 + R)*(self.Psi[a_ind, k_ind]/self.Phi[a_ind, k_ind]) + (3/(4*k**2))*Rddot - J**2)

        #########################################
        ###### Construct Array for r_s(eta) #####
        #########################################

        # Double check if cs is even needed/used anywhere? It might not be.
        
        cs = self.cs_a(self.a_s)
        rs = self.rs_a(self.a_s)
        
        cs[0] = np.sqrt(1/3)
        rs[0] = 0

        #####################################################################
        ###### Construct Monopole and Dipole Solutions at Recombination #####
        #####################################################################

        I_eta_star = np.zeros(3*self.n_ks//4)
        
        J_eta_star = self.J_ak(self.a_star, self.ks)
        J_0 = self.J_ak(self.a_s[0], self.ks)

        theta_1_hat_large_k_integral = np.zeros(3*self.n_ks//4)
        
        theta_0_hat_small_k_integral = np.zeros(self.n_ks//4)
        theta_1_hat_small_k_integral = np.zeros(self.n_ks//4) # See Lines 93-113 for where n_ks//4 comes from.

        for k_ind, k in enumerate(self.ks):

            if k_ind < self.k_matching_ind:
                # k < 0.08*h^3
                theta_0_hat_small_k_integral = self.theta_0_hat_small_k_integral_term(
                    eta_ind = self.recomb_ind, 
                    k_ind = k_ind, 
                    Phi = self.Phi, 
                    Psi = self.Psi)
                
                theta_1_hat_small_k_integral = self.theta_1_hat_small_k_integral_term(
                    eta_ind = self.recomb_ind, 
                    k_ind = k_ind, 
                    Phi = self.Phi, 
                    Psi = self.Psi)

            else:
                k_ind_large = k_ind - self.k_matching_ind - 1
                # k > 0.08*h^3
                I_eta_star[k_ind_large] = self.I_etak(
                    eta_ind = self.recomb_ind, 
                    k_ind = k_ind, 
                    Phi = self.Phi, 
                    G = G)
                
                theta_1_hat_large_k_integral[k_ind_large] = self.theta_1_hat_large_k_integral_term(
                    eta_ind = self.recomb_ind, 
                    k_ind = k_ind, 
                    Phi = self.Phi, 
                    G = G)


        # Monopole and Dipole Moments at recombination , for large and small scales, as given by D-1, D-5, D-6, and D-7.
        
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
        ) * np.sqrt(3) * (1 + self.R_a(self.a_star))**(-0.75)

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
