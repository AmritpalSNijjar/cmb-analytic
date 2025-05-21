import numpy as np
from scipy.integrate import quad
from scipy.integrate import simpson
from scipy.integrate import cumulative_simpson

class analytic_CMB:

    def __init__(self, ombh2, omch2, H0, As, ns, tau, noL_CDM = False):

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
        #self.k_eq  = np.sqrt(2*(self.om0h2/self.h**2)*(self.H0**2)/self.a_eq) # HU/S
        self.k_eq = 0.073*self.om0h2 # dodelson 7.39, also in HU/S in the paragraph right after Eqn. D9

        #TODO: a_*, eta_*, but not as actual values. we need the closes value of a in self.a_s to serve
        #      as an approximate a_*, similarly the closes value of eta in self.etas for eta_*

        # Setting scale factor values and k mode values

        self.n_as    = int(2e3) # TODO: allow for tuning of these parameters
        self.a_lower = 1e-6
        self.a_upper = 1
        
        self.n_ks    = int(1e3)
        self.k_lower = 1e-3
        self.k_upper = 1

        z_star    = 1000*(self.ombh2/self.h**2)**(-0.027/(1+0.11*np.log(self.ombh2/self.h**2))) # C2
        self.a_star = 1/(1 + z_star)

        as_1   = self.get_scales(a_lower = self.a_lower, a_upper = 1e-3, n_as = self.n_as//2, point_spacing = "log", endpoint = False) 
        as_2   = self.get_scales(a_lower = 1e-3, a_upper = self.a_upper, n_as = self.n_as//2, point_spacing = "log")


        self.a_s = np.concatenate((as_1, as_2))

        #self.a_s = self.get_scales(a_lower = self.a_lower, a_upper = self.a_upper, n_as = self.n_as, point_spacing = "log")
        self.a_s_rescaled = self.a_s/self.a_eq
        
        self.ks   = self.get_ks(k_lower = self.k_lower, k_upper = self.k_upper, n_ks = self.n_ks, point_spacing = "lin")

        self.etas = np.zeros_like(self.a_s)
        
        #if not the next line... errors when integrating!
        self.etas[0] = 0
        
        for i in range(1, self.n_as):
            self.etas[i] = self.etas[i - 1] + self.eta(self.a_s[i - 1], self.a_s[i])

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

        val = np.log(1+2.34*q)/(2.34*q)*(1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**-0.25
        
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
        Equation (7) in https://arxiv.org/abs/astro-ph/9407093
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
        Equation (D3) in https://arxiv.org/abs/astro-ph/9407093
        """

        k = self.ks[k_ind]
        
        integrand = Phi[:eta_ind + 1, k_ind]*G[:eta_ind + 1, k_ind]*np.sin(k*self.rs_a(self.a_s[eta_ind]) - k*self.rs_a(self.a_s[:eta_ind + 1]))
        
        integral_term = simpson(integrand, self.etas[:eta_ind + 1])
        
        factor = k/np.sqrt(3)

        return factor*integral_term
    
    def compute_Cls(self, inp_k):

        ells = 0 # dummy value for now
        cls  = 0 # dummy value for now

        # Construct Arrays for Potentials

        self.Phi = np.zeros((self.n_as, self.n_ks))
        self.Psi = np.zeros((self.n_as, self.n_ks))
        
        for a_ind, a in enumerate(self.a_s):
            for k_ind, k in enumerate(self.ks):
                # The a's inputed are NOT rescaled, the rescaling happens within the functions
                self.Phi[a_ind, k_ind] = self.Phi_ak(a, k) 
                self.Psi[a_ind, k_ind] = self.Psi_ak(a, k)

        # Use A19 to set the initial conditions Phi[0, k] and Psi[0, k]

        #for a_ind, a in enunmerate(self.a_s):
        # TODO: Vectorize this scaling instead of a for loop
        
        #self.Phi[a_ind, :] = self.Phi_0k(self.ks)*self.Phi[a_ind, :]/self.Phi[0, :]
        # Using the relation in Eqn. A-19
        #self.Psi[a_ind, :] = -0.86*self.Phi[0, k_ind]*self.Psi[a_ind, :]/self.Psi[0, :] # Using the relation in Eqn. A-19

        # Construct Array for G(eta, k)
        
        G = np.zeros((self.n_as, self.n_ks))
        
        for a_ind, a in enumerate(self.a_s):
            
            R = self.R_a(a)
            Rddot = self.Rddot_a()
            
            for k_ind, k in enumerate(self.ks):
                # The a's inputed are NOT rescaled, the rescaling happens within the functions

                J = self.J_ak(a, k)
                
                G[a_ind, k_ind] = ((1+R)**(-0.25))*(1 - (1 + R)*(self.Psi[a_ind, k_ind]/self.Phi[a_ind, k_ind]) + (3/(4*k**2))*Rddot - J**2)

        # Construct Array for r_s(eta)

        cs = np.zeros(self.n_as)
        rs = np.zeros(self.n_as)
        
        cs = self.cs_a(self.a_s)
        rs = self.rs_a(self.a_s)
        cs[0] = np.sqrt(1/3)
        rs[0] = 0
        #rs = np.concatenate(([0], cumulative_simpson(cs, x=self.etas)))


        # fixed-k test
        k_fixed_ind = inp_k
        k_fixed = self.ks[k_fixed_ind]

        I_k_fixed = np.zeros(self.n_as)

        G_k_fixed = G[:, k_fixed_ind]

        Phi_k_fixed = self.Phi[:, k_fixed_ind]
        Psi_k_fixed = self.Psi[:, k_fixed_ind]

        for a_ind, a in enumerate(self.a_s):

            I_k_fixed[a_ind] = self.I_etak(a_ind, k_fixed_ind, self.Phi, G)

        theta_naught_hat = np.zeros(self.n_as)

        theta_hat_naught_k_fixed = 0.43*self.Phi_ak(self.a_s[0], k_fixed)

        int_term = np.zeros(self.n_as)
        for a_ind, a in enumerate(self.a_s):

            integrand = Phi_k_fixed[:a_ind + 1] - Psi_k_fixed[:a_ind + 1]
            integrand *= np.cos(k_fixed*self.rs_a(self.a_s[a_ind]) - k_fixed*self.rs_a(self.a_s[:a_ind + 1]))
            
            int_term[a_ind] = simpson(integrand, self.etas[:a_ind + 1])
        

        for a_ind, a in enumerate(self.a_s):
            theta_naught_hat[a_ind] = (theta_hat_naught_k_fixed + self.Phi_ak(self.a_s[0], k_fixed))*(np.cos(k_fixed*self.rs_a(a)) + np.sin(k_fixed*self.rs_a(a))*self.J_ak(self.a_s[0], k_fixed))
            theta_naught_hat[a_ind] += I_k_fixed[a_ind]

            theta_naught_hat[a_ind] -= self.Phi[a_ind, k_fixed_ind]

            theta_naught_hat[a_ind] *= (1+self.R_a(self.a_s[a_ind]))**-0.25
        
        return theta_naught_hat


        




        
    











        

    