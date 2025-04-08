import numpy as np
from scipy.integrate import quad

class analytic_CMB:

    def __init__(self, ombh2, omch2, H0, As, ns, tau):

        # LCDM parameters
        self.ombh2 = ombh2
        self.omch2 = omch2
        self.H0    = H0
        self.As    = As
        self.ns    = ns
        self.tau   = tau

        self.h     = H0/100
        self.om0h2 = self.ombh2 + self.omch2
        self.omrh2 = self.a_eq * (self.om0h2/self.h**2)
        self.omLh2 = self.h**2 - self.om0h2 - self.omrh2 

        self.a_eq  = 4.15e-5/self.om0h2 # Dodelson Eq. (2.87)

        self.as   = self.get_scales(a_lower = 1e-5, a_upper = 1, n_as = 1e5, point_spacing = "log") 
        # eventually, we can make params like a_lower, a_upper, etc. tunable
        # by using a config dict passed to the constructor

        self.etas = np.array([self.eta(a) for a in self.as])

        self.f_nu  = 0.405
        self.k_eq  = np.sqrt(2*(self.om0h2/self.h**2)*(self.H0**2)*1/a_eq)

    def get_scales(self, a_lower = 1e-5, a_upper = 1, n_as = 1e5, point_spacing = "log"):

        if point_spacing == "log":
            return np.geomspace(start = a_lower, stop = a_upper, num = n_as)
        elif point_spacing == "lin":
            return np.linspace(start = a_lower, stop = a_upper, num = n_as)

    def eta_integrand(self, a):
        H = (self.H0)*np.sqrt((self.om0h2*a**-3 + self.omrh2*a**-4 + self.omLh2)/self.h**2)
        return (H*a**2)**-1

    def eta(self, a):
        return quad(self.eta_integrand, 0, a)[0]

    def UG_a(self, a):
        """
        Equation (A-6) in https://arxiv.org/abs/astro-ph/9407093
        """
        
        a_rescale = a/self.a_eq 
        
        return (a_rescale**3 + (2/9)*a_rescale**2 - (8/9)*a_rescale - 16/9 + (16/9)*np.sqrt(a_rescale + 1) )/(a_rescale*(a_rescale + 1))

    def A_k(self, k):
        """
        As found in Equation (A-20) in https://arxiv.org/abs/astro-ph/9407093
        """

        val = np.sqrt(self.As*(k**(self.ns - 1))*((k/self.k_eq)**4)*(((5/6)*(1+(2/5)*f_nu))**-2)*k**-3)
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

        return val

    def Delta_T_ak(self, a, k):
        """
        Equation (A-16) in https://arxiv.org/abs/astro-ph/9407093
        """
        
        a_rescale = a/self.a_eq 

        A = self.A_k(k)
        UG = self.UG_a(a) # NOTE: Input is a_scaled to today as we have currently implemented things.
        
        val = (1+(2/5)*f_nu*(1-0.333*a_rescale/(a_rescale + 1)))*A*UG
        
        return val


    def Phi_ak(self, a, k):
        """
        Equation (A-17a) in https://arxiv.org/abs/astro-ph/9407093
        """
        
        a_rescale = a/self.a_eq 

        Delta_T = self.Delta_T_ak(a, k) # NOTE: Input is a_scaled to today as we have currently implemented things.
        
        val = (3/4)*((k_eq/k)**2)*((a_rescale + 1)/a_rescale**2)*Delta_T
        return val

    def Psi_ak(self, a, k):
        """
        Equation (A-17b) in https://arxiv.org/abs/astro-ph/9407093
        """
        
        a_rescale = a/self.a_eq 

        Delta_T = self.Delta_T_ak(a, k) # NOTE: Input is a_scaled to today as we have currently implemented things.
        N2      = self.N2_ak(a, k, mode = "approximate") # NOTE: Input is a_scaled to today as we have currently implemented things.
        
        val = (3/4)*((k_eq/k)**2)*((a_rescale + 1)/a_rescale**2)*(Delta_T + (8/5)*self.f_nu*N2/(a_rescale + 1))
        return val



        
    














        

    