import mpmath
import numpy
import scipy

class c2g:

    def __init__(self, ws, rho_epsilons, uppers, lowers, epsilon, use_mp=True):
        self.ws = ws
        self.rho_epsilons = rho_epsilons
        self.uppers = uppers
        self.lowers = lowers
        self.epsilon = epsilon
        self.use_mp = use_mp
        self.E = None
        self.sigma = None

    def __call__(self, E, sigma):
        return self.rho_gauss(E, sigma)
    
    def K_gauss(self, w, E, sigma):
        if self.use_mp:
            z = ( self.epsilon - 1j * (w-E) ) / mpmath.sqrt(2*sigma**2)
            return 1 / mpmath.sqrt(2*mpmath.pi*sigma**2) * ( mpmath.exp(z**2) * (1+mpmath.erf(z)) ).real
        else:
            z = ( self.epsilon - 1j * (w-E) ) / numpy.sqrt(2*sigma**2)
            return 1 / numpy.sqrt(2*numpy.pi*sigma**2) * ( numpy.exp(z**2) * (1+scipy.special.erf(z)) ).real
        
    def rho_gauss(self, E, sigma):
        r'''
        Compute the spectral function smeared with a gaussian kernel of width sigma and center E
            \int_0^\infty d\omega \rho(\omega) \delta^g_\sigma(\omega,E)
        from a spectral function smeared with a Cauchy kernel of width self.epsilon,
            \int_0^\infty d\omega \rho(\omega) \delta^c_\epsilon(\omega,\omega')
        from the transition kernel K defined in self.K_gauss as
            Re[ exp(z^2) * (1+erf(z)) ] / sqrt(2*pi*sigma^2) 
        for 
            z = (epsilon-i(w-E))/sqrt(2*sigma^2)
        '''
        if self.E != E or self.sigma != sigma:
            self.E = E
            self.sigma = sigma
            Kvals = [ self.K_gauss(w, E, sigma) for w in self.ws ]
            self.Kvals = Kvals
        else:
            Kvals = self.Kvals
        integrand = [ Kval * self.rho_epsilons[i] for i, Kval in enumerate(Kvals) ]
        integral = numpy.trapezoid(integrand, self.ws)

        uppers = numpy.array(self.uppers)
        lowers = numpy.array(self.lowers)

        prod_lower = numpy.minimum(Kvals * lowers, Kvals * uppers)
        prod_upper = numpy.maximum(Kvals * lowers, Kvals * uppers)

        lower_bound_integral = numpy.trapezoid(prod_lower, self.ws)
        upper_bound_integral = numpy.trapezoid(prod_upper, self.ws)

        return integral, upper_bound_integral, lower_bound_integral

    