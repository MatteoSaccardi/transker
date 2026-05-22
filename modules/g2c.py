import mpmath
import numpy

def levy_np(x, mu, c):
    return numpy.sqrt(c/(2*numpy.pi)) * numpy.exp(-c/(2*(x-mu))) / (x-mu)**1.5

def levy_mp(x, mu, c):
    return mpmath.sqrt(c/(2*mpmath.pi)) * mpmath.exp(-c/(2*(x-mu))) / mpmath.power(x-mu,mpmath.mpf('1.5'))

class CauchySmearing:

    def __init__(self, sigmas, rho_sigmas, uppers, lowers, use_mp=True):
        self.sigmas = sigmas
        self.rho_sigmas = rho_sigmas
        self.uppers = uppers
        self.lowers = lowers
        self.use_mp = use_mp
        self.epsilon = None

    def __call__(self, epsilon):
        return self.rho_cauchy(epsilon)
    
    def levy(self, x, mu, c):
        if self.use_mp:
            return levy_mp(x,mu,c)
        return levy_np(x,mu,c)
    
    def K_cauchy(self, sigma, epsilon):
        return 2 * sigma * self.levy(sigma**2, 0, epsilon**2)
        
    def rho_cauchy(self, epsilon):
        r'''
        Compute 
            \int_0^\infty d\omega \rho(\omega) \delta^c_\epsilon(\omega,E)
        where \delta^c is a Cauchy smearing function, from
            \int_0^\infty d\sigma K(\epsilon, \sigma) \rho^g_\sigma(E)
        where \rho^g_\sigma is a spectral function smeared with a Gaussian 
        kernel \delta^g centered around E and with width \sigma,
        while K is the kernel defined in K_cauchy
        NOTE: E does not enter the definition of K_cauchy
        '''
        if self.epsilon != epsilon:
            self.epsilon = epsilon
            Kvals = [ self.K_cauchy(sigma, epsilon) for sigma in self.sigmas ]
            self.Kvals = Kvals
        else:
            Kvals = self.Kvals
        integrand = [ Kval * self.rho_sigmas[i] for i, Kval in enumerate(Kvals) ]
        integral = numpy.trapezoid(integrand, self.sigmas)

        uppers = numpy.array(self.uppers)
        lowers = numpy.array(self.lowers)

        prod_lower = numpy.minimum(Kvals * lowers, Kvals * uppers)
        prod_upper = numpy.maximum(Kvals * lowers, Kvals * uppers)

        lower_bound_integral = numpy.trapezoid(prod_lower, self.sigmas)
        upper_bound_integral = numpy.trapezoid(prod_upper, self.sigmas)

        return integral, upper_bound_integral, lower_bound_integral

    
class CauchySmearing_x:

    def __init__(self, xs, rho_sigmas, uppers, lowers, use_mp=True):
        self.xs = xs
        self.rho_sigmas = rho_sigmas
        self.uppers = uppers
        self.lowers = lowers
        self.use_mp = use_mp
        self.Kvals = numpy.array([self.K_cauchy_x(x) for x in xs])

    def __call__(self):
        return self.rho_cauchy()
    
    def levy(self, x, mu, c):
        if self.use_mp:
            return levy_mp(x,mu,c)
        return levy_np(x,mu,c)
    
    def K_cauchy_x(self, x):
        return 2 * x * self.levy(x**2, 0, 1)
        
    def rho_cauchy(self):
        r'''
        Compute 
            \int_0^\infty d\omega \rho(\omega) \delta^c_\epsilon(\omega,E)
        where \delta^c is a Cauchy smearing function, from
            \int_0^\infty dx K(x) \rho^g_{\epsilon x}(E)
        where \rho^g_\sigma is a spectral function smeared with a Gaussian 
        kernel \delta^g centered around E and with width \epsilon x,
        while K is the kernel defined in K_cauchy_x
        NOTE: E and \epsilon do not enter the definition of K_cauchy_x
        '''
        Kvals = self.Kvals
        integrand = [ Kval * self.rho_sigmas[i] for i, Kval in enumerate(Kvals) ]
        integral = numpy.trapezoid(integrand, self.xs)

        uppers = numpy.array(self.uppers)
        lowers = numpy.array(self.lowers)

        prod_lower = numpy.minimum(Kvals * lowers, Kvals * uppers)
        prod_upper = numpy.maximum(Kvals * lowers, Kvals * uppers)

        lower_bound_integral = numpy.trapezoid(prod_lower, self.xs)
        upper_bound_integral = numpy.trapezoid(prod_upper, self.xs)

        return integral, upper_bound_integral, lower_bound_integral