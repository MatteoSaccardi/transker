#################################################################################
#
# g2c.py: Gaussian-to-Cauchy (Levy) RK transition utilities
# Copyright (C) 2026 Matteo Saccardi
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
#
#################################################################################

"""Gaussian-to-Cauchy (Levy) RK transition classes based on Levy kernels."""

import numpy

from modules.kernels import levy_mp, levy_np

class CauchySmearing:
    """
    Propagate bounded Gaussian-width data to a target Cauchy width (Levy).

    Parameters
    ----------
    sigmas : array_like
        Gaussian widths at which the input data are sampled.
    rho_sigmas : array_like
        Central Gaussian-smeared data values.
    uppers, lowers : array_like
        Pointwise upper and lower admissible data values.
    use_mp : bool, optional
        If True, evaluate the Levy kernel with mpmath scalar arithmetic.
    """

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
        """Evaluate the Levy distribution with the configured backend."""
        if self.use_mp:
            return levy_mp(x,mu,c)
        return levy_np(x,mu,c)
    
    def K_cauchy(self, sigma, epsilon):
        """Evaluate the Gaussian-to-Cauchy transition kernel."""
        return 2 * sigma * self.levy(sigma**2, 0, epsilon**2)
        
    def rho_cauchy(self, epsilon):
        """
        Propagate bounded Gaussian-width data to Cauchy width `epsilon`.

        Parameters
        ----------
        epsilon : float
            Target Cauchy width.

        Returns
        -------
        tuple
            `(central, upper, lower)` propagated by trapezoidal integration.
        """
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
    """
    Dimensionless Gaussian-to-Cauchy transition on `x = sigma / epsilon`.

    Parameters
    ----------
    xs : array_like
        Dimensionless width grid.
    rho_sigmas : array_like
        Central Gaussian-smeared data values evaluated at `sigma = epsilon*x`.
    uppers, lowers : array_like
        Pointwise upper and lower admissible data values on `xs`.
    use_mp : bool, optional
        If True, evaluate the Levy kernel with mpmath scalar arithmetic.
    """

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
        """Evaluate the Levy distribution with the configured backend."""
        if self.use_mp:
            return levy_mp(x,mu,c)
        return levy_np(x,mu,c)
    
    def K_cauchy_x(self, x):
        """Evaluate the dimensionless Gaussian-to-Cauchy transition kernel."""
        return 2 * x * self.levy(x**2, 0, 1)
        
    def rho_cauchy(self):
        """
        Propagate bounded data through the dimensionless Levy kernel.

        Returns
        -------
        tuple
            `(central, upper, lower)` propagated by trapezoidal integration.
        """
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
