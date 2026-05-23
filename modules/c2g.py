#################################################################################
#
# c2g.py: Cauchy-to-Gaussian RK transition utilities
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

"""Cauchy-to-Gaussian RK transition class."""

import numpy

from modules.kernels import cauchy_to_gaussian_kernel_mp, cauchy_to_gaussian_kernel_np

class c2g:
    """
    Apply the analytic Cauchy-to-Gaussian transition kernel to bounded data.

    Parameters
    ----------
    ws : array_like
        Grid of Cauchy centers.
    rho_epsilons : array_like
        Central Cauchy-smeared data values on `ws`.
    uppers, lowers : array_like
        Pointwise upper and lower admissible data values.
    epsilon : float
        Width of the input Cauchy smearing.
    use_mp : bool, optional
        If True, evaluate the transition kernel with mpmath scalar arithmetic.

    Methods
    -------
    rho_gauss(E, sigma)
        Return central, upper, and lower propagated Gaussian-smeared values.
    """

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
        """
        Evaluate the Cauchy-to-Gaussian transition kernel.

        Parameters
        ----------
        w : float
            Input Cauchy center.
        E : float
            Target Gaussian center.
        sigma : float
            Target Gaussian width.
        """
        if self.use_mp:
            return cauchy_to_gaussian_kernel_mp(w, E, self.epsilon, sigma)
        return cauchy_to_gaussian_kernel_np(w, E, self.epsilon, sigma)
        
    def rho_gauss(self, E, sigma):
        """
        Propagate bounded Cauchy-smeared data to a Gaussian target.

        Parameters
        ----------
        E : float
            Target Gaussian center.
        sigma : float
            Target Gaussian width.

        Returns
        -------
        tuple
            `(central, upper, lower)` propagated by trapezoidal integration.
        """
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

    
