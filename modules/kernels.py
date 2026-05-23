#################################################################################
#
# kernels.py: shared NumPy and mpmath kernel functions
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

"""Kernel functions used by the transition examples.

The module exposes explicit NumPy/SciPy and mpmath variants where both
backends are useful. NumPy variants accept scalar or array inputs and are
appropriate for vectorized grids. mpmath variants are intended for scalar
high-precision evaluation in the older transition classes and plotting scripts.
"""

import mpmath
import numpy
import scipy.special


def cauchy_np(w, w1, eps):
    """Evaluate the Cauchy kernel with NumPy-compatible arithmetic."""
    return (eps / numpy.pi) / ((w - w1) ** 2 + eps**2)


def gaussian_np(w, w1, sigma):
    """Evaluate the normalized Gaussian kernel with NumPy-compatible arithmetic."""
    return 1 / numpy.sqrt(2 * numpy.pi * sigma**2) * numpy.exp(
        -((w - w1) ** 2) / (2 * sigma**2)
    )


def cauchy_mp(w, w1, eps):
    """Evaluate the Cauchy kernel with mpmath arithmetic."""
    return (eps / mpmath.pi) / ((w - w1) ** 2 + eps**2)


def gaussian_mp(w, w1, sigma):
    """Evaluate the normalized Gaussian kernel with mpmath arithmetic."""
    return 1 / mpmath.sqrt(2 * mpmath.pi * sigma**2) * mpmath.exp(
        -((w - w1) ** 2) / (2 * sigma**2)
    )


def levy_np(x, mu, c):
    """Evaluate the Levy density with NumPy-compatible arithmetic."""
    return numpy.sqrt(c / (2 * numpy.pi)) * numpy.exp(-c / (2 * (x - mu))) / (
        x - mu
    ) ** 1.5


def levy_mp(x, mu, c):
    """Evaluate the Levy density with mpmath arithmetic."""
    return (
        mpmath.sqrt(c / (2 * mpmath.pi))
        * mpmath.exp(-c / (2 * (x - mu)))
        / mpmath.power(x - mu, mpmath.mpf("1.5"))
    )


def cauchy_to_gaussian_kernel_np(w, E, eps, sigma):
    """
    Evaluate the analytic Cauchy-to-Gaussian transition kernel.

    Parameters
    ----------
    w : float or numpy.ndarray
        Input Cauchy center(s).
    E : float
        Target Gaussian center.
    eps : float
        Input Cauchy width.
    sigma : float
        Target Gaussian width.
    """
    z = (eps - 1j * (w - E)) / numpy.sqrt(2 * sigma**2)
    return (
        1
        / numpy.sqrt(2 * numpy.pi * sigma**2)
        * numpy.real(numpy.exp(z**2) * (1 + scipy.special.erf(z)))
    )


def cauchy_to_gaussian_kernel_mp(w, E, eps, sigma):
    """mpmath scalar variant of `cauchy_to_gaussian_kernel_np`."""
    z = (eps - 1j * (w - E)) / mpmath.sqrt(2 * sigma**2)
    return (
        1
        / mpmath.sqrt(2 * mpmath.pi * sigma**2)
        * (mpmath.exp(z**2) * (1 + mpmath.erf(z))).real
    )


def gaussian_to_cauchy_kernel_np(sigma, eps):
    """Evaluate the Gaussian-width to Cauchy-width Levy transition kernel."""
    return 2 * sigma * levy_np(sigma**2, 0, eps**2)


def gaussian_to_cauchy_kernel_mp(sigma, eps):
    """mpmath scalar variant of `gaussian_to_cauchy_kernel_np`."""
    return 2 * sigma * levy_mp(sigma**2, 0, eps**2)


def gaussian_to_cauchy_kernel_x_np(x):
    """Evaluate the dimensionless Gaussian-to-Cauchy Levy kernel."""
    return 2 * x * levy_np(x**2, 0, 1)


def gaussian_to_cauchy_kernel_x_mp(x):
    """mpmath scalar variant of `gaussian_to_cauchy_kernel_x_np`."""
    return 2 * x * levy_mp(x**2, 0, 1)


cauchy = cauchy_np
gaussian = gaussian_np
