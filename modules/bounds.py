#################################################################################
#
# bounds.py: containers and helpers for pointwise bounded numerical data
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

"""Containers and helpers for pointwise bounded numerical data."""

from dataclasses import dataclass
from typing import Optional

import numpy


@dataclass
class BoundedData:
    """
    Pointwise bounded samples on a one-dimensional grid.

    Parameters
    ----------
    grid : numpy.ndarray
        Coordinates at which the data are sampled.
    exact : numpy.ndarray, optional
        Central or reference values at `grid`. If omitted, the interval
        midpoint `bar` is used as the reference value.
    upper : numpy.ndarray
        Pointwise upper admissible values.
    lower : numpy.ndarray
        Pointwise lower admissible values.

    Useful Properties
    -----------------
    bar : numpy.ndarray
        Midpoint representation `(upper + lower) / 2`, used by SIP solvers.
    delta : numpy.ndarray
        Half-width representation `(upper - lower) / 2`, used by SIP solvers.
    width : numpy.ndarray
        Full pointwise interval width `upper - lower`.
    """
    grid: numpy.ndarray
    exact: Optional[numpy.ndarray]
    upper: numpy.ndarray
    lower: numpy.ndarray

    def __post_init__(self):
        self.grid = numpy.asarray(self.grid)
        self.upper = numpy.asarray(self.upper)
        self.lower = numpy.asarray(self.lower)
        if self.exact is None:
            self.exact = self.bar
        else:
            self.exact = numpy.asarray(self.exact)

    @classmethod
    def from_bounds(cls, grid, upper, lower):
        """
        Build bounded data when only admissible upper/lower values are known.

        Parameters
        ----------
        grid : array_like
            Coordinates at which the data are sampled.
        upper, lower : array_like
            Pointwise upper and lower admissible values.

        Returns
        -------
        BoundedData
            Data object whose `exact` reference is set to the midpoint `bar`.
        """
        return cls(grid=grid, exact=None, upper=upper, lower=lower)

    @property
    def bar(self):
        """Midpoint values of the bounded interval."""
        return (self.upper + self.lower) / 2.0

    @property
    def delta(self):
        """Half-width values of the bounded interval."""
        return (self.upper - self.lower) / 2.0

    @property
    def width(self):
        """Full pointwise interval widths."""
        return self.upper - self.lower


def make_bounded_data(grid, exact_func, upper_func, lower_func):
    """
    Evaluate an error model on a grid and return `BoundedData`.

    Parameters
    ----------
    grid : array_like
        Coordinates at which to evaluate the data model.
    exact_func : callable
        Function `exact_func(x)` returning the reference value at `x`.
    upper_func, lower_func : callable
        Functions receiving `(x, exact_value)` and returning the corresponding
        upper/lower admissible value.

    Returns
    -------
    BoundedData
        The evaluated exact, upper, and lower arrays on `grid`.
    """
    exact = numpy.array([exact_func(x) for x in grid])
    upper = numpy.array([upper_func(x, r) for x, r in zip(grid, exact)])
    lower = numpy.array([lower_func(x, r) for x, r in zip(grid, exact)])
    return BoundedData(grid=grid, exact=exact, upper=upper, lower=lower)
