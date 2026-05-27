#################################################################################
#
# transition.py: transition-kernel problem abstractions and solver wrappers
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

"""Problem abstractions and solver wrappers for kernel transitions."""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy
import scipy.linalg
import scipy.optimize

from modules.bounds import BoundedData
from modules.sip import compute_sip_certificates, solve_sip_exchange


@dataclass
class TransitionKernelProblem:
    """
    Mathematical data defining a transition-kernel approximation problem.

    Parameters
    ----------
    param_grid : numpy.ndarray
        Parameter values for the available basis kernels and bounded data.
        Examples are input centers, widths, or regulator grid points.
    omega_grid : numpy.ndarray
        Dense grid used to evaluate target/basis kernels and certificates.
    data : BoundedData
        Bounded input data associated with `param_grid`.
    target_func : callable
        Function `target_func(w)` evaluating the target kernel.
    basis_func : callable
        Function `basis_func(w, p)` evaluating one basis kernel at grid point
        `w` and parameter `p`.

    Methods
    -------
    target_vector()
        Evaluate the target kernel on `omega_grid`.
    basis_matrix()
        Evaluate the matrix with entries `basis_func(omega_grid[i], param_grid[j])`.
    """
    param_grid: numpy.ndarray
    omega_grid: numpy.ndarray
    data: BoundedData
    target_func: Callable
    basis_func: Callable

    def target_vector(self):
        """
        Return the target kernel sampled on `omega_grid`.

        Returns
        -------
        numpy.ndarray
            Array of shape `(len(omega_grid),)`.
        """
        return numpy.array([self.target_func(w) for w in self.omega_grid])

    def basis_matrix(self):
        """
        Return the basis matrix on `(omega_grid, param_grid)`.

        Returns
        -------
        numpy.ndarray
            Matrix of shape `(len(omega_grid), len(param_grid))`.

        Notes
        -----
        Vectorized kernels are attempted first. If the kernel cannot broadcast
        over array inputs, scalar evaluation is used instead.
        """
        try:
            matrix = self.basis_func(self.omega_grid[:, None], self.param_grid[None, :])
            return numpy.asarray(matrix)
        except Exception:
            return numpy.array(
                [[self.basis_func(w, p) for p in self.param_grid] for w in self.omega_grid]
            )


@dataclass
class SIPBoundResult:
    """
    Result of a single SIP bound solve.

    Attributes
    ----------
    bound_type : {'upper', 'lower'}
        Direction of the solved bound.
    lam : numpy.ndarray
        Optimized SIP coefficients.
    kappa : numpy.ndarray
        Reconstructed kernel sampled on the dense grid.
    candidate : float
        Solver objective before certificate correction.
    diff : numpy.ndarray
        Difference between target and reconstructed kernel on the dense grid.
    phi : numpy.ndarray
        Basis matrix used for certificate evaluation.
    deltas : numpy.ndarray
        Certificate corrections for each basis/data parameter.
    rigorous : float
        Final certificate-corrected rigorous bound.
    """
    bound_type: str
    lam: numpy.ndarray
    kappa: numpy.ndarray
    candidate: float
    diff: numpy.ndarray
    phi: numpy.ndarray
    deltas: numpy.ndarray
    rigorous: float


@dataclass
class SIPIntervalResult:
    """
    Upper and lower SIP bounds for the same transition problem.

    Attributes
    ----------
    upper, lower : SIPBoundResult
        Certificate-corrected upper and lower bound results.

    Useful Properties
    -----------------
    width : float
        Difference `upper.rigorous - lower.rigorous`.
    """
    upper: SIPBoundResult
    lower: SIPBoundResult

    @property
    def width(self):
        return self.upper.rigorous - self.lower.rigorous


class SIPTransition:
    """
    High-level wrapper around the SIP exchange solver.

    Parameters
    ----------
    problem : TransitionKernelProblem
        Transition problem to solve.
    omega_active_init : array_like, optional
        Initial active grid passed to `solve_sip_exchange`.
    omega_bounds : tuple, optional
        Bounds used by the optional local violation search.
    use_local_search : bool, optional
        Whether to refine dense-grid violations with L-BFGS-B.
    tol : float, optional
        Exchange-loop violation tolerance.
    max_iters : int, optional
        Maximum number of exchange iterations.
    scale : bool, optional
        Whether to enable objective scaling in the SIP solver.
    force_positivity : bool, optional
        Whether to constrain SIP coefficients to be nonnegative.

    Methods
    -------
    solve_bound(bound_type)
        Solve one upper or lower bound.
    solve_interval()
        Solve both upper and lower bounds.
    width()
        Solve both bounds and return the resulting interval width.
    """
    def __init__(
        self,
        problem: TransitionKernelProblem,
        omega_active_init=None,
        omega_bounds: Optional[Tuple[float, float]] = None,
        use_local_search: bool = True,
        tol: float = 1e-7,
        max_iters: int = 100,
        scale: bool = False,
        force_positivity: bool = False,
    ):
        self.problem = problem
        self.omega_active_init = omega_active_init
        self.omega_bounds = omega_bounds
        self.use_local_search = use_local_search
        self.tol = tol
        self.max_iters = max_iters
        self.scale = scale
        self.force_positivity = force_positivity

    def solve_bound(self, bound_type):
        """
        Solve one SIP bound and apply certificate corrections.

        Parameters
        ----------
        bound_type : {'upper', 'lower'}
            Direction of the requested bound.

        Returns
        -------
        SIPBoundResult
            Solver coefficients, dense reconstruction, certificate data, and
            final rigorous value.
        """
        lam, kappa, candidate, diff, phi = solve_sip_exchange(
            param_centers=self.problem.param_grid,
            rho_bar=self.problem.data.bar,
            rho_delta=self.problem.data.delta,
            omega_dense=self.problem.omega_grid,
            target_func=self.problem.target_func,
            basis_func=self.problem.basis_func,
            bound_type=bound_type,
            omega_active_init=self.omega_active_init,
            omega_bounds=self.omega_bounds,
            use_local_search=self.use_local_search,
            tol=self.tol,
            max_iters=self.max_iters,
            scale=self.scale,
            force_positivity=self.force_positivity,
        )
        deltas = compute_sip_certificates(diff, phi, bound_type)

        if bound_type == "upper":
            rigorous = numpy.min(candidate + deltas * self.problem.data.upper)
        elif bound_type == "lower":
            rigorous = numpy.max(candidate - deltas * self.problem.data.upper)
        else:
            raise ValueError("bound_type must be 'upper' or 'lower'")

        return SIPBoundResult(
            bound_type=bound_type,
            lam=lam,
            kappa=kappa,
            candidate=candidate,
            diff=diff,
            phi=phi,
            deltas=deltas,
            rigorous=rigorous,
        )

    def solve_interval(self):
        """
        Solve the upper and lower SIP bounds.

        Returns
        -------
        SIPIntervalResult
            Paired upper/lower results.
        """
        upper = self.solve_bound("upper")
        lower = self.solve_bound("lower")
        return SIPIntervalResult(upper=upper, lower=lower)

    def width(self):
        """
        Solve the interval and return its rigorous width.

        Returns
        -------
        float
            `solve_interval().upper.rigorous - solve_interval().lower.rigorous`.
        """
        return self.solve_interval().width


@dataclass
class RegulatedRKResult:
    """
    Result of a regulated RK/ridge reconstruction.

    Attributes
    ----------
    alpha : float
        Ridge regulator used in the normal equations.
    coefficients : numpy.ndarray
        Reconstruction coefficients on the problem parameter grid.
    reconstruction : numpy.ndarray
        Dense-grid reconstructed kernel.
    discrepancy : numpy.ndarray
        Absolute dense-grid target/reconstruction discrepancy.
    certificates : numpy.ndarray
        Certificate factors for each parameter-grid envelope.
    certificate_index : int
        Index of the envelope minimizing the systematic certificate penalty.
    certificate_penalty : float
        Chosen systematic penalty.
    statistical_error, systematic_error, total_error : float
        Error decomposition used to optimize `alpha`.
    upper, lower : float
        Final propagated bounds.
    """
    alpha: float
    coefficients: numpy.ndarray
    reconstruction: numpy.ndarray
    discrepancy: numpy.ndarray
    certificates: numpy.ndarray
    certificate_index: int
    certificate_penalty: float
    statistical_error: float
    systematic_error: float
    total_error: float
    upper: float
    lower: float

    @property
    def certificate(self):
        """Certificate factor selected by `certificate_index`."""
        return self.certificates[self.certificate_index]


@dataclass
class RegulatedRKIntervalResult:
    """
    Upper and lower regulated RK bounds after optimizing the ridge regulator.

    Attributes
    ----------
    RK_method : {'asymmetric', 'symmetric'}
        Optimization strategy. `asymmetric` optimizes the upper and lower
        bounds separately; `symmetric` uses one regulator optimized by total
        error for both bounds.
    upper, lower : RegulatedRKResult
        Regulated results used for the final upper and lower bounds.

    Useful Properties
    -----------------
    upper_bound, lower_bound : float
        Final propagated bounds.
    width : float
        Difference `upper_bound - lower_bound`.
    """
    RK_method: str
    upper: RegulatedRKResult
    lower: RegulatedRKResult

    @property
    def upper_bound(self):
        return self.upper.upper

    @property
    def lower_bound(self):
        return self.lower.lower

    @property
    def width(self):
        return self.upper_bound - self.lower_bound


class RegulatedRKTransition:
    """
    Solver for regulated RK/ridge transition reconstructions.

    Parameters
    ----------
    problem : TransitionKernelProblem
        Transition problem defining the target vector, basis matrix, and
        bounded input data.
    assume_a : str, optional
        Matrix structure hint passed to `scipy.linalg.solve`. The default
        `"pos"` matches the positive-definite ridge systems used in the
        regulated C2C examples.

    Methods
    -------
    coefficients(alpha)
        Solve the ridge normal equations for a fixed regulator.
    evaluate(alpha)
        Compute reconstruction, certificates, errors, and bounds at `alpha`.
    optimize_log_alpha(bounds=(-8, 1), RK_method='asymmetric', scipy_method='bounded')
        Optimize upper/lower bounds over `log10(alpha)`.
    """
    def __init__(self, problem, assume_a="pos"):
        self.problem = problem
        self.assume_a = assume_a
        self._basis_matrix = None
        self._target_vector = None
        self._gtg = None
        self._gtc = None
        self._identity = None

    @property
    def basis_matrix(self):
        if self._basis_matrix is None:
            self._basis_matrix = self.problem.basis_matrix()
        return self._basis_matrix

    @property
    def target_vector(self):
        if self._target_vector is None:
            self._target_vector = self.problem.target_vector()
        return self._target_vector

    @property
    def gtg(self):
        if self._gtg is None:
            self._gtg = self.basis_matrix.T @ self.basis_matrix
        return self._gtg

    @property
    def gtc(self):
        if self._gtc is None:
            self._gtc = self.basis_matrix.T @ self.target_vector
        return self._gtc

    @property
    def identity(self):
        if self._identity is None:
            self._identity = numpy.eye(len(self.problem.param_grid))
        return self._identity

    def coefficients(self, alpha):
        """
        Solve the ridge normal equations at fixed regulator.

        Parameters
        ----------
        alpha : float
            Positive ridge regulator.

        Returns
        -------
        numpy.ndarray
            Coefficients solving `(G.T @ G + alpha I) c = G.T @ target`.
        """
        return scipy.linalg.solve(
            self.gtg + alpha * self.identity,
            self.gtc,
            assume_a=self.assume_a,
        )

    def evaluate(self, alpha):
        """
        Evaluate one regulated reconstruction.

        Parameters
        ----------
        alpha : float
            Ridge regulator.

        Returns
        -------
        RegulatedRKResult
            Coefficients, dense reconstruction, selected certificate, error
            decomposition, and propagated upper/lower bounds.
        """
        coeffs = self.coefficients(alpha)
        reconstruction = self.basis_matrix @ coeffs
        discrepancy = numpy.abs(self.target_vector - reconstruction)
        certificates = numpy.max(discrepancy[:, None] / self.basis_matrix, axis=0)
        systematic_errors = certificates * self.problem.data.upper
        certificate_index = int(numpy.argmin(systematic_errors))
        systematic_error = systematic_errors[certificate_index]
        statistical_error = numpy.sum(numpy.abs(coeffs) * self.problem.data.delta)

        prod_lower = numpy.minimum(coeffs * self.problem.data.lower, coeffs * self.problem.data.upper)
        prod_upper = numpy.maximum(coeffs * self.problem.data.lower, coeffs * self.problem.data.upper)
        upper = numpy.sum(prod_upper) + systematic_error
        lower = numpy.sum(prod_lower) - systematic_error

        return RegulatedRKResult(
            alpha=alpha,
            coefficients=coeffs,
            reconstruction=reconstruction,
            discrepancy=discrepancy,
            certificates=certificates,
            certificate_index=certificate_index,
            certificate_penalty=systematic_error,
            statistical_error=statistical_error,
            systematic_error=systematic_error,
            total_error=statistical_error + systematic_error,
            upper=upper,
            lower=lower,
        )

    def _optimize_log_objective(self, objective, bounds, grid_size, scipy_method):
        lo, hi = bounds
        grid = numpy.linspace(lo, hi, grid_size)
        values = numpy.array([objective(x) for x in grid])
        best = int(numpy.argmin(values))

        left = grid[max(0, best - 1)]
        right = grid[min(len(grid) - 1, best + 1)]

        if left == right:
            return scipy.optimize.OptimizeResult(
                x=grid[best],
                fun=values[best],
                success=True,
                message="Minimum found at the edge of the log-alpha grid.",
            )

        return scipy.optimize.minimize_scalar(
            objective,
            bounds=(left, right),
            method=scipy_method,
        )

    def optimize_log_alpha(
        self,
        bounds=(-8, 1),
        RK_method="asymmetric",
        scipy_method="bounded",
        grid_size=200,
    ):
        """
        Optimize the regulator over a log10 interval.

        Parameters
        ----------
        bounds : tuple, optional
            Search interval for `log10(alpha)`.
        RK_method : {'asymmetric', 'symmetric'}, optional
            Regulated-bound optimization strategy. `asymmetric` optimizes the
            upper and lower propagated bounds separately. `symmetric` uses one
            regulator optimized by total error for both bounds.
        scipy_method : str, optional
            Method passed to `scipy.optimize.minimize_scalar`.
        grid_size : int, optional
            Number of log-alpha grid points used to identify the local basin
            before calling the scalar minimizer.

        Returns
        -------
        tuple
            `(scipy_result, interval_result)` for `RK_method='symmetric'`, or
            `((upper_scipy_result, lower_scipy_result), interval_result)` for
            `RK_method='asymmetric'`.
        """
        if RK_method == "symmetric":
            def objective(log_alpha):
                return self.evaluate(10**log_alpha).total_error

            res = self._optimize_log_objective(
                objective=objective,
                bounds=bounds,
                grid_size=grid_size,
                scipy_method=scipy_method,
            )
            result = self.evaluate(10**res.x)
            return res, RegulatedRKIntervalResult(
                RK_method=RK_method,
                upper=result,
                lower=result,
            )

        if RK_method == "asymmetric":
            def upper_objective(log_alpha):
                return self.evaluate(10**log_alpha).upper

            def lower_objective(log_alpha):
                return -self.evaluate(10**log_alpha).lower

            upper_res = self._optimize_log_objective(
                objective=upper_objective,
                bounds=bounds,
                grid_size=grid_size,
                scipy_method=scipy_method,
            )
            lower_res = self._optimize_log_objective(
                objective=lower_objective,
                bounds=bounds,
                grid_size=grid_size,
                scipy_method=scipy_method,
            )
            return (upper_res, lower_res), RegulatedRKIntervalResult(
                RK_method=RK_method,
                upper=self.evaluate(10**upper_res.x),
                lower=self.evaluate(10**lower_res.x),
            )

        raise ValueError("RK_method must be 'asymmetric' or 'symmetric'")
