#################################################################################
#
# ilt.py: bounds on smeared spectral functions from noisy correlator data
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

"""Inverse-Laplace SOCP bounds for covariance-constrained correlator data.

The routines in this module solve dual semi-infinite cone programs of the form

    sum_t lambda_t exp(-t omega) >= K(omega)

or the corresponding lower inequality, with a covariance-ellipsoid data term.
They are intended for exploratory inverse-Laplace tests and certificate
diagnostics; exact SOS certificates are deliberately not implemented here.
"""

from dataclasses import dataclass
from typing import Callable, Optional

import cvxpy as cp
import numpy
import scipy.optimize

SUPPORTED_SOCP_SOLVERS = ("CLARABEL", "ECOS", "SCS")

# ----------------------------------------------------------------------------
# Correlator model helpers
# ----------------------------------------------------------------------------

def laplace_basis(omega, t_values):
    """Return the matrix with entries `exp(-omega_i * t_j)`."""
    omega = numpy.asarray(omega, dtype=float)
    t_values = numpy.asarray(t_values, dtype=float)
    return numpy.exp(-omega[..., None] * t_values[None, :])

def correlator_from_peaks(t_values, peaks, weights):
    """Evaluate `sum_i weights_i exp(-t peaks_i)` on `t_values`."""
    t_values = numpy.asarray(t_values, dtype=float)
    peaks = numpy.asarray(peaks, dtype=float)
    weights = numpy.asarray(weights, dtype=float)
    return numpy.sum(weights[None, :] * numpy.exp(-t_values[:, None] * peaks[None, :]), axis=1)

def rho_kappa_from_peaks(kernel, peaks, weights):
    """Evaluate a smeared spectral observable for a positive sum of delta peaks."""
    return float(numpy.sum(numpy.asarray(weights, dtype=float) * kernel(peaks)))

# ----------------------------------------------------------------------------
# ILT data classes
# ----------------------------------------------------------------------------

@dataclass
class ILTProblem:
    """
    Scalar covariance-constrained inverse-Laplace finite-window problem.

    This object stores the data needed to bound a smeared spectral observable

        rho_K = integral dE rho(E) K(E)

    from noisy Euclidean correlator data

        C(t) = integral dE rho(E) exp(-t E).

    The implemented dual problems search for coefficients ``lambda_t`` such
    that, on the configured finite energy interval,

        sum_t lambda_t exp(-t E) >= K(E)      upper bound,

    or

        sum_t lambda_t exp(-t E) <= K(E)      lower bound.

    The objective includes the support function of the covariance ellipsoid
    around ``c_obs`` with radius ``sigma0``. Kernel inequalities and numerical
    certificates are checked only on the finite interval

        [energy_threshold + threshold_eps, check_max],

    where ``check_max`` is ``finite_cutoff`` when provided and ``omega_max``
    otherwise. No tail condition is imposed above ``check_max``.

    Parameters
    ----------
    t_values : numpy.ndarray
        One-dimensional array of Euclidean time slices used in the Laplace
        basis. Values must be finite, nonnegative, unique, and sorted.
    c_obs : numpy.ndarray
        One-dimensional observed correlator data with shape ``(len(t_values),)``.
    covariance : numpy.ndarray
        Symmetric positive-definite correlator covariance matrix with shape
        ``(len(t_values), len(t_values))``. If ``covariance_reg > 0``, the
        Cholesky factorization and ellipsoid norm use
        ``covariance + covariance_reg * I``.
    sigma0 : float
        Nonnegative radius of the covariance ellipsoid.
    kernel_func : callable
        Vectorized target smearing kernel. It must accept a NumPy array of
        energies and return values broadcastable to the same one-dimensional
        shape.
    kernel_kind : str, optional
        Descriptive label for the kernel, for example ``"gaussian"`` or
        ``"cauchy"``. This is metadata only.
    energy_threshold : float, optional
        Physical lower endpoint of the spectral domain.
    threshold_eps : float, optional
        Positive offset added to ``energy_threshold`` to avoid evaluating
        exactly at the threshold.
    omega_max : float, optional
        Default finite upper endpoint for exchange and certificate checks when
        ``finite_cutoff`` is not provided.
    finite_cutoff : float, optional
        Physical finite-window cutoff. When provided, it replaces ``omega_max``
        as the upper endpoint used by ``check_max``.
    covariance_reg : float, optional
        Nonnegative Tikhonov regulator added to the covariance before Cholesky
        factorization. The default ``0.0`` leaves the covariance unchanged.

    Attributes
    ----------
    cholesky : numpy.ndarray
        Lower Cholesky factor of the covariance matrix actually used for the
        ellipsoid norm.
    cholesky_inv : numpy.ndarray
        Inverse of ``cholesky``, used by the discretized primal sanity check.
    covariance_factor_matrix : numpy.ndarray
        The covariance matrix actually factorized, equal to ``covariance`` or
        ``covariance + covariance_reg * I``.
    """
    t_values: numpy.ndarray
    c_obs: numpy.ndarray
    covariance: numpy.ndarray
    sigma0: float
    kernel_func: Callable
    kernel_kind: str = 'default'
    energy_threshold: float = 0.0
    threshold_eps: float = 1e-6
    omega_max: float = 40.0
    finite_cutoff: Optional[float] = None
    covariance_reg: float = 0.0

    def __post_init__(self):
        self.t_values = numpy.asarray(self.t_values, dtype=float).reshape(-1)
        self.c_obs = numpy.asarray(self.c_obs, dtype=float).reshape(-1)
        self.covariance = numpy.asarray(self.covariance, dtype=float)
        self.sigma0 = float(self.sigma0)
        # self.covariance_sqrt = scipy.linalg.sqrtm(self.covariance).real
        self.energy_threshold = float(self.energy_threshold)
        self.threshold_eps = float(self.threshold_eps)
        self.omega_max = float(self.omega_max)
        self.covariance_reg = float(self.covariance_reg)

        if not callable(self.kernel_func):
            raise TypeError("[ILTProblem] kernel_func must be callable")
        
        # self.cholesky = numpy.linalg.cholesky(self.covariance)
        # self.cholesky_inv = numpy.linalg.inv(self.cholesky)

        n_times = len(self.t_values)
        if n_times == 0:
            raise ValueError("[ILTProblem] t_values must contain at least one time slice")
        if self.c_obs.shape != (n_times,):
            raise ValueError(
                f"[ILTProblem] c_obs must have shape ({n_times},), got {self.c_obs.shape}"
            )
        if self.covariance.shape != (n_times, n_times):
            raise ValueError(
                f"[ILTProblem] covariance must have shape ({n_times}, {n_times}), got {self.covariance.shape}"
            )

        if not numpy.all(numpy.isfinite(self.t_values)):
            raise ValueError("[ILTProblem] t_values contains non-finite entries")
        if not numpy.all(numpy.isfinite(self.c_obs)):
            raise ValueError("[ILTProblem] c_obs contains non-finite entries")
        if not numpy.all(numpy.isfinite(self.covariance)):
            raise ValueError("[ILTProblem] covariance contains non-finite entries")
        if not numpy.isfinite(self.sigma0) or self.sigma0 < 0.0:
            raise ValueError("[ILTProblem] sigma0 must be finite and non-negative")

        if numpy.any(self.t_values < 0.0):
            raise ValueError("[ILTProblem] t_values must be non-negative")
        if len(numpy.unique(self.t_values)) != n_times:
            raise ValueError("[ILTProblem] t_values must not contain duplicates")
        if numpy.any(numpy.diff(self.t_values) < 0.0):
            raise ValueError("[ILTProblem] t_values must be sorted in nondecreasing order")

        if not numpy.allclose(self.covariance, self.covariance.T, rtol=1e-11, atol=1e-13):
            raise ValueError("[ILTProblem] covariance must be symmetric")

        if not numpy.isfinite(self.energy_threshold):
            raise ValueError("[ILTProblem] energy_threshold must be finite")
        if not numpy.isfinite(self.threshold_eps) or self.threshold_eps <= 0.0:
            raise ValueError("[ILTProblem] threshold_eps must be finite and positive")
        if not numpy.isfinite(self.omega_max):
            raise ValueError("[ILTProblem] omega_max must be finite")
        if self.omega_max <= self.energy_threshold:
            raise ValueError("[ILTProblem] omega_max must be larger than energy_threshold")
        if self.finite_cutoff is not None:
            self.finite_cutoff = float(self.finite_cutoff)
            if not numpy.isfinite(self.finite_cutoff):
                raise ValueError("[ILTProblem] finite_cutoff must be finite when provided")
            if self.finite_cutoff <= self.energy_threshold:
                raise ValueError("[ILTProblem] finite_cutoff must be larger than energy_threshold")

        if self.omega_min >= self.check_max:
            raise ValueError(
                f"[ILTProblem] invalid omega domain: omega_min={self.omega_min} must be smaller than check_max={self.check_max}"
            )

        if self.covariance_reg < 0.0 or not numpy.isfinite(self.covariance_reg):
            raise ValueError("[ILTProblem] covariance_reg must be finite and nonnegative")

        self._factor_covariance()

    def _factor_covariance(self):
        reg = self.covariance_reg
        covariance_for_factor = self.covariance
        if reg > 0.0:
            covariance_for_factor = covariance_for_factor + reg * numpy.eye(len(self.t_values))
        self.covariance_factor_matrix = covariance_for_factor
        try:
            self.cholesky = numpy.linalg.cholesky(covariance_for_factor)
        except numpy.linalg.LinAlgError as exc:
            min_eval = float(numpy.min(numpy.linalg.eigvalsh(self.covariance)))
            min_eval_reg = float(numpy.min(numpy.linalg.eigvalsh(covariance_for_factor)))
            raise ValueError(
                "[ILTProblem] covariance used for factorization must be positive definite; "
                f"minimum eigenvalue of covariance is {min_eval:.6e}; "
                f"minimum eigenvalue after covariance_reg={reg:.6e} is {min_eval_reg:.6e}. "
                "Try increasing covariance_reg."
            ) from exc
        self.cholesky_inv = numpy.linalg.inv(self.cholesky)
        
    @property
    def omega_min(self):
        """Numerical lower endpoint, strictly above `energy_threshold`."""
        return self.energy_threshold + self.threshold_eps

    @property
    def check_max(self):
        """Upper endpoint for numerical checks."""
        return self.finite_cutoff if self.finite_cutoff is not None else self.omega_max

    def basis(self, omega):
        """Evaluate the Laplace basis on `omega`."""
        return laplace_basis(omega, self.t_values)

    def kernel(self, omega):
        """Evaluate the target kernel on `omega`."""
        return self.kernel_func(omega)


@dataclass
class ILTDualResult:
    """Result of one upper or lower inverse-Laplace dual exchange solve, without final certificates."""
    bound_type: str
    lam: numpy.ndarray
    value: float
    status: str
    solver: str
    active: numpy.ndarray
    history: list
    dense: numpy.ndarray
    violation: numpy.ndarray
    converged: bool
    max_violation: float
    worst_omega: float
    iterations: int
    message: str

    def bounded_kernel(self, problem: ILTProblem, omega):
        """Evaluate `sum_t lambda_t exp(-t omega)`."""
        return problem.basis(omega) @ self.lam

    def margin(self, problem: ILTProblem, omega):
        """Return the nonnegative margin required by this bound direction."""
        cert = self.bounded_kernel(problem, omega)
        kernel_values = problem.kernel(omega)
        if self.bound_type == "upper":
            return cert - kernel_values
        if self.bound_type == "lower":
            return kernel_values - cert
        raise ValueError("bound_type must be 'upper' or 'lower'")


@dataclass
class ILTCertificate:
    """
    Numerical SIP-style certificate for the continuous kernel inequality.

    Attributes
    ----------
    passed : bool
        Whether dense/local checks passed.
    min_margin : float
        Refined minimum of the required nonnegative margin.
    worst_omega : float
        Location of the refined minimum.
    dense_omega, dense_margin : numpy.ndarray
        Dense-grid diagnostic data.
    """
    passed: bool
    min_margin: float
    worst_omega: float
    dense_omega: numpy.ndarray
    dense_margin: numpy.ndarray


@dataclass
class ILTCorrection:
    """
    Exponential-basis correction for a kernel-inequality certificate.

    Attributes
    ----------
    passed : bool
        Whether the corrected bound is finite.
    eta : float
        Coefficient of the correcting basis function `exp(-t0 omega)`.
    t0 : float
        Euclidean time used for the correction basis.
    corrected_lam : numpy.ndarray
        Dual coefficients after adding/subtracting the correction.
    corrected_value : float
        Bound value recomputed with `corrected_lam`.
    raw_value : float
        Original uncorrected bound value from dual exchange solve.
    worst_omega : float
        Location of the largest weighted residual on the checked interval.
    weighted_residual_max : float
        Maximum of the weighted residual on the checked interval.
    dense_omega, dense_weighted_residual : numpy.ndarray
        Dense-grid diagnostic data.
    """
    passed: bool
    eta: float
    t0: float
    corrected_lam: numpy.ndarray
    corrected_value: float
    raw_value: float
    worst_omega: float
    weighted_residual_max: float
    dense_omega: numpy.ndarray
    dense_weighted_residual: numpy.ndarray

@dataclass
class ILTPrimalResult:
    """
    Result of a discretized positive-measure primal sanity check.

    Solves the primal problem on a finite omega grid and is useful for comparing 
    against the dual exchange bounds on the same finite interval.
    """
    bound_type: str
    status: str
    value: float
    grid: numpy.ndarray
    weights: numpy.ndarray
    solver: str
    feasible: bool
    message: str

    def __iter__(self):
        """Preserve legacy tuple unpacking: status, value, grid, weights."""
        yield self.status
        yield self.value
        yield self.grid
        yield self.weights

@dataclass
class ILTT0OptimizationResult:
    """
    Result of optimizing the exponential certificate correction over t0 candidates.

    The selected correction is the best certified corrected bound among passing
    candidates. For upper bounds this means the smallest corrected value; for
    lower bounds this means the largest corrected value.
    """
    bound_type: str
    t0: float
    correction: ILTCorrection
    certificate: ILTCertificate
    candidates: list # stores tuples (t0, correction, certificate)
    passed_candidates: list
    message: str

@dataclass
class ILTSolver:
    """
    Lightweight workflow wrapper for scalar finite-window ILT/SIP bounds.

    ``ILTSolver`` is an orchestration layer around the module-level numerical
    routines. It stores common exchange-solver settings, calls the low-level 
    solve/certificate helpers, and keeps the resulting objects in dictionaries 
    keyed by ``"upper"`` and ``"lower"``.

    Typical workflow
    ----------------
    problem = ILTProblem(...)
    workflow = ILTSolver(problem, solver="CLARABEL")

    bounds = workflow.solve_bounds()
    certs = workflow.optimize_t0_bounds(include_t0_zero=False)
    primal = workflow.run_primal_checks()

    workflow.plot_kernel_diagnostics()

    Stored results
    --------------
    results
        Maps ``"upper"`` and/or ``"lower"`` to ``ILTDualResult`` objects.
    corrections
        Maps bound directions to the selected ``ILTCorrection`` objects.
    certificates
        Maps bound directions to the corresponding ``ILTCertificate`` objects.
    t0_optimizations
        Maps bound directions to ``ILTT0OptimizationResult`` objects when
        ``optimize_t0`` or ``optimize_t0_bounds`` has been used.
    primal_results
        Maps bound directions to ``ILTPrimalResult`` objects from the
        discretized primal sanity check.

    Parameters
    ----------
    problem : ILTProblem
        Problem data and kernel/domain definition.
    solver : str, optional
        CVXPY solver name passed to the SOCP solves. If ``None``, the low-level
        routines try the supported installed solvers before falling back to the
        CVXPY default.
    n_initial : int, optional
        Number of points in the initial active energy grid when no explicit
        initial grid is supplied.
    n_check : int, optional
        Number of dense check points used in each exchange iteration.
    max_iters : int, optional
        Maximum number of exchange iterations for each bound direction.
    violation_tol : float, optional
        Exchange loop tolerance. A dual result has ``converged=True`` only when
        its final maximum kernel-inequality violation is at most this value.

    Notes
    -----
    The primal checks are finite-grid diagnostics, not continuous primal
    certificates. The corrected dual certificates are numerical SIP-style
    checks on the configured finite interval, different from SOS certificates.
    """
    problem: ILTProblem
    solver: Optional[str] = None
    n_initial: int = 2000
    n_check: int = 4000
    max_iters: int = 40
    violation_tol: float = 1e-7

    def __post_init__(self):
        self.results = {}
        self.corrections = {}
        self.certificates = {}
        self.t0_optimizations = {}
        self.primal_results = {}
    
    def solve_bound(self, bound_type, initial_grid=None):
        """Solve one upper or lower dual exchange bound and store the result."""
        if bound_type not in ["upper", "lower"]:
            raise ValueError("[ILTSolver.solve_bound] bound_type must be 'upper' or 'lower'")

        result = solve_dual_exchange(
            self.problem,
            bound_type,
            initial_grid=initial_grid,
            n_initial=self.n_initial,
            n_check=self.n_check,
            max_iters=self.max_iters,
            violation_tol=self.violation_tol,
            solver=self.solver,
        )
        self.results[bound_type] = result
        return result

    def solve_bounds(self, initial_grid=None):
        """Solve and store both upper and lower dual exchange bounds."""
        upper = self.solve_bound("upper", initial_grid=initial_grid)
        lower = self.solve_bound("lower", initial_grid=initial_grid)
        return {"upper": upper, "lower": lower}

    def get_result(self, bound_type):
        """Return a stored result, raising a clear error if it has not been solved."""
        if bound_type not in self.results:
            raise ValueError(f"[ILTSolver] no stored {bound_type} result; call solve_bound first")
        return self.results[bound_type]
    
    def certify_bound(
        self,
        bound_type,
        t0=None,
        correction_n_check=12000,
        margin_n_check=8000,
        tol=1e-7,
    ):
        """Apply fixed-t0 correction/certification to a stored bound."""
        result = self.get_result(bound_type)
        correction, certificate = certify_corrected_bound(
            self.problem,
            result,
            t0=t0,
            correction_n_check=correction_n_check,
            margin_n_check=margin_n_check,
            tol=tol,
        )
        self.corrections[bound_type] = correction
        self.certificates[bound_type] = certificate
        return correction, certificate
    
    def optimize_t0(
        self,
        bound_type,
        t0_candidates=None,
        include_t0_zero=True,
        correction_n_check=12000,
        margin_n_check=8000,
        tol=1e-7,
        require_certificate=True,
    ):
        """Optimize correction over t0 candidates for a stored bound."""
        result = self.get_result(bound_type)
        opt = optimize_t0_correction(
            self.problem,
            result,
            t0_candidates=t0_candidates,
            include_t0_zero=include_t0_zero,
            correction_n_check=correction_n_check,
            margin_n_check=margin_n_check,
            tol=tol,
            require_certificate=require_certificate,
        )
        self.t0_optimizations[bound_type] = opt
        self.corrections[bound_type] = opt.correction
        self.certificates[bound_type] = opt.certificate
        return opt

    def optimize_t0_bounds(
        self,
        t0_candidates=None,
        include_t0_zero=True,
        correction_n_check=12000,
        margin_n_check=8000,
        tol=1e-7,
        require_certificate=True,
    ):
        """Optimize t0 corrections for both stored upper and lower bounds."""
        upper = self.optimize_t0(
            "upper",
            t0_candidates=t0_candidates,
            include_t0_zero=include_t0_zero,
            correction_n_check=correction_n_check,
            margin_n_check=margin_n_check,
            tol=tol,
            require_certificate=require_certificate,
        )
        lower = self.optimize_t0(
            "lower",
            t0_candidates=t0_candidates,
            include_t0_zero=include_t0_zero,
            correction_n_check=correction_n_check,
            margin_n_check=margin_n_check,
            tol=tol,
            require_certificate=require_certificate,
        )
        return {"upper": upper, "lower": lower}
    
    def run_primal_check(self, bound_type, omega_max=None, n_grid=1200, solver=None):
        """Run and store a discretized primal sanity check for one bound direction."""
        primal = solve_primal_grid(
            self.problem,
            bound_type,
            omega_max=omega_max,
            n_grid=n_grid,
            solver=self.solver if solver is None else solver,
        )
        self.primal_results[bound_type] = primal
        return primal
    
    def run_primal_checks(self, omega_max=None, n_grid=1200, solver=None):
        """Run and store primal sanity checks for both upper and lower directions."""
        upper = self.run_primal_check("upper", omega_max=omega_max, n_grid=n_grid, solver=solver)
        lower = self.run_primal_check("lower", omega_max=omega_max, n_grid=n_grid, solver=solver)
        return {"upper": upper, "lower": lower}

    def plot_kernel_bounds(
        self,
        ax=None,
        n_grid=1000,
        omega_min=None,
        omega_max=None,
        corrected=True,
        show_active=True,
    ):
        """
        Plot stored upper/lower kernel reconstructions against the target kernel.

        If ``corrected`` is True, use stored corrections when available. Missing
        results or corrections are simply omitted.
        """
        upper = self.results.get("upper")
        lower = self.results.get("lower")
        upper_correction = self.corrections.get("upper") if corrected else None
        lower_correction = self.corrections.get("lower") if corrected else None
        return plot_kernel_bounds(
            self.problem,
            upper=upper,
            lower=lower,
            upper_correction=upper_correction,
            lower_correction=lower_correction,
            ax=ax,
            n_grid=n_grid,
            omega_min=omega_min,
            omega_max=omega_max,
            show_active=show_active,
        )

    def plot_kernel_margins(
        self,
        ax=None,
        n_grid=1000,
        omega_min=None,
        omega_max=None,
        corrected=True,
    ):
        """
        Plot stored upper/lower kernel margins.

        If ``corrected`` is True, use stored corrections when available. Negative
        margins indicate wrong-side kernel inequality violations.
        """
        upper = self.results.get("upper")
        lower = self.results.get("lower")
        upper_correction = self.corrections.get("upper") if corrected else None
        lower_correction = self.corrections.get("lower") if corrected else None
        return plot_kernel_margins(
            self.problem,
            upper=upper,
            lower=lower,
            upper_correction=upper_correction,
            lower_correction=lower_correction,
            ax=ax,
            n_grid=n_grid,
            omega_min=omega_min,
            omega_max=omega_max,
        )
    
    def plot_kernel_diagnostics(
        self,
        n_grid=1000,
        omega_min=None,
        omega_max=None,
        corrected=True,
        title=None,
        show_active=True,
    ):
        """
        Plot stored kernel bounds and margins side by side.

        If ``corrected`` is True, use stored corrections when available.
        """
        upper = self.results.get("upper")
        lower = self.results.get("lower")
        upper_correction = self.corrections.get("upper") if corrected else None
        lower_correction = self.corrections.get("lower") if corrected else None
        return plot_kernel_diagnostics(
            self.problem,
            upper=upper,
            lower=lower,
            upper_correction=upper_correction,
            lower_correction=lower_correction,
            n_grid=n_grid,
            omega_min=omega_min,
            omega_max=omega_max,
            corrected_title=title,
            show_active=show_active,
        )
    
# ----------------------------------------------------------------------------
# ILT solvers and associated helpers
# ----------------------------------------------------------------------------

def _objective(problem, lam, bound_type):
    assert bound_type in ["upper", "lower"], "[_objective] bound_type must be 'upper' or 'lower'"
    # stat = problem.sigma0 * cp.norm(problem.covariance_sqrt @ lam, 2)
    stat = problem.sigma0 * cp.norm(problem.cholesky.T @ lam, 2)
    central = problem.c_obs @ lam
    if bound_type == "upper":
        return cp.Minimize(central + stat)
    if bound_type == "lower":
        return cp.Maximize(central - stat)

def evaluate_dual_bound(problem, lam, bound_type):
    """Evaluate the covariance-ellipsoid dual objective at fixed coefficients."""
    assert bound_type in ["upper", "lower"], "[evaluate_dual_bound] bound_type must be 'upper' or 'lower'"
    lam = numpy.asarray(lam, dtype=float)
    central = float(problem.c_obs @ lam)
    # stat = float(problem.sigma0 * numpy.linalg.norm(problem.covariance_sqrt @ lam))
    stat = float(problem.sigma0 * numpy.linalg.norm(problem.cholesky.T @ lam))
    if bound_type == "upper":
        return central + stat
    if bound_type == "lower":
        return central - stat


def solve_socp_on_grid(problem, active_omega, bound_type, solver=None):
    """
    Solve the finite-grid SOCP for one bound direction.

    Parameters
    ----------
    problem : ILTProblem
        Inverse-Laplace problem data.
    active_omega : array_like
        Grid points where the kernel inequality is imposed.
    bound_type : {'upper', 'lower'}
        Direction of the requested bound.
    solver : str, optional
        Explicit CVXPY solver name.

    Returns
    -------
    tuple
        `(lambda, value, status, solver_used)`.
    """
    active_omega = numpy.asarray(active_omega, dtype=float)
    lam = cp.Variable(len(problem.t_values))
    basis = problem.basis(active_omega)
    kernel_values = problem.kernel(active_omega)
    if bound_type == "upper":
        constraints = [basis @ lam >= kernel_values]
    elif bound_type == "lower":
        constraints = [basis @ lam <= kernel_values]
    else:
        raise ValueError("bound_type must be 'upper' or 'lower'")

    prob = cp.Problem(_objective(problem, lam, bound_type), constraints)
    installed = set(cp.installed_solvers())
    if solver is None:
        solvers = [candidate for candidate in SUPPORTED_SOCP_SOLVERS if candidate in installed]
        solvers.append(None)
    else:
        solvers = [solver]

    attempts = []
    for candidate in solvers:
        if candidate is not None and candidate not in installed:
            attempts.append((candidate, "not installed"))
            continue
        try:
            if candidate is None:
                prob.solve(verbose=False)
                solver_used = "CVXPY default"
            else:
                prob.solve(solver=candidate, verbose=False)
                solver_used = candidate
            attempts.append((solver_used, prob.status))
            if lam.value is not None and prob.status in ("optimal", "optimal_inaccurate"):
                return numpy.asarray(lam.value).ravel(), float(prob.value), prob.status, solver_used
        except cp.error.SolverError as exc:
            attempts.append((candidate or "CVXPY default", f"SolverError: {exc}"))

    statuses = [status for _, status in attempts]
    if statuses and all(status == "unbounded" for status in statuses):
        raise RuntimeError(
            "[solve_socp_on_grid] The finite-grid SOCP is unbounded. "
            "This usually indicates an inconsistent covariance ellipsoid "
            "or an insufficient certificate grid. "
            f"Installed solvers: {sorted(installed)}. Attempts: {attempts}."
        )

    raise RuntimeError(
        "[solve_socp_on_grid] No SOCP solver succeeded. "
        f"Installed solvers: {sorted(installed)}. Attempts: {attempts}."
    )

def violation_values(problem, lam, omega, bound_type):
    """Return positive values where the requested kernel inequality is violated."""
    assert bound_type in ["upper", "lower"], "[violation_values] bound_type must be 'upper' or 'lower'"
    cert = problem.basis(omega) @ lam
    kernel_values = problem.kernel(omega)
    if bound_type == "upper":
        return kernel_values - cert
    if bound_type == "lower":
        return cert - kernel_values

def find_worst_violation(problem, lam, bound_type, n_check=4000):
    """Find the worst dense/local violation on the configured finite domain."""
    n_check = int(n_check)
    if n_check < 2:
        raise ValueError("[find_worst_violation] n_check must be at least 2")
    dense = numpy.linspace(problem.omega_min, problem.check_max, n_check)
    vals = violation_values(problem, lam, dense, bound_type)
    idx = int(numpy.argmax(vals))
    best_omega = float(dense[idx])
    best_val = float(vals[idx])

    left = dense[max(0, idx - 2)]
    right = dense[min(len(dense) - 1, idx + 2)]
    if right > left:
        def objective(w):
            return -float(violation_values(problem, lam, numpy.array([w]), bound_type)[0])

        res = scipy.optimize.minimize_scalar(objective, bounds=(left, right), method="bounded")
        if res.success and -res.fun > best_val:
            best_omega = float(res.x)
            best_val = float(-res.fun)

    return best_omega, best_val, dense, vals

def make_initial_energy_grid(
    problem,
    n_initial=2000,
    low_fraction=0.55,
    log_fraction=0.30,
):
    """
    Build a generic initial exchange grid for Laplace SIP problems.

    The grid combines:
    - a uniform grid on the full finite interval;
    - an extra dense low-energy grid;
    - a logarithmic grid in distance from the threshold.

    This avoids relying on known spectral peak locations while giving the first
    finite-grid SOCP enough low-energy resolution to avoid artificial
    primal-grid infeasibility.
    """
    n_initial = int(n_initial)
    if n_initial < 2:
        raise ValueError("[make_initial_energy_grid] n_initial must be at least 2")

    lo = float(problem.omega_min)
    hi = float(problem.check_max)
    if hi <= lo:
        raise ValueError("[make_initial_energy_grid] invalid problem energy interval")

    n_low = max(2, int(low_fraction * n_initial))
    n_log = max(2, int(log_fraction * n_initial))
    n_uniform = max(2, n_initial - n_low - n_log)

    span = hi - lo
    low_hi = min(hi, lo + 0.25 * span)

    uniform = numpy.linspace(lo, hi, n_uniform)
    low = numpy.linspace(lo, low_hi, n_low)

    log_min = max(problem.threshold_eps, 1e-12)
    log_max = span
    log = lo + numpy.geomspace(log_min, log_max, n_log)
    log = log[(log >= lo) & (log <= hi)]

    return numpy.unique(numpy.r_[lo, uniform, low, log, hi])

def solve_dual_exchange(
    problem,
    bound_type,
    initial_grid=None,
    n_initial=2000,
    n_check=4000,
    max_iters=40,
    violation_tol=1e-7,
    solver=None,
):
    """
    Solve one covariance-dual bound using a finite-grid exchange loop.

    Parameters
    ----------
    problem : ILTProblem
        Inverse-Laplace problem data.
    bound_type : {'upper', 'lower'}
        Direction of the requested bound.
    initial_grid : array_like, optional
        Initial active energy grid. If omitted, a uniform grid is constructed.
    n_initial : int, optional
        Number of initial grid points.
    n_check : int, optional
        Number of dense check points per exchange iteration.
    max_iters : int, optional
        Maximum exchange iterations.
    violation_tol : float, optional
        Stop when the worst violation is below this threshold.
    solver : str, optional
        Explicit CVXPY solver.

    Returns
    -------
    ILTDualResult
        Optimized dual coefficients and exchange diagnostics.
    """
    if bound_type not in ["upper", "lower"]:
        raise ValueError("[solve_dual_exchange] bound_type must be 'upper' or 'lower'")

    n_initial = int(n_initial)
    n_check = int(n_check)
    max_iters = int(max_iters)
    violation_tol = float(violation_tol)

    if n_initial < 2:
        raise ValueError("[solve_dual_exchange] n_initial must be at least 2")
    if n_check < 2:
        raise ValueError("[solve_dual_exchange] n_check must be at least 2")
    if max_iters < 1:
        raise ValueError("[solve_dual_exchange] max_iters must be at least 1")
    if not numpy.isfinite(violation_tol) or violation_tol < 0.0:
        raise ValueError("[solve_dual_exchange] violation_tol must be finite and nonnegative")
    
    if initial_grid is None:
        initial_grid = make_initial_energy_grid(problem, n_initial=n_initial)
    initial_grid = numpy.asarray(initial_grid, dtype=float).reshape(-1)
    if len(initial_grid) == 0:
        raise ValueError("[solve_dual_exchange] initial_grid must contain at least one point")
    if not numpy.all(numpy.isfinite(initial_grid)):
        raise ValueError("[solve_dual_exchange] initial_grid contains non-finite entries")
    if numpy.any(initial_grid < problem.omega_min) or numpy.any(initial_grid > problem.check_max):
        raise ValueError("[solve_dual_exchange] initial_grid must lie inside [omega_min, check_max]")
    initial_grid = numpy.unique(initial_grid)

    active = list(initial_grid)
    history = []
    dense = None
    vals = None
    lam = None
    value = float("nan")
    status = "not_started"
    solver_used = solver or "not_started"
    worst_omega = float("nan")
    worst_val = float("inf")
    converged = False
    message = "not started"

    for iteration in range(max_iters):
        lam, value, status, solver_used = solve_socp_on_grid(problem, active, bound_type, solver=solver)
        worst_omega, worst_val, dense, vals = find_worst_violation(
            problem, lam, bound_type, n_check=n_check
        )
        history.append((iteration, value, worst_omega, worst_val, len(active), status, solver_used))
        if worst_val <= violation_tol:
            converged = True
            message = "converged: max_violation <= violation_tol"
            break
        active.append(worst_omega)

    if not converged:
        message = (
            "max_iters reached before satisfying violation_tol; "
            f"final max_violation={worst_val:.6e}, violation_tol={violation_tol:.6e}"
        )
        
    return ILTDualResult(
        bound_type=bound_type,
        lam=lam,
        value=value,
        status=status,
        solver=solver_used,
        active=numpy.asarray(active),
        history=history,
        dense=dense,
        violation=vals,
        converged=converged,
        max_violation=float(worst_val),
        worst_omega=float(worst_omega),
        iterations=len(history),
        message=message,
    )

def solve_dual_bounds(
    problem,
    initial_grid=None,
    n_initial=2000,
    n_check=4000,
    max_iters=40,
    violation_tol=1e-7,
    solver=None,
):
    """Solve upper and lower covariance-dual bounds with the same exchange settings."""
    upper = solve_dual_exchange(
        problem,
        "upper",
        initial_grid=initial_grid,
        n_initial=n_initial,
        n_check=n_check,
        max_iters=max_iters,
        violation_tol=violation_tol,
        solver=solver,
    )
    lower = solve_dual_exchange(
        problem,
        "lower",
        initial_grid=initial_grid,
        n_initial=n_initial,
        n_check=n_check,
        max_iters=max_iters,
        violation_tol=violation_tol,
        solver=solver,
    )
    return {"upper": upper, "lower": lower}

def certify_dual_margin(problem, result, n_check=8000, tol=1e-7):
    """
    Numerically certify the continuous kernel inequality on the check domain.

    This is a SIP-style certificate diagnostic: it refines the smallest margin
    on the finite check interval.
    """
    n_check = int(n_check)
    if n_check < 2:
        raise ValueError("[certify_dual_margin] n_check must be at least 2")
    dense = numpy.linspace(problem.omega_min, problem.check_max, n_check)
    margins = result.margin(problem, dense)
    idx = int(numpy.argmin(margins))
    worst_omega = float(dense[idx])
    min_margin = float(margins[idx])

    left = dense[max(0, idx - 2)]
    right = dense[min(len(dense) - 1, idx + 2)]
    if right > left:
        def objective(w):
            return float(result.margin(problem, numpy.array([w]))[0])

        res = scipy.optimize.minimize_scalar(objective, bounds=(left, right), method="bounded")
        if res.success and res.fun < min_margin:
            min_margin = float(res.fun)
            worst_omega = float(res.x)

    return ILTCertificate(
        passed=min_margin >= -tol,
        min_margin=min_margin,
        worst_omega=worst_omega,
        dense_omega=dense,
        dense_margin=margins,
    )


def compute_exponential_certificate_correction(
    problem,
    result,
    t0=None,
    n_check=12000,
):
    """
    Correct a dual certificate using one available Laplace basis function.

    The correction uses `w(omega) = exp(-t0 omega)`. For an upper bound it adds
    `eta w` to the certificate; for a lower bound it subtracts `eta w`. The
    corrected bound is recomputed from the covariance-ellipsoid support function
    using the corrected dual coefficients.

    Parameters
    ----------
    problem : ILTProblem
        Inverse-Laplace problem data.
    result : ILTDualResult
        Raw upper or lower dual result.
    t0 : float, optional
        Time slice to use as the correction basis. Defaults to `min(t_values)`.
    n_check : int, optional
        Dense points used to maximize the weighted residual.

    Returns
    -------
    ILTCorrection
        Correction coefficient, corrected coefficients, and corrected bound.
    """
    n_check = int(n_check)
    if n_check < 2:
        raise ValueError("[compute_exponential_certificate_correction] n_check must be at least 2")
    t_values = numpy.asarray(problem.t_values, dtype=float)
    if t0 is None:
        correction_index = int(numpy.argmin(t_values))
    else:
        correction_index = int(numpy.argmin(numpy.abs(t_values - t0)))
        if abs(t_values[correction_index] - t0) > 1e-12:
            raise ValueError("t0 must match one of problem.t_values")
    t0 = float(t_values[correction_index])

    dense = numpy.linspace(problem.omega_min, problem.check_max, n_check)
    weight = numpy.exp(-t0 * dense)
    residual = violation_values(problem, result.lam, dense, result.bound_type)
    weighted = numpy.divide(
        residual,
        weight,
        out=numpy.full_like(residual, numpy.inf),
        where=weight > 0.0,
    )
    idx = int(numpy.argmax(weighted))
    worst_omega = float(dense[idx])
    weighted_max = float(weighted[idx])

    left = dense[max(0, idx - 2)]
    right = dense[min(len(dense) - 1, idx + 2)]
    if right > left:
        def objective(w):
            ww = numpy.exp(-t0 * w)
            return -float(violation_values(problem, result.lam, numpy.array([w]), result.bound_type)[0] / ww)

        res = scipy.optimize.minimize_scalar(objective, bounds=(left, right), method="bounded")
        if res.success and -res.fun > weighted_max:
            weighted_max = float(-res.fun)
            worst_omega = float(res.x)

    eta = max(0.0, weighted_max)
    corrected_lam = numpy.array(result.lam, dtype=float, copy=True)
    if result.bound_type == "upper":
        corrected_lam[correction_index] += eta
    elif result.bound_type == "lower":
        corrected_lam[correction_index] -= eta
    else:
        raise ValueError("bound_type must be 'upper' or 'lower'")

    corrected_value = evaluate_dual_bound(problem, corrected_lam, result.bound_type)
    return ILTCorrection(
        passed=numpy.isfinite(corrected_value),
        eta=eta,
        t0=t0,
        corrected_lam=corrected_lam,
        corrected_value=corrected_value,
        raw_value=result.value,
        worst_omega=worst_omega,
        weighted_residual_max=weighted_max,
        dense_omega=dense,
        dense_weighted_residual=weighted,
    )

def certify_corrected_bound(
    problem,
    result,
    t0=None,
    correction_n_check=12000,
    margin_n_check=8000,
    tol=1e-7,
):
    """Apply an exponential correction and certify the corrected dual margin."""
    correction_n_check = int(correction_n_check)
    margin_n_check = int(margin_n_check)
    if correction_n_check < 2:
        raise ValueError("[certify_corrected_bound] correction_n_check must be at least 2")
    if margin_n_check < 2:
        raise ValueError("[certify_corrected_bound] margin_n_check must be at least 2")
    correction = compute_exponential_certificate_correction(
        problem,
        result,
        t0=t0,
        n_check=correction_n_check,
    )
    corrected_result = ILTDualResult(
        bound_type=result.bound_type,
        lam=correction.corrected_lam,
        value=correction.corrected_value,
        status=result.status,
        solver=result.solver,
        active=result.active,
        history=result.history,
        dense=result.dense,
        violation=result.violation,
        converged=result.converged,
        max_violation=result.max_violation,
        worst_omega=result.worst_omega,
        iterations=result.iterations,
        message=result.message,
    )
    if (
        not correction.passed
        or not numpy.isfinite(correction.eta)
        or not numpy.isfinite(correction.corrected_value)
        or not numpy.all(numpy.isfinite(correction.corrected_lam))
    ):
        dense = numpy.linspace(problem.omega_min, problem.check_max, int(margin_n_check))
        failed_certificate = ILTCertificate(
            passed=False,
            min_margin=float("nan"),
            worst_omega=float("nan"),
            dense_omega=dense,
            dense_margin=numpy.full_like(dense, numpy.nan),
        )
        return correction, failed_certificate
    certificate = certify_dual_margin(
        problem,
        corrected_result,
        n_check=margin_n_check,
        tol=tol,
    )
    return correction, certificate

def default_t0_candidates(problem, include_t0_zero=True):
    """
    Return default t0 candidates for exponential certificate correction.

    By default, t=0 is included if available in problem.t_values, but note that a 
    constant correction might be too large.
    """
    t_values = numpy.asarray(problem.t_values, dtype=float).reshape(-1)
    if include_t0_zero:
        return t_values.copy()

    positive = t_values[t_values > 0.0]
    if len(positive) > 0:
        return positive
    return t_values.copy()

def optimize_t0_correction(
    problem,
    result,
    t0_candidates=None,
    include_t0_zero=True,
    correction_n_check=12000,
    margin_n_check=8000,
    tol=1e-7,
    require_certificate=True,
):
    """
    Optimize the exponential certificate correction over candidate t0 values.

    Parameters
    ----------
    problem : ILTProblem
        Inverse-Laplace problem data.
    result : ILTDualResult
        Raw upper or lower dual result.
    t0_candidates : array_like, optional
        Candidate times. If omitted, use problem.t_values by default, optionally excluding t=0.
    include_t0_zero : bool, optional
        Whether the default candidate list may include t=0. Default is True.
    correction_n_check, margin_n_check, tol
        Passed to certify_corrected_bound.
    require_certificate : bool, optional
        If True (default), select only candidates whose final corrected margin certificate passes.
        If False, select from all finite corrected values.
    """
    if result.bound_type not in ["upper", "lower"]:
        raise ValueError("[optimize_t0_correction] result.bound_type must be 'upper' or 'lower'")

    if t0_candidates is None:
        t0_candidates = default_t0_candidates(problem, include_t0_zero=include_t0_zero)
    else:
        t0_candidates = numpy.asarray(t0_candidates, dtype=float).reshape(-1)

    if len(t0_candidates) == 0:
        raise ValueError("[optimize_t0_correction] t0_candidates must contain at least one value")

    candidates = []
    passed_candidates = []
    errors = []

    for t0 in t0_candidates:
        try:
            correction, certificate = certify_corrected_bound(
                problem,
                result,
                t0=float(t0),
                correction_n_check=correction_n_check,
                margin_n_check=margin_n_check,
                tol=tol,
            )
        except Exception as exc:
            errors.append((float(t0), str(exc)))
            continue

        entry = (float(t0), correction, certificate)
        candidates.append(entry)

        usable = numpy.isfinite(correction.corrected_value)
        if require_certificate:
            usable = usable and certificate.passed
        if usable:
            passed_candidates.append(entry)

    if len(passed_candidates) == 0:
        raise RuntimeError(
            "[optimize_t0_correction] no usable t0 candidate found; "
            f"tried {list(numpy.asarray(t0_candidates, dtype=float))}; errors={errors}"
        )

    if result.bound_type == "upper":
        best = min(passed_candidates, key=lambda item: item[1].corrected_value)
    else:
        best = max(passed_candidates, key=lambda item: item[1].corrected_value)

    best_t0, best_correction, best_certificate = best
    return ILTT0OptimizationResult(
        bound_type=result.bound_type,
        t0=best_t0,
        correction=best_correction,
        certificate=best_certificate,
        candidates=candidates,
        passed_candidates=passed_candidates,
        message=(
            f"selected t0={best_t0} from {len(passed_candidates)} usable "
            f"candidate(s) out of {len(t0_candidates)}"
        ),
    )

def solve_primal_grid(problem, bound_type, omega_max=None, n_grid=1200, solver=None):
    """
    Solve a discretized positive-measure primal sanity check.

    It is useful for comparing against the dual exchange result on a finite energy
    grid. By default, the primal grid uses the same upper endpoint as the dual
    check interval.
    """
    if bound_type not in ["upper", "lower"]:
        raise ValueError("[solve_primal_grid] bound_type must be 'upper' or 'lower'")

    n_grid = int(n_grid)
    if n_grid < 2:
        raise ValueError("[solve_primal_grid] n_grid must be at least 2")

    omega_max = problem.check_max if omega_max is None else float(omega_max)
    if not numpy.isfinite(omega_max):
        raise ValueError("[solve_primal_grid] omega_max must be finite")
    if omega_max <= problem.omega_min:
        raise ValueError(
            f"[solve_primal_grid] omega_max={omega_max} must be larger than omega_min={problem.omega_min}"
        )

    grid = numpy.linspace(problem.omega_min, omega_max, n_grid)
    weights = cp.Variable(n_grid, nonneg=True)
    basis = problem.basis(grid).T
    residual = problem.cholesky_inv @ (basis @ weights - problem.c_obs)
    observable = problem.kernel(grid) @ weights
    constraints = [cp.norm(residual, 2) <= problem.sigma0]

    if bound_type == "upper":
        objective = cp.Maximize(observable)
    else:
        objective = cp.Minimize(observable)

    prob = cp.Problem(objective, constraints)

    installed = set(cp.installed_solvers())
    if solver is None:
        solvers = [candidate for candidate in SUPPORTED_SOCP_SOLVERS if candidate in installed]
        solvers.append(None)
    else:
        solvers = [solver]

    attempts = []
    solver_used = "not_run"
    last_status = "not_run"
    for candidate in solvers:
        if candidate is not None and candidate not in installed:
            attempts.append((candidate, "not installed"))
            continue
        try:
            if candidate is None:
                prob.solve(verbose=False)
                solver_used = "CVXPY default"
            else:
                prob.solve(solver=candidate, verbose=False)
                solver_used = candidate
            attempts.append((solver_used, prob.status))
            last_status = prob.status
            if prob.status in ("optimal", "optimal_inaccurate") and weights.value is not None:
                return ILTPrimalResult(
                    bound_type=bound_type,
                    status=prob.status,
                    value=float(prob.value),
                    grid=grid,
                    weights=numpy.asarray(weights.value).ravel(),
                    solver=solver_used,
                    feasible=True,
                    message="solved",
                )
        except cp.error.SolverError as exc:
            attempts.append((candidate or "CVXPY default", f"SolverError: {exc}"))
            last_status = "solver_error"

    value = float(prob.value) if prob.value is not None else float("nan")
    primal_weights = (
        numpy.asarray(weights.value).ravel()
        if weights.value is not None
        else numpy.full(n_grid, numpy.nan)
    )
    return ILTPrimalResult(
        bound_type=bound_type,
        status=last_status,
        value=value,
        grid=grid,
        weights=primal_weights,
        solver=solver_used,
        feasible=False,
        message=(
            "[solve_primal_grid] primal grid solve did not produce an optimal solution; "
            f"installed solvers: {sorted(installed)}; attempts: {attempts}"
        ),
    )

# ----------------------------------------------------------------------------
# ILT plotting helpers
# ----------------------------------------------------------------------------

def _kernel_bound_values(problem, result, omega, correction=None):
    """Evaluate a raw or corrected dual kernel reconstruction."""
    if correction is None:
        lam = result.lam
    else:
        lam = correction.corrected_lam
    return problem.basis(omega) @ lam


def _kernel_margin_values(problem, result, omega, correction=None):
    """Evaluate the required nonnegative kernel margin."""
    bound = _kernel_bound_values(problem, result, omega, correction=correction)
    kernel = problem.kernel(omega)
    if result.bound_type == "upper":
        return bound - kernel
    if result.bound_type == "lower":
        return kernel - bound
    raise ValueError("[_kernel_margin_values] result.bound_type must be 'upper' or 'lower'")


def plot_kernel_bounds(
    problem,
    upper=None,
    lower=None,
    upper_correction=None,
    lower_correction=None,
    ax=None,
    n_grid=1000,
    omega_min=None,
    omega_max=None,
    show_active=True,
):
    """
    Plot the target kernel and raw/corrected upper/lower dual kernel bounds.

    Parameters
    ----------
    problem : ILTProblem
        Problem defining the kernel and finite energy interval.
    upper, lower : ILTDualResult, optional
        Raw upper/lower exchange results.
    upper_correction, lower_correction : ILTCorrection, optional
        Corrected coefficients to plot instead of the raw coefficients.
    ax : matplotlib axis, optional
        Axis to draw on. If omitted, use the current axis.
    n_grid : int, optional
        Number of energy points used for the plotted curves.
    omega_min, omega_max : float, optional
        Plot interval. Defaults to ``problem.omega_min`` and ``problem.check_max``.
    show_active : bool, optional
        Whether to mark active exchange points for supplied raw results.
    """
    import matplotlib.pyplot as plt

    ax = plt.gca() if ax is None else ax
    omega_min = problem.omega_min if omega_min is None else float(omega_min)
    omega_max = problem.check_max if omega_max is None else float(omega_max)
    if omega_max <= omega_min:
        raise ValueError("[plot_kernel_bounds] omega_max must be larger than omega_min")

    n_grid = int(n_grid)
    if n_grid < 2:
        raise ValueError("[plot_kernel_bounds] n_grid must be at least 2")
    omega = numpy.linspace(omega_min, omega_max, int(n_grid))
    kernel = problem.kernel(omega)
    ax.plot(omega, kernel, color="black", linewidth=2.0, label="target kernel")

    if upper is not None:
        upper_values = _kernel_bound_values(problem, upper, omega, correction=upper_correction)
        label = "upper corrected" if upper_correction is not None else "upper raw"
        ax.plot(omega, upper_values, color="tab:red", label=label)
        if show_active:
            active = numpy.asarray(upper.active, dtype=float)
            active = active[(active >= omega_min) & (active <= omega_max)]
            if len(active) > 0:
                ax.plot(
                    active,
                    _kernel_bound_values(problem, upper, active, correction=upper_correction),
                    "|",
                    color="tab:red",
                    alpha=0.45,
                    markersize=8,
                    label="upper active",
                )

    if lower is not None:
        lower_values = _kernel_bound_values(problem, lower, omega, correction=lower_correction)
        label = "lower corrected" if lower_correction is not None else "lower raw"
        ax.plot(omega, lower_values, color="tab:blue", label=label)
        if show_active:
            active = numpy.asarray(lower.active, dtype=float)
            active = active[(active >= omega_min) & (active <= omega_max)]
            if len(active) > 0:
                ax.plot(
                    active,
                    _kernel_bound_values(problem, lower, active, correction=lower_correction),
                    "|",
                    color="tab:blue",
                    alpha=0.45,
                    markersize=8,
                    label="lower active",
                )

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\kappa(\omega)$")
    ax.set_xlim(omega_min, omega_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_kernel_margins(
    problem,
    upper=None,
    lower=None,
    upper_correction=None,
    lower_correction=None,
    ax=None,
    n_grid=1000,
    omega_min=None,
    omega_max=None,
):
    """
    Plot signed nonnegative margins for upper/lower kernel certificates.

    The plotted margin is ``dual_kernel - target_kernel`` for an upper bound and
    ``target_kernel - dual_kernel`` for a lower bound. Negative values therefore
    indicate wrong-side kernel inequality violations.
    """
    import matplotlib.pyplot as plt

    ax = plt.gca() if ax is None else ax
    omega_min = problem.omega_min if omega_min is None else float(omega_min)
    omega_max = problem.check_max if omega_max is None else float(omega_max)
    if omega_max <= omega_min:
        raise ValueError("[plot_kernel_margins] omega_max must be larger than omega_min")

    n_grid = int(n_grid)
    if n_grid < 2:
        raise ValueError("[plot_kernel_margins] n_grid must be at least 2")
    omega = numpy.linspace(omega_min, omega_max, int(n_grid))
    ax.axhline(0.0, color="black", linewidth=0.8)

    if upper is not None:
        margin = _kernel_margin_values(problem, upper, omega, correction=upper_correction)
        label = "upper corrected margin" if upper_correction is not None else "upper raw margin"
        ax.plot(omega, margin, color="tab:red", label=label)

    if lower is not None:
        margin = _kernel_margin_values(problem, lower, omega, correction=lower_correction)
        label = "lower corrected margin" if lower_correction is not None else "lower raw margin"
        ax.plot(omega, margin, color="tab:blue", label=label)

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("kernel margin")
    ax.set_xlim(omega_min, omega_max)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax

def plot_kernel_diagnostics(
    problem,
    upper=None,
    lower=None,
    upper_correction=None,
    lower_correction=None,
    n_grid=1000,
    omega_min=None,
    omega_max=None,
    corrected_title=None,
    show_active=True,
):
    """
    Plot kernel bounds and signed margins side by side.

    This is a convenience diagnostic for checking whether the upper/lower dual
    reconstructions sit on the correct side of the target kernel over the
    configured finite interval.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    plot_kernel_bounds(
        problem,
        upper=upper,
        lower=lower,
        upper_correction=upper_correction,
        lower_correction=lower_correction,
        ax=axes[0],
        n_grid=n_grid,
        omega_min=omega_min,
        omega_max=omega_max,
        show_active=show_active,
    )
    plot_kernel_margins(
        problem,
        upper=upper,
        lower=lower,
        upper_correction=upper_correction,
        lower_correction=lower_correction,
        ax=axes[1],
        n_grid=n_grid,
        omega_min=omega_min,
        omega_max=omega_max,
    )

    axes[0].set_title("kernel bounds")
    axes[1].set_title("kernel margins")
    if corrected_title is not None:
        fig.suptitle(corrected_title)

    return fig