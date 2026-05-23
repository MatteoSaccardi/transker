#################################################################################
#
# sip.py: semi-infinite-programming exchange solver and certificate utilities
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

"""Semi-infinite-programming exchange solver and certificate utilities."""

import numpy
import cvxpy as cp
import scipy.optimize

SUPPORTED_SOLVERS = (cp.CLARABEL, cp.ECOS, cp.SCS)


def available_sip_solvers():
    """
    Return installed CVXPY solvers supported by this module.

    Returns
    -------
    list
        Supported solver names in preferred order. `CLARABEL` is preferred,
        followed by `ECOS` and `SCS` when installed.
    """
    installed = set(cp.installed_solvers())
    return [solver for solver in SUPPORTED_SOLVERS if solver in installed]


def solve_sip_exchange(param_centers, rho_bar, rho_delta, omega_dense, 
                       target_func, basis_func, bound_type='upper', 
                       omega_active_init=None, omega_bounds=None, 
                       use_local_search=True, tol=1e-7, max_iters=100,
                       scale=False, force_positivity=False,):
    """
    Solve one upper or lower SIP dual problem by exchange/cutting planes.
    
    Parameters
    ----------
    param_centers : array_like, 1D
        The parameters defining the basis functions (e.g., positions alpha or widths sigma).
        Must have the same length as rho_bar and rho_delta.
    rho_bar : array_like, 1D
        Central values of the data constraints.
    rho_delta : array_like, 1D
        Absolute error widths of the data constraints.
    omega_dense : array_like, 1D
        Dense grid for fast continuous violation checking and certificate evaluation.
    target_func : callable
        Function representing the target kernel: target_func(omega) -> float
    basis_func : callable
        Function representing the basis kernel: basis_func(omega, param) -> float
    bound_type : {'upper', 'lower'}
        Type of rigorous bound to construct.
    omega_active_init : array_like, 1D, optional
        Initial set of omega points to enforce constraints. Defaults to a coarse 
        subsample of omega_dense.
    omega_bounds : tuple, optional
        (min, max) bounds for the L-BFGS-B local search. Required if use_local_search=True.
    use_local_search : bool, optional
        If True, uses scipy L-BFGS-B to find the exact continuous maximum violation. 
        If False, relies purely on the discrete omega_dense grid.
    tol : float
        Tolerance for the maximum constraint violation.
    max_iters : int
        Maximum number of cutting-plane exchange iterations.
    scale : bool
        If True, scales the problem to avoid numerical issues.
    force_positivity : bool
        If True, adds a positivity constraint to the coefficients of the problem.

    Returns
    -------
    lam_opt : numpy.ndarray
        Optimized coefficients on `param_centers`.
    kappa_lambda_dense : numpy.ndarray
        Reconstructed kernel sampled on `omega_dense`.
    prob_val : float
        Solver objective value before certificate correction.
    diff_dense : numpy.ndarray
        `target_func(omega_dense) - kappa_lambda_dense`.
    Phi_dense : numpy.ndarray
        Basis matrix sampled on `(omega_dense, param_centers)`.

    Raises
    ------
    RuntimeError
        If no supported CVXPY solver is installed.
    """
    N_params = len(param_centers)
    
    # Initialize the active constraint grid
    if omega_active_init is not None:
        omega_active = list(omega_active_init)
    else:
        omega_active = list(omega_dense[::10]) # Default coarse grid
        
    # Precompute dense arrays for the fast violation check
    target_dense = numpy.array([target_func(w) for w in omega_dense])
    Phi_dense = numpy.array([[basis_func(w, p) for p in param_centers] for w in omega_dense])
    
    lam_opt = None
    prob_val = 0.0
    
    # --- COLUMN SCALING ---
    # Normalize basis functions so their maximum value is exactly 1.0
    col_scales = numpy.max(Phi_dense, axis=0)
    col_scales[col_scales < 1e-14] = 1.0 
    
    # Scale objective coefficients inversely
    rho_bar_c = rho_bar / col_scales
    rho_delta_c = rho_delta / col_scales

    if scale:
        # Normalize the objective to prevent solver failures when rho penalties explode
        # Normalize entire objective to O(1)
        obj_scale = numpy.max(numpy.abs(rho_bar_c) + numpy.abs(rho_delta_c))
        if obj_scale < 1e-12:
            obj_scale = 1.0
    else:
        obj_scale = 1.0

    solvers_to_try = available_sip_solvers()
    if not solvers_to_try:
        supported = ", ".join(SUPPORTED_SOLVERS)
        raise RuntimeError(
            f"No supported CVXPY solver is installed. Install at least one of: {supported}."
        )

    for iteration in range(max_iters):
        # 1. Build constraints on the ACTIVE grid
        Phi_active = numpy.array([[basis_func(w, p) for p in param_centers] for w in omega_active])
        target_active = numpy.array([target_func(w) for w in omega_active])
        
        # Apply column scaling to the active matrix
        Phi_active_c = Phi_active / col_scales[None, :]

        # --- PRECONDITIONING: ROW SCALING ---
        row_scales = numpy.max(Phi_active_c, axis=1)
        row_scales[row_scales < 1e-14] = 1.0  # Prevent division by zero
        Phi_active_scaled = Phi_active_c / row_scales[:, None]
        target_active_scaled = target_active / row_scales
        
        # 2. Formulate L1 Dual Problem
        if force_positivity:
            lam = cp.Variable(N_params, nonneg=True)
            abs_lam = lam
        else:
            lam = cp.Variable(N_params)
            abs_lam = cp.abs(lam)

        kappa_lambda_scaled = Phi_active_scaled @ lam
        
        if bound_type == 'upper':
            obj = cp.Minimize( (lam @ rho_bar_c + cp.sum(cp.multiply(abs_lam, rho_delta_c)) ) / obj_scale)
            cons = [kappa_lambda_scaled >= target_active_scaled]
            sign = 1.0
        else:
            obj = cp.Maximize( (lam @ rho_bar_c - cp.sum(cp.multiply(abs_lam, rho_delta_c))) / obj_scale)
            cons = [kappa_lambda_scaled <= target_active_scaled]
            sign = -1.0
            
        prob = cp.Problem(obj, cons)
        for solver in solvers_to_try:
            try:
                prob.solve(solver=solver, verbose=False)
                if lam.value is not None:
                    break  # Success!
            except cp.error.SolverError:
                continue

        if lam.value is None:
            print(f"[sip.solve_sip_exchange] Warning: Solvers failed at iter {iteration}. Halting early.")
            if lam_opt is None: 
                lam_opt = numpy.zeros(N_params)
            break
            
        lam_opt = lam.value / col_scales
        prob_val = prob.value * obj_scale
        
        # 3. Find continuous violation
        kappa_lambda_dense = Phi_dense @ lam_opt
        diff_dense = target_dense - kappa_lambda_dense
        violation_dense = sign * diff_dense
        
        guess_idx = numpy.argmax(violation_dense)
        guess_omega = omega_dense[guess_idx]
        max_violation = violation_dense[guess_idx]
        
        # 4. Optional L-BFGS continuous refinement
        worst_omega = guess_omega
        if use_local_search and omega_bounds is not None and max_violation > 0:
            def neg_violation(w_arr):
                w = w_arr[0]
                t_w = target_func(w)
                phi_w = numpy.array([basis_func(w, p) for p in param_centers])
                k_lam_w = numpy.dot(phi_w, lam_opt)
                return -(sign * (t_w - k_lam_w))
            
            res = scipy.optimize.minimize(neg_violation, x0=[guess_omega], 
                                          bounds=[omega_bounds], method='L-BFGS-B')
            if -res.fun > max_violation:
                max_violation = -res.fun
                worst_omega = res.x[0]
        
        # 5. Exchange check
        if max_violation < tol:
            break
            
        omega_active.append(worst_omega)
        
    # Final dense evaluation
    kappa_lambda_dense = Phi_dense @ lam_opt
    diff_dense = target_dense - kappa_lambda_dense
    
    return lam_opt, kappa_lambda_dense, prob_val, diff_dense, Phi_dense

def compute_sip_certificates(diff_dense, Phi_dense, bound_type='upper'):
    """
    Compute pointwise certificate corrections for SIP residuals.

    Parameters
    ----------
    diff_dense : numpy.ndarray
        Difference between target and reconstructed kernel on the dense grid.
    Phi_dense : numpy.ndarray
        Basis matrix sampled on the same dense grid.
    bound_type : {'upper', 'lower'}, optional
        Direction of the certificate.

    Returns
    -------
    numpy.ndarray
        Certificate value for each basis/data parameter.
    """
    N_params = Phi_dense.shape[1]
    deltas = numpy.zeros(N_params)
    
    for i in range(N_params):
        phi_param = Phi_dense[:, i]
        mask = phi_param > 0 #1e-14 # Threshold for division by basis elements
        
        if not numpy.any(mask):
            continue
            
        if bound_type == 'upper':
            weighted_diff = diff_dense[mask] / phi_param[mask]
        else:
            weighted_diff = -diff_dense[mask] / phi_param[mask]
            
        deltas[i] = max(0.0, numpy.max(weighted_diff))
        
    return deltas
