import numpy
from tqdm.auto import tqdm
import cvxpy as cp
import os
import scipy
import shutil

import sys
sys.path.append("../")
from modules.sip import solve_sip_exchange, compute_sip_certificates

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=shutil.which("latex") is not None)
plt.rc('font', family='serif')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

plot_folder = '../paperplots/sip'

# ==============================================================================
# Kernel and model spectral density definitions
# ==============================================================================

def cauchy(w, w1, eps):
    return (eps / numpy.pi) / ((w - w1)**2 + eps**2)

def gaussian(w, w1, sigma):
    return 1 / numpy.sqrt(2 * numpy.pi * sigma**2) * numpy.exp(- (w - w1)**2 / (2 * sigma**2))

# Model Spectral Density Parameters
ms = [0.5, 4.0]
As = [1.0, 2.0]

def rho_eps(omega, eps):
    return sum([As[i] * cauchy(omega, ms[i], eps) for i in range(len(ms))])

def eps0(omega):
    return 0.75
def correction(omega, eps):
    p = 2
    return 1 + (eps0(omega) / eps)**p

def rho_sigma(omega, sigma):
    return sum([As[i] * gaussian(omega, ms[i], sigma) for i in range(len(ms))])

def compute_sip_bounds_at_eps(eps, omega_t, sigma_t, alpha_grid, dense_grid):
    # 1. Generate Input Data at THIS epsilon
    r_exact = numpy.array([rho_eps(a, eps) for a in alpha_grid])
    r_plus  = numpy.array([r * correction(a, eps) for r, a in zip(r_exact, alpha_grid)])
    r_minus = numpy.array([r / correction(a, eps) for r, a in zip(r_exact, alpha_grid)])
    
    r_bar = (r_plus + r_minus) / 2.0
    r_delta = (r_plus - r_minus) / 2.0
    
    target_func = lambda w: gaussian(w, omega_t, sigma_t)
    basis_func = lambda w, alpha: cauchy(w, alpha, eps)

    # 2. Upper Bound
    _, _, val_up, diff_up, Phi_d = solve_sip_exchange(
        param_centers=alpha_grid, 
        rho_bar=r_bar, 
        rho_delta=r_delta, 
        omega_dense=dense_grid, 
        target_func=target_func, 
        basis_func=basis_func, 
        bound_type='upper', 
        omega_active_init=alpha_grid,
        omega_bounds=(dense_grid[0], dense_grid[-1]),
        tol=1e-8,
        max_iters=15
    )
    deltas_up = compute_sip_certificates(diff_up, Phi_d, bound_type='upper')
    rig_up = numpy.min(val_up + deltas_up * r_plus)
    
    # 3. Lower Bound
    _, _, val_low, diff_low, _ = solve_sip_exchange(
        param_centers=alpha_grid, 
        rho_bar=r_bar, 
        rho_delta=r_delta, 
        omega_dense=dense_grid, 
        target_func=target_func, 
        basis_func=basis_func, 
        bound_type='lower', 
        omega_active_init=alpha_grid,
        omega_bounds=(dense_grid[0], dense_grid[-1]),
        tol=1e-8,
        max_iters=15
    )
    deltas_low = compute_sip_certificates(diff_low, Phi_d, bound_type='lower')
    rig_low = numpy.max(val_low - deltas_low * r_plus)
    
    return rig_up, rig_low

# c2g transition kernel
def K_gauss_static(w, E, eps, sig):
    """Analytic Cauchy-to-Gaussian transition kernel."""
    z = (eps - 1j * (w - E)) / numpy.sqrt(2 * sig**2)
    return 1 / numpy.sqrt(2 * numpy.pi * sig**2) * numpy.real(numpy.exp(z**2) * (1 + scipy.special.erf(z)))


def optimize_sip_eps(omega_t, sigma_t, alpha_grid, dense_grid, max_iters=100, tol=1e-7):
    """
    Computes the rigorous SIP width for a given epsilon, using given tolerances.
    """
    target_func = lambda w: gaussian(w, omega_t, sigma_t)
    def sip_width_cost(eps):
        r_ex = numpy.array([rho_eps(a, eps) for a in alpha_grid])
        r_pl = numpy.array([r * correction(a, eps) for r, a in zip(r_ex, alpha_grid)])
        r_mi = numpy.array([r / correction(a, eps) for r, a in zip(r_ex, alpha_grid)])
        
        r_bar = (r_pl + r_mi) / 2.0
        r_delta = (r_pl - r_mi) / 2.0
        
        basis_func = lambda w, alpha: cauchy(w, alpha, eps)
        # Upper Bound
        _, _, val_up, diff_up, Phi_d = solve_sip_exchange(
            param_centers=alpha_grid, 
            rho_bar=r_bar, 
            rho_delta=r_delta, 
            omega_dense=dense_grid, 
            target_func=target_func, 
            basis_func=basis_func, 
            bound_type='upper', 
            omega_active_init=alpha_grid,
            omega_bounds=(dense_grid[0], dense_grid[-1]),
            tol=tol, 
            max_iters=max_iters
        )
        deltas_up = compute_sip_certificates(diff_up, Phi_d, bound_type='upper')
        rig_up = numpy.min(val_up + deltas_up * r_pl)
        
        # Lower Bound
        _, _, val_low, diff_low, Phi_d = solve_sip_exchange(
            param_centers=alpha_grid, 
            rho_bar=r_bar, 
            rho_delta=r_delta, 
            omega_dense=dense_grid, 
            target_func=target_func, 
            basis_func=basis_func, 
            bound_type='lower', 
            omega_active_init=alpha_grid,
            omega_bounds=(dense_grid[0], dense_grid[-1]),
            tol=tol, 
            max_iters=max_iters
        )
        deltas_low = compute_sip_certificates(diff_low, Phi_d, bound_type='lower')
        rig_low = numpy.max(val_low - deltas_low * r_pl)
        
        return rig_up - rig_low
    
    # Search for the optimal eps in the range [0.5, 3.5] using loose tolerance
    res = scipy.optimize.minimize_scalar(
        sip_width_cost, bounds=(0.5, 3.5), method='bounded', options={'xatol': 0.1}
    )
    eps_star = res.x
    
    # Re-evaluate the bounds exactly at eps_star to extract up/low values
    r_ex = numpy.array([rho_eps(a, eps_star) for a in alpha_grid])
    r_pl = numpy.array([r * correction(a, eps_star) for r, a in zip(r_ex, alpha_grid)])
    r_mi = numpy.array([r / correction(a, eps_star) for r, a in zip(r_ex, alpha_grid)])
    r_bar = (r_pl + r_mi) / 2.0
    r_delta = (r_pl - r_mi) / 2.0
    
    basis_func = lambda w, alpha: cauchy(w, alpha, eps_star)
    _, _, v_up, d_up, Pd = solve_sip_exchange(
        param_centers=alpha_grid, 
        rho_bar=r_bar, 
        rho_delta=r_delta, 
        omega_dense=dense_grid,
        target_func=target_func, 
        basis_func=basis_func, 
        bound_type='upper', 
        omega_active_init=alpha_grid,
        omega_bounds=(dense_grid[0], dense_grid[-1]),
        tol=tol, 
        max_iters=max_iters
    )
    r_up = numpy.min(v_up + compute_sip_certificates(d_up, Pd, 'upper') * r_pl)
    
    _, _, v_lo, d_lo, _ = solve_sip_exchange(
        param_centers=alpha_grid, 
        rho_bar=r_bar, 
        rho_delta=r_delta, 
        omega_dense=dense_grid,
        target_func=target_func, 
        basis_func=basis_func, 
        bound_type='lower', 
        omega_active_init=alpha_grid,
        omega_bounds=(dense_grid[0], dense_grid[-1]),
        tol=tol, 
        max_iters=max_iters
    )
    r_lo = numpy.max(v_lo - compute_sip_certificates(d_lo, Pd, 'lower') * r_pl)
    
    return eps_star, r_up, r_lo

def optimize_rk_eps(omega_t, sigma_t, grid):
    """
    Computes the RK width for a given epsilon using the exact same data grid as SIP.
    """
    def rk_width_cost(eps):
        r_ex = numpy.array([rho_eps(a, eps) for a in grid])
        r_pl = numpy.array([r * correction(a, eps) for r, a in zip(r_ex, grid)])
        r_mi = numpy.array([r / correction(a, eps) for r, a in zip(r_ex, grid)])
        
        K_vals = K_gauss_static(grid, omega_t, eps, sigma_t)
        
        term_plus = K_vals * r_pl
        term_minus = K_vals * r_mi
        
        integrand_up = numpy.where(K_vals >= 0, term_plus, term_minus)
        integrand_low = numpy.where(K_vals >= 0, term_minus, term_plus)
        
        rk_up = numpy.trapezoid(integrand_up, grid)
        rk_low = numpy.trapezoid(integrand_low, grid)

        # prod_lower = numpy.minimum(term_minus, term_plus)
        # prod_upper = numpy.maximum(term_minus, term_plus)

        # rk_low = numpy.trapezoid(prod_lower, grid)
        # rk_up = numpy.trapezoid(prod_upper, grid)
        
        return rk_up - rk_low

    # Optimize RK epsilon
    res = scipy.optimize.minimize_scalar(
        rk_width_cost, bounds=(0.5, 3.5), method='bounded', options={'xatol': 0.01}
    )
    eps_star_rk = res.x
    
    # Re-evaluate exactly at optimal epsilon
    r_ex = numpy.array([rho_eps(a, eps_star_rk) for a in grid])
    r_pl = numpy.array([r * correction(a, eps_star_rk) for r, a in zip(r_ex, grid)])
    r_mi = numpy.array([r / correction(a, eps_star_rk) for r, a in zip(r_ex, grid)])
    
    K_vals = K_gauss_static(grid, omega_t, eps_star_rk, sigma_t)
    term_plus = K_vals * r_pl
    term_minus = K_vals * r_mi
    
    rk_up = numpy.trapezoid(numpy.where(K_vals >= 0, term_plus, term_minus), grid)
    rk_low = numpy.trapezoid(numpy.where(K_vals >= 0, term_minus, term_plus), grid)
    
    return eps_star_rk, rk_up, rk_low

def main():

    if not os.path.exists('../paperplots'):
        os.makedirs('../paperplots')

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    print('[sip] Starting...')

    # ------------------------------------------------------------------------------
    # PLOT 1: SIP vs RK Stability Analysis
    # ------------------------------------------------------------------------------

    print('[sip] Plot 1: Cauchy to Gaussian stability analysis. This will take 6 minutes...')

    eps_fixed = 2.0
    sigma_target = 1.0
    omega_target = 2.5

    target_func = lambda w: gaussian(w, omega_target, sigma_target)

    omega_dense = numpy.linspace(omega_target - 8.0, omega_target + 8.0, 1000)

    # For the input data, let's take 200 discrete measurements spread across the grid
    alpha_centers = numpy.linspace(omega_target - 8.0, omega_target + 8.0, 300)

    # Generate exact data and bounds at the alpha centers
    rhos_exact = numpy.array([rho_eps(a, eps_fixed) for a in alpha_centers])
    rhos_plus  = numpy.array([r * correction(a, eps_fixed) for r, a in zip(rhos_exact, alpha_centers)])
    rhos_minus = numpy.array([r / correction(a, eps_fixed) for r, a in zip(rhos_exact, alpha_centers)])

    # Convert to SIP format: central value and absolute error
    rho_bar = (rhos_plus + rhos_minus) / 2.0
    rho_delta = (rhos_plus - rhos_minus) / 2.0

    # --- Setup the Epsilon Scan ---
    omega_target = 2.5
    sigma_target = 1.0

    N_eps = 50
    epsilons = numpy.linspace(0.025, 4.0, 50)

    sip_widths = numpy.zeros(N_eps)

    for i, eps in enumerate(tqdm(epsilons)):
        up, low = compute_sip_bounds_at_eps(eps, omega_target, sigma_target, alpha_centers, omega_dense)
        sip_widths[i] = up - low

    # Find optimal Epsilon
    imin = numpy.argmin(sip_widths)
    eps_min = epsilons[imin]
    width_min = sip_widths[imin]

    rk_widths = numpy.zeros(len(epsilons))
    rk_uppers = numpy.zeros(len(epsilons))
    rk_lowers = numpy.zeros(len(epsilons))

    for i, eps in enumerate(epsilons):
        # 1. Regenerate the exact input data vectors given to SIP at this epsilon
        r_ex = numpy.array([rho_eps(a, eps) for a in alpha_centers])
        r_pl = numpy.array([r * correction(a, eps) for r, a in zip(r_ex, alpha_centers)])
        r_mi = numpy.array([r / correction(a, eps) for r, a in zip(r_ex, alpha_centers)])
        
        # 2. Evaluate analytic kernel exactly on alpha_centers
        K_vals = K_gauss_static(alpha_centers, omega_target, eps, sigma_target)
        
        # 3. Construct RK integrands
        term_plus = K_vals * r_pl
        term_minus = K_vals * r_mi
        
        integrand_up = numpy.where(K_vals >= 0, term_plus, term_minus)
        integrand_low = numpy.where(K_vals >= 0, term_minus, term_plus)
        
        # 4. Integrate using only the data points SIP had access to
        rk_up = numpy.trapezoid(integrand_up, alpha_centers)
        rk_low = numpy.trapezoid(integrand_low, alpha_centers)
        rk_widths[i] = rk_up - rk_low

        rk_uppers[i] = max(r_pl-r_mi) * numpy.trapezoid(abs(numpy.array(K_vals)), alpha_centers)
        rk_lowers[i] = numpy.trapezoid(abs(numpy.array(K_vals) * (r_pl-r_mi)), alpha_centers)

    # --- PLOT ---

    plt.figure(figsize=(10,6))
    plt.plot(epsilons[::2], rk_widths[::2], 'o', color='C0', label=r'RK: $[K\rho_\varepsilon^{\mathtt c}]_+ - [K\rho_\varepsilon^{\mathtt c}]_-$', markersize=10)
    plt.plot(epsilons, rk_uppers, color='C1', linewidth=3, label=r'$\Vert \rho_{\varepsilon+}^{\mathtt c} - \rho_{\varepsilon-}^{\mathtt c} \Vert_{\infty} \cdot \Vert K \Vert_1$')
    plt.plot(epsilons, rk_lowers, color='C2', linewidth=3, label=r'$\Vert K \cdot (\rho_{\varepsilon+}^{\mathtt c} - \rho_{\varepsilon-}^{\mathtt c}) \Vert_1$')

    plt.plot(epsilons, sip_widths, color='C3', linewidth=3, markersize=8, label=r'SIP: $\rho[\kappa]_+^{\mathrm{rig}} - \rho[\kappa]_-^{\mathrm{rig}}$')

    epsilons_low = numpy.linspace(float(epsilons[0]), 0.25, 1000)
    # plt.plot(epsilons_low, 10/epsilons_low, linestyle='--', color='k', label=r'$\propto 1/\varepsilon$', linewidth=2)
    plt.plot(epsilons_low, 1/epsilons_low**2, linestyle='--', color='k', label=r'$1/\varepsilon^2$', linewidth=2)

    epsilons_large = numpy.linspace(3.0, float(epsilons[-1]), 1000)
    plt.plot(epsilons_large, 0.1*numpy.exp(epsilons_large**2/2), linestyle=':', color='k', linewidth=2, label=r'$\mathrm{exp}\left\{\varepsilon^2/(2\sigma^2)\right\}$')

    imin = numpy.argmin(rk_widths)
    eps_min = epsilons[imin]
    width_min = rk_widths[imin]
    # vertical dashed line
    # plt.axvline(eps_min, linestyle='--', linewidth=2, color='k', zorder=0)
    # star marker at the minimum
    plt.plot(eps_min, width_min, marker='*', markersize=18, color='k', zorder=5)
    plt.annotate(r'$\varepsilon^\star/\sigma$', xy=(eps_min, width_min), xytext=(5, 10), textcoords='offset points', fontsize=26, ha='center', va='bottom', color='k')

    imin_sip = numpy.argmin(sip_widths)
    eps_min_sip = epsilons[imin_sip]
    plt.plot(eps_min_sip, sip_widths[imin_sip], marker='*', markersize=20, markeredgecolor='black', color='C3', zorder=5)

    plt.text(0.95, 0.95, rf"$\omega'/\sigma={omega_target:.2f}$", 
            transform=plt.gca().transAxes, 
            fontsize=22, 
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)) # Optional: adds a background box

    plt.xlabel(r'$\varepsilon/\sigma$', fontsize=26)
    plt.yscale('log')
    plt.legend(fontsize=22, loc='center', frameon=True, bbox_to_anchor=(0.43, 0.65))
    plt.tight_layout()
    plt.savefig(plot_folder + '/stability_analysis.pdf', bbox_inches='tight')

    # ------------------------------------------------------------------------------
    # PLOT 2: SIP vs RK Stability Analysis
    # ------------------------------------------------------------------------------

    print('[sip] Plot 2: Cauchy to Gaussian optimized reconstructions. This will take 40 minutes...')

    sigma_target = 1.0
    N_scan_global = 30
    omegas1_global = numpy.linspace(0.0, 5.0, N_scan_global)

    exact_global = numpy.zeros(N_scan_global)
    up_global = numpy.zeros(N_scan_global)
    low_global = numpy.zeros(N_scan_global)
    eps_star_global = numpy.zeros(N_scan_global)

    rk_up_global = numpy.zeros(N_scan_global)
    rk_low_global = numpy.zeros(N_scan_global)
    rk_eps_star_global = numpy.zeros(N_scan_global)

    for i, w_target in enumerate(tqdm(omegas1_global)):
        
        # --- SIP ---
        # Reduced alpha_grid to 300 points to accelerate CVXPY (scales like N^3)
        a_grid = numpy.linspace(w_target - 8.0, w_target + 10.0, 300)
        d_grid = numpy.linspace(w_target - 8.0, w_target + 10.0, 1000)
        
        exact_global[i] = rho_sigma(w_target, sigma_target)
        
        e_star, up_val, low_val = optimize_sip_eps(w_target, sigma_target, a_grid, d_grid, max_iters=100, tol=1e-7)
        
        eps_star_global[i] = e_star
        up_global[i] = up_val
        low_global[i] = low_val

        # --- RK ---
        e_star, up_val, low_val = optimize_rk_eps(w_target, sigma_target, d_grid)
    
        rk_eps_star_global[i] = e_star
        rk_up_global[i] = up_val
        rk_low_global[i] = low_val
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                   gridspec_kw=dict(width_ratios=[1], height_ratios=[3, 1]),
                                   figsize=(10, 6))

    # Panel 1: Spectral Density Reconstruction
    # Exact
    ax1.plot(omegas1_global, exact_global, color='C0', label=r"$\rho_\sigma^{\mathtt g}(\omega')$", linestyle='-', linewidth=3, zorder=5)

    # SIP Bands
    ax1.plot(omegas1_global, up_global, color='C1', label=r"$\rho[\kappa]_\pm^{\mathrm{rig}}(\omega')$", linestyle='-', linewidth=2.5)
    ax1.plot(omegas1_global, low_global, color='C1', linestyle='-', linewidth=2.5)
    ax1.fill_between(omegas1_global, low_global, up_global, color='C1', alpha=0.4)

    # RK Bands
    ax1.plot(omegas1_global, rk_up_global, color='C2', label=r"$[K\rho_{\varepsilon^\star}^{\mathtt c}]_\pm(\omega')$", linestyle='--', linewidth=2)
    ax1.plot(omegas1_global, rk_low_global, color='C2', linestyle='--', linewidth=2)
    ax1.fill_between(omegas1_global, rk_low_global, rk_up_global, color='C2', alpha=0.15)

    # Add positions of masses
    ylims = ax1.get_ylim()
    for m, A in zip(ms, As):
        ax1.axvline(m, 0, A / max(As) * ylims[1] * 0.2, linestyle=':', color='gray', linewidth=2)
    ax1.set_ylim(ylims)
    ax1.set_ylabel(r"$\rho_\sigma^{\mathtt g}(\omega')$", fontsize=24)

    ax1.legend(fontsize=20, loc='upper left', ncol=2)
    ax1.grid(True, alpha=0.3)

    # Panel 2: The Optimal Epsilon profiles
    ax2.plot(omegas1_global, eps_star_global, color='C1', linewidth=3, markersize=6, label=r"SIP")
    ax2.plot(omegas1_global, rk_eps_star_global, color='C2', linewidth=2, markersize=6, linestyle='--', label=r"RK")

    ax2.set_xlabel(r"$\omega'/\sigma$", fontsize=24)
    ax2.set_ylabel(r"$\varepsilon^\star / \sigma$", fontsize=24)
    ax2.set_ylim(0.5, 3.5)
    ax2.legend(fontsize=20, loc='upper center', ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.subplots_adjust(wspace=0, hspace=0.05, left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.savefig(plot_folder + '/reconstructions.pdf', bbox_inches='tight')

    print('[sip] Done! Exiting now...')

    return


if __name__ == '__main__':
    main()
