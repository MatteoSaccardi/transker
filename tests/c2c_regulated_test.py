import numpy
import scipy.linalg
from tqdm import tqdm
import scipy
import mpmath
import os
import shutil

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=shutil.which("latex") is not None)
plt.rc('font', family='serif')

import sys
sys.path.append("../")
from modules.sip import solve_sip_exchange, compute_sip_certificates

plot_folder = '../paperplots/c2c_regulated'

# ==============================================================================
# Kernel and model spectral density definitions
# ==============================================================================

def cauchy(w, w1, eps):
    return (eps / numpy.pi) / ((w - w1)**2 + eps**2)

ms = [0.5, 4.0]
As = [1.0, 2.0]

def rho_exact_cauchy(omega, eps):
    return sum([As[i] * cauchy(omega, ms[i], eps) for i in range(len(ms))])

# Synthetic Data Errors
def eps0(omega):
    return 0.75
p = 2
def correction(omega, eps): 
    return 1 + (eps0(omega)/eps)**p

def main():

    if not os.path.exists('../paperplots'):
        os.makedirs('../paperplots')

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    print('[c2c_regulated] Starting...')

    # ------------------------------------------------------------------------------
    # PLOT 1: optimization of regulator
    # ------------------------------------------------------------------------------

    print('[c2c_regulated] Plot 1: optimization of regulator. This will take a few seconds...')

    # c2c Sharpening Parameters
    omega_target = 4.0
    eps_in = 2.0
    eps_out = 1.0

    # Grids
    centers_grid = numpy.linspace(omega_target - 8.0, omega_target + 10.0, 300)
    omega_dense = numpy.linspace(omega_target - 20.0, omega_target + 20.0, 1000)

    rhos_exact = numpy.array([rho_exact_cauchy(c, eps_in) for c in centers_grid])
    rhos_plus = numpy.array([r * correction(w, eps_in) 
                             for r, w in zip(rhos_exact, centers_grid)])
    rhos_minus = numpy.array([r / correction(w, eps_in) 
                              for r, w in zip(rhos_exact, centers_grid)])
    rho_delta = (rhos_plus - rhos_minus) / 2.0

    # --- Optimization step ---
    G_mat = cauchy(omega_dense[:, None], centers_grid[None, :], eps_in)
    C_vec = cauchy(omega_dense, omega_target, eps_out)

    # For other values of eps, for finer grids and other setups, 
    # it is better to use mpmath; here, numpy is found to suffice
    GTG = G_mat.T @ G_mat
    GTC = G_mat.T @ C_vec
    I_mat = numpy.eye(len(centers_grid))

    # Array of regulator values to test
    alpha_regs = numpy.logspace(-2, 2, 1000)

    stat_errors = numpy.zeros_like(alpha_regs)
    syst_errors = numpy.zeros_like(alpha_regs)
    total_errors = numpy.zeros_like(alpha_regs)

    for i, alpha_reg in enumerate(alpha_regs):
        # Solve (G^T G + alpha * I) w = G^T C
        g = scipy.linalg.solve(GTG + alpha_reg * I_mat, GTC)
        
        # 1. Systematic Error (Bias)
        K_rec = G_mat @ g
        discrepancy = numpy.abs(C_vec - K_rec)
        
        # --- RIGOROUS OPTIMIZATION OVER w_b ---
        # G_mat is already the matrix of envelopes: G_mat[j, i] = delta(omega_j, centers_grid[i])
        # D_matrix[j, i] evaluates the discrepancy at omega_j divided by the envelope centered at centers_grid[i]
        D_matrix = discrepancy[:, None] / G_mat
        
        # Supremum over omega (axis 0) for each possible envelope center w_b
        supremum_certs = numpy.max(D_matrix, axis=0)
        
        # Multiply by the exact upper bounds at those specific centers
        syst_errs_all_wb = supremum_certs * rhos_plus
        
        # The rigorous systematic error is the MINIMUM across all valid envelope centers
        syst_errors[i] = numpy.min(syst_errs_all_wb)
        
        # 2. Statistical Error (Variance)
        stat_errors[i] = numpy.sum(numpy.abs(g) * rho_delta)
        
        # Total Error
        total_errors[i] = stat_errors[i] + syst_errors[i]

    # Find the exact optimal point
    opt_idx = numpy.argmin(total_errors)
    alpha_opt = alpha_regs[opt_idx]
    error_opt = total_errors[opt_idx]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(alpha_regs, syst_errors, 'C0-', lw=3, 
            label=r'$\delta \cdot \rho_{\varepsilon_1+}^{\mathtt c}$')
    ax.plot(alpha_regs, stat_errors, 'C1--', lw=3, 
            label=r'$\Delta[K\rho_{\varepsilon_1}^{\mathtt c}]$')
    ax.plot(alpha_regs, total_errors, 'k-', lw=3, 
            label=r'Total Error')

    plt.plot(alpha_opt, error_opt, marker='*', markersize=18, color='k', zorder=5)
    plt.annotate(r'$\xi^\star$', xy=(alpha_opt, error_opt), xytext=(5, 10), 
                 textcoords='offset points', fontsize=22, ha='center', va='bottom', color='k')

    plt.text(0.95, 0.65, rf"$\omega'/\varepsilon_2={omega_target/eps_out:.2f}$" + '\n' + 
            rf"$\varepsilon_1/\varepsilon_2={eps_in/eps_out:.2f}$", 
            transform=plt.gca().transAxes, 
            fontsize=22, 
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)) # Optional: adds a background box

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlim(left=alpha_regs[0], right=alpha_regs[-1])

    ax.set_xlabel(r'Regulator $\xi$', fontsize=22)
    ax.set_ylabel(r'RK Errors', fontsize=22)
    ax.legend(loc='lower center', fontsize=22)

    plt.tight_layout()
    plt.savefig(plot_folder + '/optimization.pdf', bbox_inches='tight')

    # ------------------------------------------------------------------------------
    # PLOT 2: Kernel reconstructions (RK vs SIP)
    # ------------------------------------------------------------------------------

    print('[c2c_regulated] Plot 2: kernel reconstructions (RK vs SIP). This will take a few seconds...')

    # --- RK OPTIMIZATION (Optimized over alpha AND omega_b)
    G_mat = cauchy(omega_dense[:, None], centers_grid[None, :], eps_in)
    C_vec = cauchy(omega_dense, omega_target, eps_out)
    GTG = G_mat.T @ G_mat
    GTC = G_mat.T @ C_vec
    I_mat = numpy.eye(len(centers_grid))

    def compute_rk_width(log_alpha):
        
        w = scipy.linalg.solve(GTG + (10**log_alpha) * I_mat, GTC, assume_a='pos')
        K_rec = G_mat @ w
        discrepancy = numpy.abs(C_vec - K_rec)
        D_matrix = discrepancy[:, None] / G_mat
        supremum_certs = numpy.max(D_matrix, axis=0)
        syst_errs = supremum_certs * rhos_plus
        syst_err = numpy.min(syst_errs)
        stat_err = numpy.sum(numpy.abs(w) * rho_delta)
        return syst_err + stat_err

    # --- Optimize regulator alpha
    res = scipy.optimize.minimize_scalar(compute_rk_width, bounds=(-8, 1), method='bounded')
    alpha_opt = 10**res.x
    w_opt = scipy.linalg.solve(GTG + alpha_opt * I_mat, GTC, assume_a='pos')
    K_rec_opt = G_mat @ w_opt

    # Extract Optimal w_b envelope and RK bounds
    discrepancy_opt = numpy.abs(C_vec - K_rec_opt)
    D_matrix_opt = discrepancy_opt[:, None] / G_mat
    supremum_certs_opt = numpy.max(D_matrix_opt, axis=0)
    syst_errs_all_wb = supremum_certs_opt * rhos_plus

    opt_wb_idx = numpy.argmin(syst_errs_all_wb)
    opt_wb = centers_grid[opt_wb_idx]
    opt_supremum = supremum_certs_opt[opt_wb_idx]
    rk_syst_penalty = syst_errs_all_wb[opt_wb_idx]
    bounding_envelope = G_mat[:, opt_wb_idx]

    prod_lower = numpy.minimum(w_opt * rhos_minus, w_opt * rhos_plus)
    prod_upper = numpy.maximum(w_opt * rhos_minus, w_opt * rhos_plus)
    rk_upper = numpy.sum(prod_upper) + rk_syst_penalty
    rk_lower = numpy.sum(prod_lower) - rk_syst_penalty

    # --- SIP reconstruction
    target_f = lambda w: cauchy(w, omega_target, eps_out)
    basis_f = lambda w, a: cauchy(w, a, eps_in)

    rho_bar = (rhos_plus + rhos_minus) / 2.0
    rho_delta = (rhos_plus - rhos_minus) / 2.0

    lam_up, k_up, val_up, diff_up, Phi_d = solve_sip_exchange(
        centers_grid, rho_bar, rho_delta, omega_dense, target_f, basis_f, 
        bound_type='upper', scale=True, force_positivity=False
    )
    deltas_up = compute_sip_certificates(diff_up, Phi_d, 'upper')
    sip_upper = numpy.min(val_up + deltas_up * rhos_plus)

    lam_low, k_low, val_low, diff_low, _ = solve_sip_exchange(
        centers_grid, rho_bar, rho_delta, omega_dense, target_f, basis_f, 
        bound_type='lower', scale=True, force_positivity=False
    )
    deltas_low = compute_sip_certificates(diff_low, Phi_d, 'lower')
    sip_lower = numpy.max(val_low - deltas_low * rhos_plus)

    fig, axs = plt.subplots(nrows=2, ncols=1,
                            sharex=True,
                            gridspec_kw=dict(width_ratios=[1], height_ratios=[3, 1]),
                            figsize=(10, 6))
    axs = axs.ravel()

    xomegas = (omega_dense - omega_target) / eps_out

    # Top: Kernels
    axs[0].plot(xomegas, C_vec, 'k-', lw=3, label=rf"Target")
    axs[0].plot(xomegas, K_rec_opt, 'C1-', alpha=0.7, lw=3, label=rf"RK")
    axs[0].plot(xomegas, k_up, 'C0-', lw=2, label=r"SIP")
    axs[0].plot(xomegas, k_low, 'C0-', lw=2)
    axs[0].fill_between(xomegas, k_up, k_low, color='C0', alpha=0.2)
    axs[0].set_ylabel(r"$\delta_{\varepsilon_2}^{\mathtt c}(\omega-\omega')$", fontsize=22)
    axs[0].legend(fontsize=20, loc="upper left")
    axs[0].set_xlim(-3.5, 3.5)
    # axs[0].grid(alpha=0.3)
    axs[0].text(0.95, 0.95, rf"$\omega'/\varepsilon_2={omega_target/eps_out:.2f}$" + '\n' + rf"$\varepsilon_1/\varepsilon_2={eps_in/eps_out:.2f}$", 
            transform=axs[0].transAxes, 
            fontsize=22, 
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)) # Optional: adds a background box

    # Fix 2: Add zoom-in for x in (x1, x2) in the lower-left region of axs[0]
    x1, x2 = -3.45, -2.5
    axins = axs[0].inset_axes([0.3, 0.08, 0.35, 0.35])
    axins.plot(xomegas, C_vec, 'k-', lw=2)
    axins.plot(xomegas, K_rec_opt, 'C1-', alpha=0.7, lw=2)
    axins.plot(xomegas, k_up, 'C0-', lw=1.5)
    axins.plot(xomegas, k_low, 'C0-', lw=1.5)
    axins.fill_between(xomegas, k_up, k_low, color='C0', alpha=0.2)

    # Set inset limits and format
    axins.set_xlim(x1, x2)
    mask = (xomegas >= x1) & (xomegas <= x2)
    ymax = max(numpy.max(C_vec[mask]), numpy.max(k_up[mask]))
    axins.set_ylim(-0.002, ymax * 1.05)
    axins.tick_params()

    axs[0].indicate_inset_zoom(axins, edgecolor="black")

    # Bottom: RK Certificate
    axs[1].plot(xomegas, discrepancy_opt, 'C1-', lw=2, label=r"RK")
    axs[1].plot(xomegas, bounding_envelope * opt_supremum, 'k--', lw=2, label=rf"Certificate ")#($\omega^\ast/\varepsilon_2={opt_wb/eps_out:.2f}$)")
    axs[1].set_yscale('log')
    axs[1].set_ylabel(r'$|\delta_{\varepsilon_2}^{\mathtt c}-\overline\delta_{\varepsilon_2}^{\mathtt c}|$', fontsize=22)
    axs[1].legend(fontsize=20, ncols=2)
    axs[1].set_xlabel(r"$(\omega-\omega')/\varepsilon_2$", fontsize=22)
    axs[1].set_xlim(-3.5, 3.5)
    axs[1].set_ylim(5e-5, 3e-1)

    fig.subplots_adjust(wspace=0, hspace=0, left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.savefig(plot_folder + '/kernel_comparison.pdf', bbox_inches='tight')

    # ------------------------------------------------------------------------------
    # PLOT 3: energy scan (RK vs SIP)
    # ------------------------------------------------------------------------------

    print('[c2c_regulated] Plot 3: energy scan (RK vs SIP). This will take 8 minutes...')

    eps_in  = 2.0  # Input Smearing
    eps_out = 1.0  # Target Smearing (Sharpening)

    # Energy Scan Grid
    omegas_scan = numpy.linspace(-1.0, 6.0, 200)

    # Buffers for results
    exact_vals        = numpy.zeros(len(omegas_scan))
    rk_ups, rk_lows   = numpy.zeros(len(omegas_scan)), numpy.zeros(len(omegas_scan))
    sip_ups, sip_lows = numpy.zeros(len(omegas_scan)), numpy.zeros(len(omegas_scan))

    # --- THE ENERGY SCAN LOOP ---
    print(f"Scanning energy spectrum: eps_1={eps_in} -> eps_2={eps_out}")
    for i, w_t in enumerate(tqdm(omegas_scan)):
        exact_vals[i] = rho_exact_cauchy(w_t, eps_out)
        
        # Local Grids for the point w_t
        cg = numpy.linspace(w_t - 8.0, w_t + 10.0, 200)
        od = numpy.linspace(w_t - 20.0, w_t + 20.0, 1000)
        
        # Local Data
        r_ex = numpy.array([rho_exact_cauchy(c, eps_in) for c in cg])
        r_pl = r_ex * numpy.array([correction(c, eps_in) for c in cg])
        r_mi = r_ex / numpy.array([correction(c, eps_in) for c in cg])
        rb, rd = (r_pl + r_mi)/2.0, (r_pl - r_mi)/2.0
        
        # --- RK OPTIMIZATION ---
        Gm = cauchy(od[:, None], cg[None, :], eps_in)
        Cv = cauchy(od, w_t, eps_out)
        GtG, GtC, Im = Gm.T @ Gm, Gm.T @ Cv, numpy.eye(len(cg))
        
        def rk_width_obj(la):
            w = scipy.linalg.solve(GtG + (10**la)*Im, GtC, assume_a='pos')
            # Optimized w_b envelope
            sys = numpy.min(numpy.max(numpy.abs(Cv - Gm@w)[:, None] / Gm, axis=0) * r_pl)
            stat = numpy.sum(numpy.abs(w) * rd)
            return sys + stat
        
        res_a = scipy.optimize.minimize_scalar(rk_width_obj, bounds=(-8, 3), method='bounded')
        alpha_opt = 10**res_a.x
        wf = scipy.linalg.solve(GtG + alpha_opt*Im, GtC, assume_a='pos')
        
        # Final RK Bounds
        rk_sys = numpy.min(numpy.max(numpy.abs(Cv - Gm@wf)[:, None] / Gm, axis=0) * r_pl)
        rk_ups[i] = numpy.sum(numpy.maximum(wf*r_mi, wf*r_pl)) + rk_sys
        rk_lows[i] = numpy.sum(numpy.minimum(wf*r_mi, wf*r_pl)) - rk_sys
        
        # --- SIP BOUNDS ---
        tf = lambda w: cauchy(w, w_t, eps_out)
        bf = lambda w, a: cauchy(w, a, eps_in)
        
        _, _, vu, du, Pd = solve_sip_exchange(cg, rb, rd, od, tf, bf, bound_type='upper', scale=True, force_positivity=False)
        _, _, vl, dl, _  = solve_sip_exchange(cg, rb, rd, od, tf, bf, bound_type='lower', scale=True, force_positivity=False)
        
        sip_ups[i] = numpy.min(vu + compute_sip_certificates(du, Pd, 'upper') * r_pl)
        sip_lows[i] = numpy.max(vl - compute_sip_certificates(dl, Pd, 'lower') * r_pl)

    plt.figure(figsize=(10, 6))

    xs = omegas_scan / eps_out

    # Exact Target
    plt.plot(xs, exact_vals, 'C0', lw=3, label=r"$\rho_{\varepsilon_2}^{\mathtt c}(\omega')$")

    # Input Data (More Smeared)
    input_rho = [rho_exact_cauchy(w, eps_in) for w in omegas_scan]
    input_rho_plus = input_rho * numpy.array([correction(w, eps_in) for w in omegas_scan])
    input_rho_minus = input_rho / numpy.array([correction(w, eps_in) for w in omegas_scan])
    plt.plot(xs, input_rho_plus, 'k-', lw=1.5, 
             label=r"$\rho_{\varepsilon_1\pm}^{\mathtt c}(\omega')$")
    plt.plot(xs, input_rho_minus, 'k-', lw=1.5)
    plt.fill_between(xs, input_rho_minus, input_rho_plus, color='k', alpha=0.4)

    # RK Region
    NSKIP = 1
    plt.fill_between(xs, rk_lows, rk_ups, color='C2', alpha=0.15)
    plt.plot(xs[::NSKIP], rk_ups[::NSKIP], 'C2--', lw=3, 
             label=r"$[K\rho_{\varepsilon_1}^{\mathtt c}]_\pm(\omega')$")
    plt.plot(xs[::NSKIP], rk_lows[::NSKIP], 'C2--', lw=3)

    # SIP Points
    plt.plot(xs, sip_ups, 'C1', label=r"$\rho[\kappa]_\pm^{\mathrm{rig}}(\omega')$")
    plt.plot(xs, sip_lows, 'C1')
    plt.fill_between(xs, sip_lows, sip_ups, color='C1', alpha=0.4)

    plt.text(0.95, 0.95, rf"$\varepsilon_1/\varepsilon_2={eps_in/eps_out:.2f}$", 
             transform=plt.gca().transAxes, 
             fontsize=20, 
             ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)) 

    plt.xlabel(r"$\omega'/\varepsilon_2$", fontsize=20)
    plt.ylabel(r"$\rho_{\varepsilon_2}^{\mathtt c}(\omega')$", fontsize=20)
    plt.legend(fontsize=20, ncols=2)
    plt.tight_layout()
    plt.savefig(plot_folder + '/reconstructions.pdf', bbox_inches='tight')

    print('[c2c_regulated] Done! Exiting now...')

    return

if __name__ == '__main__':
    main()
