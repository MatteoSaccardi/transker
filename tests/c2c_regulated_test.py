import numpy
from tqdm import tqdm
import os
import shutil

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=shutil.which("latex") is not None)
plt.rc('font', family='serif')

import sys
sys.path.append("../")
from modules.bounds import BoundedData
from modules.kernels import cauchy_np as cauchy
from modules.transition import RegulatedRKTransition, SIPTransition, TransitionKernelProblem

plot_folder = '../paperplots/c2c_regulated'

# ==============================================================================
# Kernel and model spectral density definitions
# ==============================================================================

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

def make_c2c_problem(omega_target, eps_in, eps_out, centers_grid, omega_dense):
    rhos_exact = numpy.array([rho_exact_cauchy(c, eps_in) for c in centers_grid])
    rhos_plus = numpy.array([r * correction(w, eps_in) for r, w in zip(rhos_exact, centers_grid)])
    rhos_minus = numpy.array([r / correction(w, eps_in) for r, w in zip(rhos_exact, centers_grid)])
    data = BoundedData(grid=centers_grid, exact=rhos_exact, upper=rhos_plus, lower=rhos_minus)
    return TransitionKernelProblem(
        param_grid=centers_grid,
        omega_grid=omega_dense,
        data=data,
        target_func=lambda w: cauchy(w, omega_target, eps_out),
        basis_func=lambda w, a: cauchy(w, a, eps_in),
    )

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

    problem = make_c2c_problem(omega_target, eps_in, eps_out, centers_grid, omega_dense)
    rk_transition = RegulatedRKTransition(problem)

    # --- Optimization step ---
    G_mat = rk_transition.basis_matrix
    C_vec = rk_transition.target_vector

    # For other values of eps, for finer grids and other setups, 
    # it is better to use mpmath; here, numpy is found to suffice

    # Array of regulator values to test
    alpha_regs = numpy.logspace(-2, 2, 1000)

    stat_errors = numpy.zeros_like(alpha_regs)
    syst_errors = numpy.zeros_like(alpha_regs)
    total_errors = numpy.zeros_like(alpha_regs)

    for i, alpha_reg in enumerate(alpha_regs):
        result = rk_transition.evaluate(alpha_reg)
        syst_errors[i] = result.systematic_error
        stat_errors[i] = result.statistical_error
        total_errors[i] = result.total_error

    # Find the exact optimal point
    opt_idx = numpy.argmin(total_errors)
    alpha_opt = alpha_regs[opt_idx]
    error_opt = total_errors[opt_idx]

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(alpha_regs, syst_errors, 'C0-', lw=3, 
            label=r'$\Delta_{\mathrm{prop}}(\xi)$')
    ax.plot(alpha_regs, stat_errors, 'C1--', lw=3, 
            label=r'$\Delta_{\mathrm{sys}}(\xi)$')
    ax.plot(alpha_regs, total_errors, 'k-', lw=3, 
            label=r'$\Delta_{\mathrm{tot}}(\xi)$')

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
    _, rk_interval = rk_transition.optimize_log_alpha(bounds=(-8, 1), RK_method="symmetric")
    rk_result = rk_interval.upper
    alpha_opt = rk_result.alpha
    w_opt = rk_result.coefficients
    K_rec_opt = rk_result.reconstruction
    discrepancy_opt = rk_result.discrepancy
    opt_wb_idx = rk_result.certificate_index
    opt_supremum = rk_result.certificate
    bounding_envelope = G_mat[:, opt_wb_idx]

    # --- SIP reconstruction
    sip_interval = SIPTransition(
        problem,
        scale=True,
        force_positivity=False,
    ).solve_interval()
    k_up = sip_interval.upper.kappa
    k_low = sip_interval.lower.kappa
    sip_upper = sip_interval.upper.rigorous
    sip_lower = sip_interval.lower.rigorous

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

    print('[c2c_regulated] Plot 3: energy scan (RK vs SIP). This will take about 10 minutes...')

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
        
        problem_i = make_c2c_problem(w_t, eps_in, eps_out, cg, od)
        
        # --- RK OPTIMIZATION ---
        _, rk_interval_i = RegulatedRKTransition(problem_i).optimize_log_alpha(bounds=(-8, 3))
        rk_ups[i] = rk_interval_i.upper_bound
        rk_lows[i] = rk_interval_i.lower_bound
        
        # --- SIP BOUNDS ---
        sip_interval_i = SIPTransition(
            problem_i,
            scale=True,
            force_positivity=False,
        ).solve_interval()
        sip_ups[i] = sip_interval_i.upper.rigorous
        sip_lows[i] = sip_interval_i.lower.rigorous

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
