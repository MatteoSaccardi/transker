import numpy
from tqdm.auto import tqdm
import os
import shutil

import random
random.seed(42)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=shutil.which("latex") is not None)
plt.rc('font', family='serif')

import sys
sys.path.append("../")
from modules.bounds import BoundedData
from modules.g2c import CauchySmearing_x
from modules.kernels import cauchy_np as cauchy
from modules.kernels import gaussian_np as gaussian
from modules.kernels import gaussian_to_cauchy_kernel_mp as K_cauchy
from modules.kernels import gaussian_to_cauchy_kernel_x_mp as K_cauchy_x
from modules.transition import SIPTransition, TransitionKernelProblem

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

plot_folder = '../paperplots/levy'

# ==============================================================================
# Model Spectral Density
# ==============================================================================

ms = [0.5, 4.0]
As = [1.0, 2.0]
epsilon_target = 1.0

def rho_sigma_exact(omega, sigma):
    return sum([As[i] * gaussian(omega, ms[i], sigma) for i in range(len(ms))])
rho_sigma = rho_sigma_exact

def rho_cauchy_exact(omega, eps):
    return sum([As[i] * cauchy(omega, ms[i], eps) for i in range(len(ms))])

# Error definitions
sigma_plus = lambda sigma: 0.55 
sigma_minus = lambda sigma: 0.6 
correction_plus = lambda sigma: (sigma_plus(sigma)/sigma)**2
correction_minus = lambda sigma: (sigma_minus(sigma)/sigma)**2

def make_levy_bounded_data(omega_t, sigmas_grid):
    rhos = numpy.array([rho_sigma_exact(omega_t, s) for s in sigmas_grid])
    rhos_plus = numpy.array([r + correction_plus(s) for r, s in zip(rhos, sigmas_grid)])
    rhos_minus = numpy.array([r / (1 + correction_minus(s)) for r, s in zip(rhos, sigmas_grid)])
    return BoundedData(grid=sigmas_grid, exact=rhos, upper=rhos_plus, lower=rhos_minus)

def make_levy_sip_problem(omega_t, data, omega_dense):
    return TransitionKernelProblem(
        param_grid=data.grid,
        omega_grid=omega_dense,
        data=data,
        target_func=lambda w: cauchy(w, omega_t, epsilon_target),
        basis_func=lambda w, sigma: gaussian(w, omega_t, sigma),
    )

def main():

    if not os.path.exists('../paperplots'):
        os.makedirs('../paperplots')

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    print('[levy] Starting...')

    # ------------------------------------------------------------------------------
    # PLOT 1: transition kernel
    # ------------------------------------------------------------------------------

    print('[levy] Plot 1: Cauchy to Gaussian transition kernel. This will take less than a second...')

    xs = numpy.linspace(0.05,10,1000)
    K_cauchy_x_values = numpy.array([K_cauchy_x(x) for x in xs])

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, K_cauchy_x_values, color='C0', linewidth=3)
    ax.set_xlabel(r'$x$', fontsize=26)
    ax.set_ylabel(r'$2x f(x^2;0,1)$', fontsize=26)

    # --- Inset 1: Close to zero, log scale on y ---
    axins1 = ax.inset_axes([0.24, 0.5, 0.32, 0.32])
    x_zoom1 = numpy.linspace(0.05, 0.2, 500)
    y_zoom1 = numpy.array([K_cauchy_x(x) for x in x_zoom1])
    axins1.plot(x_zoom1, y_zoom1, color='C0', linewidth=2)
    axins1.set_yscale('log')
    axins1.set_xlim(0.05, 0.2)

    # connection line
    pp1, p1_1, p1_2 = mark_inset(ax, axins1, loc1=3, loc2=4, fc='none', ec='0.5')
    pp1.set_visible(False)

    # --- Inset 2: Large x ---
    axins2 = ax.inset_axes([0.64, 0.5, 0.32, 0.32])

    x_zoom2 = numpy.linspace(8, 10, 200)
    y_zoom2 = numpy.array([K_cauchy_x(x) for x in x_zoom2])
    axins2.plot(x_zoom2, y_zoom2, color='C0', linewidth=2)

    # Dashed line proportional to x^-2
    # Let's pick a reference point, e.g., x=9
    x_ref = numpy.linspace(8, 10, 200)
    y_mid = K_cauchy_x(9)
    # Proportional constant C
    C = y_mid * (9**2)
    y_ref = C * x_ref**(-2)

    axins2.plot(x_ref, y_ref+0.0003, color='k', label=r'$\propto x^{-2}$', linestyle='--')
    axins2.text(0.5, 0.55, r'$\propto x^{-2}$', transform=axins2.transAxes, fontsize=18, color='k')
    pp1, p1_1, p1_2 = mark_inset(ax, axins2, loc1=3, loc2=4, fc='none', ec='0.5')
    pp1.set_visible(False)

    plt.tight_layout()
    plt.savefig(plot_folder + '/gaussian_to_cauchy_kernel.pdf', bbox_inches='tight')

    # ------------------------------------------------------------------------------
    # PLOT 2: model spectral density and transition kernel
    # ------------------------------------------------------------------------------

    print('[levy] Plot 2: model spectral density and transition kernel. This will take a few seconds...')

    omega1 = 2.5
    epsilon = 1.0

    xs = numpy.linspace(1e-3, 1e3, 100000)
    sigmas = xs * epsilon

    ms = [ 0.5, 4.0 ]
    As = [ 1.0, 2.0 ]

    rho_sigma = lambda omega, sigma: sum([ As[i] * gaussian(omega,ms[i],sigma) 
                                           for i in range(len(ms)) ])

    import random
    random.seed(42)

    sigma_plus = lambda sigma: 0.55 # mpmath.rand()
    sigma_minus = lambda sigma: 0.6 # mpmath.rand()
    p_plus = 2
    p_minus = 2
    correction_plus = lambda sigma: (sigma_plus(sigma)/sigma)**p_plus
    correction_minus = lambda sigma: (sigma_minus(sigma)/sigma)**p_minus

    rhos = numpy.array([ rho_sigma(omega1, sigma) for sigma in sigmas ])
    rhos_plus = numpy.array([ r*(1+correction_minus(sigma)*0)+correction_plus(sigma) 
                              for r,sigma in zip(rhos,sigmas) ])
    rhos_minus = numpy.array([ r/(1+correction_minus(sigma)) 
                               for r,sigma in zip(rhos,sigmas) ])

    K_cauchy_values = numpy.array([K_cauchy(sigma,epsilon) for sigma in sigmas])

    fig, (ax0, ax1, ax_invisible, ax2) = plt.subplots(4, 1, figsize=(10,6), 
                                                      height_ratios=[3, 3, 0.8, 2])

    ax0.plot(sigmas/epsilon, K_cauchy_values, color='C0', linewidth=3)
    ax0.set_ylabel(r'$2\sigma f(\sigma^2;0,\varepsilon^2)$', fontsize=26)

    ax1.plot(sigmas/epsilon, rhos,       label=r'$\rho_\sigma^{\mathtt g}(\omega)$',    
             color='C0', linewidth=3)
    ax1.plot(sigmas/epsilon, rhos_plus,  label=r'$\rho_{\sigma+}^{\mathtt g}(\omega)$', 
             color='C1', linewidth=3)
    ax1.plot(sigmas/epsilon, rhos_minus, label=r'$\rho_{\sigma-}^{\mathtt g}(\omega)$', 
             color='C2', linewidth=3)
    ax1.legend(fontsize=22, ncols=3, loc='upper right')
    ax1.set_ylim(-0.05,0.9)
    xlims = ax1.get_ylim()
    ax1.set_xlim(xlims[0],5+0.05)
    ax0.set_xlim(xlims[0],5+0.05)
    ax1.set_ylabel(r'$\rho_\sigma^{\mathtt g}(\omega)$', fontsize=26)
    ax1.text(0.95, 0.07, rf"$\omega/\varepsilon={omega1/epsilon:.2f}$", 
             transform=ax1.transAxes, 
             fontsize=22, 
             ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)) 

    ax2.plot(sigmas/epsilon, rhos_plus-rhos_minus, color='C3', linewidth=3, 
             label=r'$\rho_{\sigma+}^{\mathtt g}(\omega) - \rho_{\sigma-}^{\mathtt g}(\omega)$')
    ref_point = (rhos_plus-rhos_minus)[0]
    asymptotics = (epsilon/sigmas[0:100])**2
    asymptotics = numpy.array([a*ref_point / float(asymptotics[0]) / 10 for a in asymptotics])
    ax2.plot(sigmas[0:100]/epsilon, asymptotics, linestyle='--', color='k')
    plt.annotate(r'$\propto (\varepsilon/\sigma)^2$', xy=((sigmas/epsilon)[0], asymptotics[0]), 
                 xytext=(10, -45), textcoords='offset points', fontsize=26, ha='left', va='bottom', color='k')

    ax2.set_ylabel(r'$\Delta\rho_{\sigma}^{\mathtt g}(\omega)$', fontsize=26)
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel(r'$\sigma/\varepsilon$', fontsize=26)

    ax_invisible.set_visible(False)
    fig.subplots_adjust(wspace=0, hspace=0, left=0.15, right=0.95, bottom=0.15, top=0.95)

    plt.savefig(plot_folder + '/model_rho.pdf')

    # ------------------------------------------------------------------------------
    # PLOT 3: RK bounds
    # ------------------------------------------------------------------------------

    print('[levy] Plot 3: RK bounds. This will less than a second...')

    x_range = [ 0, int(numpy.where(sigmas > 5)[0][0]) ]
    K_cauchy_values = numpy.array([K_cauchy(sigma,epsilon) 
                                   for sigma in sigmas[x_range[0]:x_range[1]]])

    Krhos = K_cauchy_values * rhos[x_range[0]:x_range[1]]
    Krhos_plus = K_cauchy_values * rhos_plus[x_range[0]:x_range[1]]
    Krhos_minus = K_cauchy_values * rhos_minus[x_range[0]:x_range[1]]

    sigmas_pos = sigmas[x_range[0]:x_range[1]][K_cauchy_values >= 0]
    sigmas_neg = sigmas[x_range[0]:x_range[1]][K_cauchy_values < 0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,6), sharex=True)

    ax1.plot(sigmas_pos, Krhos[K_cauchy_values >= 0], color='C0', linewidth=2,
             label=r"$2\sigma f(\sigma^2;0,\varepsilon^2) \rho_\sigma^{\mathtt g}(\omega)$")
    ax1.plot(sigmas_neg, Krhos[K_cauchy_values <  0], color='C0', linewidth=2, linestyle='--')

    ax1.plot(sigmas_pos, Krhos_plus[K_cauchy_values >= 0], color='C1', linewidth=2,
             label=r"$2\sigma f(\sigma^2;0,\varepsilon^2) \rho_{\sigma+}^{\mathtt g}(\omega)$")
    ax1.plot(sigmas_neg, Krhos_plus[K_cauchy_values <  0], color='C1', linewidth=2, linestyle='--')

    ax1.plot(sigmas_pos, Krhos_minus[K_cauchy_values >= 0], color='C2', linewidth=2,
             label=r"$2\sigma f(\sigma^2;0,\varepsilon^2) \rho_{\sigma-}^{\mathtt g}(\omega)$")
    ax1.plot(sigmas_neg, Krhos_minus[K_cauchy_values <  0], color='C2', linewidth=2, linestyle='--')

    ax1.plot(sigmas_pos, Krhos_plus[K_cauchy_values >= 0], color='k', linewidth=2,
             label=r"$\mathrm{max }[2\sigma f(\sigma^2;0,\varepsilon^2) \rho_\sigma^{\mathtt g}](\sigma)$")
    ax1.fill_between(sigmas_pos, [float(x) for x in Krhos_plus[K_cauchy_values >= 0]], 0, 
                     color='k', alpha=0.1)
    ax1.plot(sigmas_neg, Krhos_minus[K_cauchy_values <  0], color='k', linewidth=2)
    ax1.fill_between(sigmas_neg, [float(x) for x in Krhos_minus[K_cauchy_values <  0]], 0, 
                     color='k', alpha=0.1)

    ax1.legend(fontsize=22, loc='center right')
    ax1.axhline(y=0, linestyle='-', color='k', linewidth=2)

    ########################################

    ax2.plot(sigmas_pos, Krhos[K_cauchy_values >= 0], color='C0', linewidth=2)
    ax2.plot(sigmas_neg, Krhos[K_cauchy_values <  0], color='C0', linewidth=2, linestyle='--')

    ax2.plot(sigmas_pos, Krhos_plus[K_cauchy_values >= 0], color='C1', linewidth=2)
    ax2.plot(sigmas_neg, Krhos_plus[K_cauchy_values <  0], color='C1', linewidth=2, linestyle='--')

    ax2.plot(sigmas_pos, Krhos_minus[K_cauchy_values >= 0], color='C2', linewidth=2)
    ax2.plot(sigmas_neg, Krhos_minus[K_cauchy_values <  0], color='C2', linewidth=2, linestyle='--')

    ax2.plot(sigmas_pos, Krhos_minus[K_cauchy_values >= 0], color='k', linewidth=2,
             label=r"$\mathrm{min }[2\sigma f(\sigma^2;0,\varepsilon^2) \rho_\sigma^{\mathtt g}](\sigma)$")
    ax2.fill_between(sigmas_pos, [float(x) for x in Krhos_minus[K_cauchy_values >= 0]], 0, color='k', alpha=0.1)
    ax2.plot(sigmas_neg, Krhos_plus[K_cauchy_values <  0], color='k', linewidth=2)
    ax2.fill_between(sigmas_neg, [float(x) for x in Krhos_plus[K_cauchy_values <  0]], 0, color='k', alpha=0.1)

    ax2.legend(fontsize=22, loc='upper right', ncols=2)
    ax2.axhline(y=0, linestyle='-', color='k', linewidth=2)
    ax2.set_xlabel(r'$\sigma/\varepsilon$', fontsize=26)
    ax2.text(0.95, 0.25, rf"$\omega/\varepsilon={omega1/epsilon:.2f}$", 
             transform=ax2.transAxes, 
             fontsize=22, 
             ha='right', va='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.subplots_adjust(wspace=0, hspace=0, left=0.15, right=0.95, bottom=0.15, top=0.95)

    plt.savefig(plot_folder + '/RK_bounds.pdf')

    # ------------------------------------------------------------------------------
    # PLOT 4: Reconstructions (RK vs SIP)
    # ------------------------------------------------------------------------------

    print('[levy] Plot 4: reconstructions (RK vs SIP). This will take two minutes...')

    N_scan = 40
    omegas_scan = numpy.linspace(-1.0, 6.0, N_scan)

    exacts = numpy.zeros(N_scan)
    rk_uppers, rk_lowers = numpy.zeros(N_scan), numpy.zeros(N_scan)
    sip_uppers, sip_lowers = numpy.zeros(N_scan), numpy.zeros(N_scan)

    # Shared integration grid (using dimensionless x = sigma / epsilon)
    xs_grid = numpy.linspace(1e-2, 25.0, 1000)
    sigmas_grid = xs_grid * epsilon_target

    print("Running Modular Global Scan: Gaussian -> Cauchy (Levy)")
    for i, omega_t in enumerate(tqdm(omegas_scan)):
        
        # --- A. Exact Data & Synthetic Bounds ---
        exacts[i] = rho_cauchy_exact(omega_t, epsilon_target)
        
        data = make_levy_bounded_data(omega_t, sigmas_grid)
        
        # --- B. RK Bounds (via external modules/g2c.py, CauchySmearing_x) ---
        # Because Levy K > 0 everywhere, mpmath isn't strictly necessary and faster; we still use it, but it is also ok to set use_mp=False
        cs = CauchySmearing_x(xs_grid, data.exact, data.upper, data.lower, use_mp=True)
        _, rk_up, rk_low = cs.rho_cauchy()
        
        rk_uppers[i] = rk_up
        rk_lowers[i] = rk_low
        
        omega_dense = numpy.linspace(omega_t - 15.0, omega_t + 15.0, 500)
        sip_problem = make_levy_sip_problem(omega_t, data, omega_dense)
        sip_interval = SIPTransition(
            sip_problem,
            omega_active_init=[omega_t],
            use_local_search=True,
            scale=True, force_positivity=True
        ).solve_interval()
        sip_uppers[i] = sip_interval.upper.rigorous
        sip_lowers[i] = sip_interval.lower.rigorous

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))

    ax1.plot(omegas_scan, exacts, color='C0', linewidth=3, zorder=5,
             label=r"$\rho_\varepsilon^{\mathtt c}(\omega')$", linestyle='-')

    ax1.plot(omegas_scan, sip_uppers, color='C1', linestyle='-', linewidth=3,
             label=r"$\rho[\kappa]^{\mathrm{rig}}_\pm(\omega')$")
    ax1.plot(omegas_scan, sip_lowers, color='C1', linestyle='-', linewidth=3)
    ax1.fill_between(omegas_scan, sip_uppers, sip_lowers, color='C1', alpha=0.4)

    ax1.plot(omegas_scan, rk_uppers, color='C2', linestyle='--', linewidth=3,
             label=r"$[K\rho_\sigma^{\mathtt g}]_\pm(\omega')$")
    ax1.plot(omegas_scan, rk_lowers, color='C2', linestyle='--', linewidth=3)
    ax1.fill_between(omegas_scan, rk_uppers, rk_lowers, color='C2', alpha=0.15)

    ylims = ax1.get_ylim()
    for i in range(len(ms)):
        ax1.axvline(ms[i], 0, As[i] / max(As) * ylims[1] * 0.3, linestyle='--', 
                    color='C3', linewidth=2)
    ax1.set_ylim(ylims)

    ax1.legend(fontsize=22, loc='upper left')
    ax1.set_xlabel(r"$\omega' / \varepsilon$", fontsize=22)
    ax1.set_ylabel(r"$\rho_\varepsilon^{\mathtt c}(\omega')$", fontsize=22)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.savefig(plot_folder + "/reconstructions.pdf", bbox_inches='tight')

    print('[levy] Done! Exiting now...')

    return


if __name__ == '__main__':
    main()
