import numpy
from tqdm import tqdm
import scipy
import random
import shutil
random.seed(42)

import sys
sys.path.append("../")
from modules.bounds import BoundedData
from modules.c2g import c2g
from modules.kernels import cauchy_mp as cauchy
from modules.kernels import cauchy_to_gaussian_kernel_mp as K_gauss
from modules.kernels import gaussian_mp as gaussian

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({'font.size': 16})
plt.rc('text', usetex=shutil.which("latex") is not None)
plt.rc('font', family='serif')

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os

plot_folder = '../paperplots/c2g'

def make_c2g_bounded_data(omega_grid, eps, ms, As, correction):
    rho_eps = lambda omega: sum([As[i] * cauchy(omega, ms[i], eps) for i in range(len(ms))])
    rhos = numpy.array([rho_eps(omega) for omega in omega_grid])
    rhos_plus = numpy.array([r * correction(omega, eps) for r, omega in zip(rhos, omega_grid)])
    rhos_minus = numpy.array([r / correction(omega, eps) for r, omega in zip(rhos, omega_grid)])
    return BoundedData(grid=omega_grid, exact=rhos, upper=rhos_plus, lower=rhos_minus)

def main():
    if not os.path.exists('../paperplots'):
        os.makedirs('../paperplots')

    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    print('[c2g] Starting...')

    # ------------------------------------------------------------------------------
    # PLOT 1: transition kernel
    # ------------------------------------------------------------------------------

    print('[c2g] Plot 1: Cauchy to Gaussian transition kernel. This will take less than a second...')

    sigma = 1.

    omegas = numpy.linspace(-5,5,1000)
    epsilons = [ 0.75, 1., 1.25 ]
    K_gauss_values = numpy.array([[K_gauss(omega,0,eps,sigma) 
                                   for omega in omegas] for eps in epsilons])
    gaussians = numpy.array([gaussian(omega,0,sigma) for omega in omegas])

    plt.figure(figsize=(10,6))
    plt.plot(omegas, gaussians, label=r'$\varepsilon/\sigma=0$', 
             color='C0', linewidth=3)
    for ieps,eps in enumerate(epsilons):
        plt.plot(omegas, K_gauss_values[ieps], 
                 label=rf'$\varepsilon/\sigma={eps:.2f}$', color=f'C{ieps+1}', linewidth=3)
    plt.legend(fontsize=26)
    plt.xlabel(r"$(\omega'-\omega)/\sigma$", fontsize=26)
    plt.ylabel(r"$K_{\sigma \leftarrow \varepsilon}^{\mathtt{g} \leftarrow \mathtt{c}}(\omega'-\omega)$", 
               fontsize=26)
    plt.tight_layout()
    plt.savefig(plot_folder+'/cauchy_to_gaussian_kernel.pdf', bbox_inches='tight')

    # ------------------------------------------------------------------------------
    # PLOT 2: Model spectral density and RK bounds
    # ------------------------------------------------------------------------------

    print('[c2g] Plot 2: Model spectral density and RK bounds. This will take a minute...')

    # Transition details
    sigma = 1.0
    omega1 = 2.5

    omegas = numpy.linspace(omega1-4.0,omega1+4.0,100000)
    eps = 2.00
    K_gauss_values = numpy.array([K_gauss(omega1,omega,eps,sigma) for omega in omegas])
    K_pos = numpy.where(K_gauss_values >= 0, K_gauss_values, numpy.nan)
    K_neg = numpy.where(K_gauss_values < 0, K_gauss_values, numpy.nan)

    # Model details
    ms = [ 0.5, 4.0 ]
    As = [ 1.0, 2.0 ]

    eps0 = lambda omega: 0.75 # mpmath.rand()
    p = 2
    correction = lambda omega, eps: 1 + (eps0(omega)/eps)**p

    data = make_c2g_bounded_data(omegas, eps, ms, As, correction)
    rhos = data.exact
    rhos_plus = data.upper
    rhos_minus = data.lower
    
    # Compute RK bounds
    Krhos = K_gauss_values * rhos
    Krhos_plus = K_gauss_values * rhos_plus
    Krhos_minus = K_gauss_values * rhos_minus

    omegas_pos = omegas[K_gauss_values >= 0]
    omegas_neg = omegas[K_gauss_values < 0]

    # Plot
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True, sharex=True)

    # ==============================================
    # (0,0) Kernel K
    # ==============================================
    ax = axs[0, 0]

    ax.plot(omegas, K_pos, label=r'$K>0$', color='C0', linewidth=3)
    ax.plot(omegas, K_neg, label=r'$K<0$', color='C0', linewidth=3, linestyle='--')

    ax.axhline(0, color='k', linewidth=3)
    ax.axvline(x=omega1, linestyle=':', color='k', linewidth=3)

    ax.set_yscale('symlog', linthresh=1e-1)
    # ax.set_xlabel(r"$\omega/\sigma$", fontsize=20)
    ax.set_ylabel(r"$K_{\sigma \leftarrow \varepsilon}^{\mathtt{g} \leftarrow \mathtt{c}}(\omega'-\omega)$",
                  fontsize=30)
    ax.legend(fontsize=24, loc='upper right')

    # ==============================================
    # (0,1) Spectral density + bounds
    # ==============================================
    ax = axs[1, 0]

    ax.plot(omegas, rhos, label=r'$\rho_\varepsilon^{\mathtt c}(\omega)$', 
            color='C0', linewidth=3)
    ax.plot(omegas, rhos_plus, label=r'$\rho_{\varepsilon+}^{\mathtt c}(\omega)$', 
            color='C1', linewidth=3)
    ax.plot(omegas, rhos_minus, label=r'$\rho_{\varepsilon-}^{\mathtt c}(\omega)$', 
            color='C2', linewidth=3)

    ax.axvline(x=omega1, linestyle=':', color='k', linewidth=3)
    ax.annotate(r"$\omega'/\sigma$", xy=(omega1 - 1.5, 0.02),
                xycoords='data', fontsize=30)

    ylims = ax.get_ylim()
    for i in range(len(ms)):
        ax.axvline(ms[i], 0, As[i] / max(As) * ylims[1] * 0.9,
                   linestyle='--', color='C3', linewidth=2)
    ax.set_ylim(ylims)

    ax.set_xlabel(r"$\omega/\sigma$", fontsize=30)
    ax.legend(fontsize=24, loc='upper right')

    # ==============================================
    # (1,0) Max envelope of K rho
    # ==============================================
    ax = axs[0, 1]

    ax.plot(omegas_pos, Krhos[K_gauss_values >= 0],
            color='C0', linewidth=3)
    ax.plot(omegas_neg, Krhos[K_gauss_values < 0],
            color='C0', linewidth=2, linestyle='--')

    ax.plot(omegas_pos, Krhos_plus[K_gauss_values >= 0],
            color='C1', linewidth=3)
    ax.plot(omegas_neg, Krhos_plus[K_gauss_values < 0],
            color='C1', linewidth=2, linestyle='--')

    ax.plot(omegas_pos, Krhos_minus[K_gauss_values >= 0],
            color='C2', linewidth=3)
    ax.plot(omegas_neg, Krhos_minus[K_gauss_values < 0],
            color='C2', linewidth=2, linestyle='--')

    ax.plot(omegas_pos, Krhos_plus[K_gauss_values >= 0],
            label=r"$\max(K\rho)$", color='k', linewidth=3)
    ax.fill_between(omegas_pos, [float(x) for x in Krhos_plus[K_gauss_values >= 0]],
                    0, color='k', alpha=0.1, label=r'$[K\rho_\varepsilon^{\mathtt c}]_-$')

    ax.plot(omegas_neg, Krhos_minus[K_gauss_values < 0],
            color='k', linewidth=3)
    ax.fill_between(omegas_neg,
                    [float(x) for x in Krhos_minus[K_gauss_values < 0]],
                    0, color='k', alpha=0.1)

    ax.axhline(0, color='k', linewidth=3)
    ax.axvline(x=omega1, linestyle=':', color='k', linewidth=3)

    ax.set_yscale('symlog', linthresh=0.02)
    ax.legend(fontsize=24, loc='center right', bbox_to_anchor=(1.02, 0.75))

    # ==============================================
    # (1,1) Min envelope of K rho
    # ==============================================
    ax = axs[1, 1]

    ax.plot(omegas_pos, Krhos[K_gauss_values >= 0],
            color='C0', linewidth=3)
    ax.plot(omegas_neg, Krhos[K_gauss_values < 0],
            color='C0', linewidth=2, linestyle='--')

    ax.plot(omegas_pos, Krhos_plus[K_gauss_values >= 0],
            color='C1', linewidth=3)
    ax.plot(omegas_neg, Krhos_plus[K_gauss_values < 0],
            color='C1', linewidth=2, linestyle='--')

    ax.plot(omegas_pos, Krhos_minus[K_gauss_values >= 0],
            color='C2', linewidth=3)
    ax.plot(omegas_neg, Krhos_minus[K_gauss_values < 0],
            color='C2', linewidth=2, linestyle='--')

    ax.plot(omegas_pos, Krhos_minus[K_gauss_values >= 0],
            label=r"$\min(K\rho)$", color='k', linewidth=3)
    ax.fill_between(omegas_pos,
                    [float(x) for x in Krhos_minus[K_gauss_values >= 0]],
                    0, color='k', alpha=0.1, label=r'$[K\rho_\varepsilon^{\mathtt c}]_-$')

    ax.plot(omegas_neg, Krhos_plus[K_gauss_values < 0],
            color='k', linewidth=3)
    ax.fill_between(omegas_neg, [float(x) for x in Krhos_plus[K_gauss_values < 0]],
                    0, color='k', alpha=0.1)

    ax.axhline(0, color='k', linewidth=3)
    ax.axvline(x=omega1, linestyle=':', color='k', linewidth=3)

    ax.set_yscale('symlog', linthresh=0.02)
    ax.set_xlabel(r"$\omega/\sigma$", fontsize=30)
    ax.legend(fontsize=24, loc='center right', bbox_to_anchor=(1.02, 0.75))

    shared_handles = [
        Line2D([0], [0], color='C0', lw=3, linestyle='-',
                label=r"$K\rho_\varepsilon^{\mathtt c}$"),
        Line2D([0], [0], color='C1', lw=3, linestyle='-',
                label=r"$K\rho_{\varepsilon+}^{\mathtt c}$"),
        Line2D([0], [0], color='C2', lw=3, linestyle='-',
                label=r"$K\rho_{\varepsilon-}^{\mathtt c}$"),
    ]
    fig.legend(handles=shared_handles, loc='center', bbox_to_anchor=(0.75, 0.54),
               ncol=3, fontsize=24, frameon=True)

    for ax in axs[0, :]:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    labels = ['(a)', '(b)', '(c)', '(d)']
    for ax, lab in zip(axs.flat, labels):
        ax.text(0.02, 0.90, lab, transform=ax.transAxes, fontsize=24, va='top', ha='left')

    plt.tight_layout()

    plt.savefig(plot_folder+'/RK_bounds.pdf', bbox_inches='tight')

    # ------------------------------------------------------------------------------
    # PLOT 3: Stability analysis for RK bounds
    # ------------------------------------------------------------------------------

    print('[c2g] Plot 3: Stability analysis for RK bounds. This will take a minute...')

    sigma = 1.0
    omega1 = 2.5

    epsilons = numpy.arange(0.025, 4.0+0.025, 0.025)

    omegas = numpy.linspace(omega1-8.0,omega1+10.0,1000)

    widths = numpy.zeros(len(epsilons))
    uppers = numpy.zeros(len(epsilons))
    lowers = numpy.zeros(len(epsilons))
    for ieps,eps in enumerate(tqdm(epsilons)):
        data = make_c2g_bounded_data(omegas, eps, ms, As, correction)

        gs = c2g(omegas, data.exact, data.upper, data.lower, eps, use_mp=True)

        res, up, low = gs.rho_gauss(omega1, sigma)

        width = up - low

        upper = max(data.width) * numpy.trapezoid(abs(numpy.array(gs.Kvals)), omegas)
        lower = numpy.trapezoid(abs(numpy.array(gs.Kvals) * data.width), omegas)

        widths[ieps] = width
        uppers[ieps] = upper
        lowers[ieps] = lower

    plt.figure(figsize=(10,6))
    plt.plot(epsilons[::5], widths[::5], 'o', color='C0', linewidth=3, 
             label=r'$[K\rho_\varepsilon^{\mathtt c}]_+ - [K\rho_\varepsilon^{\mathtt c}]_-$', 
             markersize=10)
    plt.plot(epsilons, uppers, color='C1', linewidth=3, 
             label=r'$\Vert \rho_{\varepsilon+}^{\mathtt c} - \rho_{\varepsilon-}^{\mathtt c} \Vert_{\infty} \cdot \Vert K \Vert_1$')
    plt.plot(epsilons, lowers, color='C2', linewidth=3, 
             label=r'$\Vert K \cdot (\rho_{\varepsilon+}^{\mathtt c} - \rho_{\varepsilon-}^{\mathtt c}) \Vert_1$')

    epsilons_low = numpy.linspace(float(epsilons[0]), 0.25, 1000)
    plt.plot(epsilons_low, 1/epsilons_low**2, linestyle='--', color='k', 
             label=r'$1/\varepsilon^2$', linewidth=2)

    epsilons_large = numpy.linspace(3.0, float(epsilons[-1]), 1000)
    plt.plot(epsilons_large, 0.1*numpy.exp(epsilons_large**2/2), linestyle=':', color='k', 
             linewidth=2, label=r'$\mathrm{exp}\left\{\varepsilon^2/(2\sigma^2)\right\}$')

    imin = numpy.argmin(widths)
    eps_min = epsilons[imin]
    width_min = widths[imin]
    plt.plot(eps_min, width_min, marker='*', markersize=18, color='k', zorder=5)
    plt.annotate(r'$\varepsilon^\star/\sigma$', xy=(eps_min, width_min), 
                 xytext=(5, 10), textcoords='offset points', fontsize=26, 
                 ha='center', va='bottom', color='k')

    plt.text(0.95, 0.95, rf"$\omega'/\sigma={omega1:.2f}$", 
             transform=plt.gca().transAxes, 
             fontsize=22, 
             ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.xlabel(r'$\varepsilon/\sigma$', fontsize=26)
    plt.yscale('log')
    plt.legend(fontsize=22, loc='center', frameon=True, bbox_to_anchor=(0.43, 0.65))
    plt.tight_layout()
    plt.savefig(plot_folder+'/stability_analysis.pdf', bbox_inches='tight')

    # ------------------------------------------------------------------------------
    # PLOT 4: Energy scan with optimized inputs
    # ------------------------------------------------------------------------------
    
    print('[c2g] Plot 4: Energy scan with optimized inputs. This will take 5 minutes...')

    def width_at_eps(eps, omega1, sigma, omega_grid, print_progress=False):        
        if print_progress: 
            print(f'eps = {eps:.3f}', end='\r')

        data = make_c2g_bounded_data(omega_grid, eps, ms, As, correction)

        gs = c2g(omega_grid, data.exact, data.upper, data.lower, eps, use_mp=True)

        _, up, low = gs.rho_gauss(omega1, sigma)
        return float(up - low)
    
    sigma = 1.
    omegas1 = numpy.linspace(-1,6,100)

    rho_sigma = lambda omega: sum([ As[i] * gaussian(omega,ms[i],sigma) for i in range(len(ms)) ])
    exacts = numpy.array([ rho_sigma(omega1) for omega1 in omegas1 ])

    uppers = numpy.zeros(len(omegas1))
    lowers = numpy.zeros(len(omegas1))
    centers = numpy.zeros(len(omegas1))
    eps_stars = numpy.zeros(len(omegas1))

    for iomega1,omega1 in enumerate(tqdm(omegas1)):
        omegas = numpy.linspace(omega1-8.0,omega1+10.0,1000)

        res = scipy.optimize.minimize_scalar(lambda eps: width_at_eps(eps, omega1, sigma, omegas), 
                                             bounds=(0.02, 4.0), method='bounded', 
                                             options={'xatol': 1e-3})
        eps_star  = res.x
        eps_stars[iomega1] = eps_star

        data = make_c2g_bounded_data(omegas, eps_star, ms, As, correction)

        gs = c2g(omegas, data.exact, data.upper, data.lower, eps_star, use_mp=True)
        res, up, low = gs.rho_gauss(omega1, sigma)

        uppers[iomega1] = up
        lowers[iomega1] = low
        centers[iomega1] = res

    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,
                                   sharex=True,
                                   gridspec_kw=dict(width_ratios=[1], height_ratios=[3, 1]),
                                   figsize=(10, 6))

    ax1.plot(omegas1, exacts, color='C0', linestyle='-', linewidth=3,
             label=r"$\rho_\sigma^{\mathtt g}(\omega')$")
    ax1.plot(omegas1, uppers, color='C1', linestyle='-', linewidth=3,
             label=r"$[K\rho_{\varepsilon^\star}^{\mathtt c}]_+(\omega')$")
    ax1.plot(omegas1, lowers, color='C1', linestyle='--', linewidth=3,
             label=r"$[K\rho_{\varepsilon^\star}^{\mathtt c}]_-(\omega')$")
    ax1.fill_between(omegas1, lowers, uppers, color='C1', alpha=0.3)
    ylims = ax1.get_ylim()
    for i in range(len(ms)):
        ax1.axvline(ms[i], 0, As[i] / max(As) * ylims[1] * 0.5, linestyle='--', 
                    color='C3', linewidth=2)
    ax1.set_ylim(ylims)
    ax1.legend(fontsize=24, loc='upper left')
    ax1.set_ylabel(r"$\rho_\sigma^{\mathtt g}(\omega')$", fontsize=24)    # ax1.grid()

    ax2.plot(omegas1, eps_stars, color='C0', linewidth=3, 
             label=r"$\varepsilon^\star(\omega')/\sigma$")
    ax2.set_xlabel(r"$\omega'/\sigma$", fontsize=24)
    ax2.set_ylabel(r"$\varepsilon^\star(\omega')/\sigma$", fontsize=24)

    fig.subplots_adjust(wspace=0, hspace=0, left=0.15, right=0.95, bottom=0.15, top=0.95)
    plt.savefig(plot_folder+'/reconstructions.pdf', bbox_inches='tight')

    print('[c2g] Done! Exiting now...')

    return

if __name__ == '__main__':
    main()
