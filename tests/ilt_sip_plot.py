import os
import shutil
import sys

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib-transker"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.ilt import ILTProblem, ILTSolver, correlator_from_peaks, rho_kappa_from_peaks
from modules.kernels import cauchy_np, gaussian_np

PLOT_FOLDER = "../paperplots/ilt_sip"


def make_mock_data():
    nt = 16
    energy_threshold = 0.10
    energy_spacing = 0.02
    n_state = 100
    gamma_inv = 1.3
    gamma = 1.0 / gamma_inv
    s_floor = 0.05
    base_rel_err = 0.01
    noise_growth_rate = 0.20
    random_seed = 42

    k = numpy.arange(n_state)
    energies = energy_threshold + energy_spacing * (k + 1)
    z = numpy.zeros(n_state)
    for n in range(n_state):
        if n == 0:
            z[n] = 1.0 / 5.0
        elif n % 2 == 1:
            z[n] = 1.0
        else:
            z[n] = -1.0 / 5.0
    weights = z**2
    weights *= cauchy_np(energies, 1.15, 0.15)

    t_values = numpy.arange(1, nt + 1, dtype=float)
    c_true = correlator_from_peaks(t_values, energies, weights)

    rel_err = base_rel_err * numpy.exp(noise_growth_rate * t_values)
    covariance = numpy.zeros((nt, nt))
    for i in range(nt):
        for j in range(nt):
            corr = (s_floor + (1.0 - s_floor) * (i == j)) * numpy.exp(-gamma * abs(i - j))
            covariance[i, j] = (rel_err[i] * c_true[i]) * (rel_err[j] * c_true[j]) * corr

    rng = numpy.random.default_rng(random_seed)
    c_obs = rng.multivariate_normal(c_true, covariance)
    chi2_actual = float((c_obs - c_true) @ numpy.linalg.solve(covariance, c_obs - c_true))
    sigma0 = numpy.sqrt(chi2_actual)

    return {
        "energy_threshold": energy_threshold,
        "energies": energies,
        "weights": weights,
        "t_values": t_values,
        "c_obs": c_obs,
        "covariance": covariance,
        "sigma0": sigma0,
    }


KERNEL_SPECS = {
    "gaussian": {"func": gaussian_np, "color": "C0"},
    "cauchy": {"func": cauchy_np, "color": "C1"},
}


def make_centered_kernel(kind, center, width):
    kernel_np = KERNEL_SPECS[kind]["func"]
    return lambda omega: kernel_np(omega, center, width)


def exact_smeared_observable(data, kind, center, width, cutoff=None):
    kernel_func = make_centered_kernel(kind, center, width)
    energies = data["energies"]
    weights = data["weights"]
    if cutoff is None:
        mask = numpy.ones_like(energies, dtype=bool)
    else:
        mask = (energies >= data["energy_threshold"]) & (energies <= cutoff)
    return rho_kappa_from_peaks(kernel_func, energies[mask], weights[mask])


def build_problem(data, kind, center, width, cutoff, threshold_eps=1e-7, covariance_reg=0.0):
    return ILTProblem(
        t_values=data["t_values"],
        c_obs=data["c_obs"],
        covariance=data["covariance"],
        sigma0=data["sigma0"],
        kernel_func=make_centered_kernel(kind, center, width),
        kernel_kind=kind,
        energy_threshold=data["energy_threshold"],
        threshold_eps=threshold_eps,
        omega_max=cutoff,
        finite_cutoff=cutoff,
        covariance_reg=covariance_reg,
    )


def corrected_bound(workflow, bound_type):
    if bound_type in workflow.corrections:
        return workflow.corrections[bound_type].corrected_value
    return workflow.results[bound_type].value


def solve_kernel_case(data, kind, center, width, cutoff, settings):
    problem = build_problem(data, kind, center=center, width=width, cutoff=cutoff)
    workflow = ILTSolver(
        problem,
        solver=settings["solver"],
        n_initial=settings["n_initial"],
        n_check=settings["n_check"],
        max_iters=settings["max_iters"],
        violation_tol=settings["violation_tol"],
    )
    workflow.solve_bounds()
    workflow.optimize_t0_bounds(include_t0_zero=False)
    return {
        "center": float(center),
        "exact_full": exact_smeared_observable(data, kind, center, width, cutoff=None),
        "exact_truncated": exact_smeared_observable(data, kind, center, width, cutoff=cutoff),
        "lower": corrected_bound(workflow, "lower"),
        "upper": corrected_bound(workflow, "upper"),
        "workflow": workflow,
    }


def scan_centers_for_kernel(data, kind, centers, width, cutoff, settings):
    rows = []
    for i, center in enumerate(centers):
        row = solve_kernel_case(data, kind, float(center), width, cutoff, settings)
        print(
            f"[ilt-sip-plot] {kind:8s} ({i + 1}/{len(centers)})",
            end="\r",
            flush=True,
        )
        rows.append(row)
    print(" " * 100, end="\r")
    print(f"[ilt-sip-plot] {kind:8s} scan complete")
    dtype = [
        ("center", float),
        ("exact_full", float),
        ("exact_truncated", float),
        ("lower", float),
        ("upper", float),
    ]
    return numpy.array([tuple(row[name] for name, _ in dtype) for row in rows], dtype=dtype)


def make_final_plot(gaussian_scan, cauchy_scan, width, output_path):
    plt.rcParams.update({"font.size": 15})
    plt.rc("text", usetex=shutil.which("latex") is not None)
    plt.rc("font", family="serif")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )

    ax = axes[0]
    for scan, color, label, label_sip in (
        (
            gaussian_scan,
            "C0",
            r"$\rho_\sigma^{\mathtt{g}}(\omega^\prime)$",
            r"$\rho[\delta_\sigma^{\mathtt{g}}]_\pm^{\mathrm{rig}}(\omega^\prime)$",
        ),
        (
            cauchy_scan,
            "C1",
            r"$\rho_\sigma^{\mathtt{c}}(\omega^\prime)$",
            r"$\rho[\delta_\sigma^{\mathtt{c}}]_\pm^{\mathrm{rig}}(\omega^\prime)$",
        ),
    ):
        x = scan["center"] / width
        ax.plot(x, scan["exact_full"], "-", color=color, lw=2, label=label)
        ax.fill_between(x, scan["lower"], scan["upper"], color=color, alpha=0.20, linewidth=0, label=label_sip)

    ax.set_ylabel(r"$\rho[\kappa]$", fontsize=24)
    ax.grid(True, alpha=0.3)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin - 0.20 * (ymax - ymin), ymax)
    ax.legend(fontsize=22, ncol=2, loc="lower center")

    ratio_ax = axes[1]
    for scan, color in ((gaussian_scan, "C0"), (cauchy_scan, "C1")):
        center = scan["center"]
        lower = scan["lower"]
        upper = scan["upper"]
        exact = scan["exact_truncated"]
        interval_width = 0.5 * (upper - lower)
        midpoint = 0.5 * (upper + lower)
        normalized_offset = numpy.divide(
            exact - midpoint,
            interval_width,
            out=numpy.full_like(exact, numpy.nan, dtype=float),
            where=interval_width > 0.0,
        )
        ratio_ax.plot(center / width, normalized_offset, "-", lw=2, color=color)

    ratio_ax.set_ylabel(r"$\frac{\mathrm{exact}-\mathrm{center}}{\mathrm{width}}$", fontsize=24)
    ratio_ax.set_xlabel(r"$\omega'/\sigma$", fontsize=24)
    ratio_ax.set_yticks([-1.0, 0.0, 1.0])
    ratio_ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    return fig


def main():
    os.makedirs(PLOT_FOLDER, exist_ok=True)

    data = make_mock_data()
    width = float(os.environ.get("ILT_SIP_WIDTH", "0.5"))
    n_centers = int(os.environ.get("ILT_SIP_NCENTERS", "50"))
    center_max = float(os.environ.get("ILT_SIP_CENTER_MAX", "2.2"))
    cutoff = float(os.environ.get("ILT_SIP_CUTOFF", str(2.0 * numpy.max(data["energies"]))))
    output_path = os.environ.get("ILT_SIP_OUTPUT", os.path.join(PLOT_FOLDER, "sip_ILT.pdf"))
    settings = {
        "solver": os.environ.get("ILT_SIP_SOLVER") or None,
        "n_initial": int(os.environ.get("ILT_SIP_N_INITIAL", "1200")),
        "n_check": int(os.environ.get("ILT_SIP_N_CHECK", "5000")),
        "max_iters": int(os.environ.get("ILT_SIP_MAX_ITERS", "60")),
        "violation_tol": float(os.environ.get("ILT_SIP_VIOLATION_TOL", "3e-8")),
    }

    centers = numpy.linspace(0.01, center_max, n_centers)
    print(f"[ilt-sip-plot] writing {output_path}")

    gaussian_scan = scan_centers_for_kernel(data, "gaussian", centers, width, cutoff, settings)
    cauchy_scan = scan_centers_for_kernel(data, "cauchy", centers, width, cutoff, settings)
    make_final_plot(gaussian_scan, cauchy_scan, width, output_path)
    print(f"[ilt-sip-plot] wrote {output_path}")


if __name__ == "__main__":
    main()
