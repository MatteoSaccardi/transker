#!/usr/bin/env python3

import importlib
import importlib.metadata
import os
import shutil
import sys
import tempfile


CORE_PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("mpmath", "mpmath"),
    ("tqdm", "tqdm"),
    ("cvxpy", "cvxpy"),
]

PREFERRED_SOLVER = "CLARABEL"
FALLBACK_SOLVERS = ["SCS", "ECOS"]


def configure_matplotlib_cache():
    if "MPLCONFIGDIR" in os.environ:
        return

    cache_dir = os.path.join(tempfile.gettempdir(), "transker-matplotlib")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = cache_dir


def package_version(distribution_name):
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def check_imports():
    ok = True
    print("Python packages:")

    for module_name, distribution_name in CORE_PACKAGES:
        try:
            importlib.import_module(module_name)
            version = package_version(distribution_name)
            print(f"  [ok]      {module_name} {version}")
        except Exception as exc:
            ok = False
            print(f"  [missing] {module_name}: {exc}")

    return ok


def solve_tiny_problem(cvxpy_module, solver_name):
    x = cvxpy_module.Variable()
    problem = cvxpy_module.Problem(
        cvxpy_module.Minimize(x),
        [x >= 1],
    )
    problem.solve(solver=solver_name, verbose=False)
    return x.value is not None and abs(float(x.value) - 1.0) < 1e-5


def check_cvxpy_solvers():
    try:
        import cvxpy as cp
    except Exception:
        print()
        print("CVXPY solvers: skipped because cvxpy could not be imported")
        return False

    installed = set(cp.installed_solvers())
    supported = [PREFERRED_SOLVER] + FALLBACK_SOLVERS
    ok = True

    print()
    print(f"CVXPY version: {package_version('cvxpy')}")
    print("Installed CVXPY solvers:")
    if installed:
        print("  " + ", ".join(sorted(installed)))
    else:
        print("  none detected")

    print()
    print("Project solver status:")
    if PREFERRED_SOLVER in installed:
        print(f"  [ok]      {PREFERRED_SOLVER} preferred solver is installed")
    else:
        ok = False
        print(f"  [missing] {PREFERRED_SOLVER} preferred solver is not installed")

    for solver_name in FALLBACK_SOLVERS:
        if solver_name in installed:
            print(f"  [ok]      {solver_name} fallback solver is installed")
        else:
            print(f"  [warn]    {solver_name} fallback solver is not installed")

    if not any(solver_name in installed for solver_name in supported):
        ok = False
        print("  [error]   no supported solver is installed")

    print()
    print("Tiny solver smoke tests:")
    for solver_name in supported:
        if solver_name not in installed:
            print(f"  [skip]    {solver_name}")
            continue

        try:
            if solve_tiny_problem(cp, solver_name):
                print(f"  [ok]      {solver_name}")
            else:
                ok = False
                print(f"  [failed]  {solver_name}: unexpected solution")
        except Exception as exc:
            if solver_name == PREFERRED_SOLVER:
                ok = False
            print(f"  [failed]  {solver_name}: {exc}")

    return ok


def check_latex():
    print()
    print("External tools:")
    latex = shutil.which("latex")
    if latex:
        print(f"  [ok]      latex found at {latex}")
        return True

    print("  [warn]    latex not found; plots will use Matplotlib mathtext instead of usetex")
    return True


def check_matplotlib_cache():
    print()
    print("Matplotlib cache:")
    cache_dir = os.environ.get("MPLCONFIGDIR")
    if cache_dir and os.access(cache_dir, os.W_OK):
        print(f"  [ok]      MPLCONFIGDIR is writable: {cache_dir}")
        return True

    default_dir = os.path.join(os.path.expanduser("~"), ".matplotlib")
    if os.access(default_dir, os.W_OK):
        print(f"  [ok]      default cache is writable: {default_dir}")
        return True

    print("  [warn]    matplotlib cache directory is not writable")
    print("            set MPLCONFIGDIR to a writable directory before running plot tests")
    return True


def main():
    configure_matplotlib_cache()

    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print()

    imports_ok = check_imports()
    solvers_ok = check_cvxpy_solvers()
    check_latex()
    check_matplotlib_cache()

    print()
    if imports_ok and solvers_ok:
        print("Environment check passed.")
        return 0

    print("Environment check failed.")
    print("Try: python3 -m pip install -r requirements.txt")
    print("If only solvers are missing, try: python3 -m pip install clarabel scs ecos")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
