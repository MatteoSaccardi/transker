# transker

`transker` contains reusable Python utilities and reproducibility scripts for
transition kernels between smeared spectral densities. It implements analytic
RK-style transitions, semi-infinite-programming (SIP) bounds with CVXPY
certificates, bounded-data helpers, and the plotting scripts used to reproduce
the figures of the paper
[Kernel Transformations etc...](https://arxiv.org/abs/1504.00108).
[INSERT CORRECT LINK!!!!!]

The main workflows currently cover:

- Cauchy-to-Gaussian transition kernels (`modules/c2g.py`)
- Gaussian-to-Cauchy from Levy kernels (`modules/g2c.py`)
- Generic bounded transition-kernel problems (`modules/transition.py`)
- Semi-infinite-programming solvers and certificates (`modules/sip.py`)
- Finite-window inverse-Laplace SIP bounds for noisy correlator data (`modules/ilt.py`)
- Environment checks, smoke tests, and long plot-generation scripts

## Quick Start Checklist

For a fresh checkout, the practical path is:

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run `python3 check_install.py` and confirm that the environment passes.
4. Run the smoke tests first with `./run_tests.sh smoke`.
5. To reproduce the paper plots, run the scripts through `./run_tests.sh`.

The sections below give the exact commands and troubleshooting notes.

## Repository Layout

```text
modules/
  __init__.py             package marker for `modules.*` imports
  kernels.py              shared kernel functions with NumPy and mpmath backends
  bounds.py               bounded-data container and bound conversion helpers
  transition.py           generic transition-kernel problem and solver wrappers
  c2g.py                  Cauchy-to-Gaussian transition kernel utilities
  g2c.py                  Gaussian-to-Cauchy Levy kernel utilities
  sip.py                  SIP exchange solver and certificate computation
  ilt.py                  inverse-Laplace SIP bounds for covariance-constrained correlator data

tests/
  smoke_test.py           fast import/API/solver smoke tests
  c2g_test.py             Cauchy-to-Gaussian plots, with RK bounds propagation
  levy_test.py            Levy Gaussian-to-Cauchy plots
  sip_test.py             SIP stability and reconstruction plots for Cauchy-to-Gaussian transitions
  c2c_regulated_test.py   regulated Cauchy-to-Cauchy plots
  ilt_sip_plot.py         finite-window inverse-Laplace SIP plot for Gaussian/Cauchy smearings

check_install.py          environment and solver diagnostic
paperplots_original/      reference plots kept for comparison
run_tests.sh              interactive test/plot runner
requirements.txt          Python dependencies
```

Generated figures are automatically written under `paperplots/`.
The `paperplots_original/` directory is kept as a
reference copy for comparison with newly generated figures.

## Installation

Create and activate a virtual environment from the repository root:

```bash
python3 -m venv .transker
source .transker/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Then check that the environment is usable:

```bash
python3 check_install.py
```

The diagnostic checks the core scientific Python packages, CVXPY, available
CVXPY solvers, LaTeX, and Matplotlib cache writability.

## CVXPY Solvers

The SIP code uses CVXPY. The preferred solver is:

- `CLARABEL`

Fallback solvers checked by the diagnostic are:

- `SCS`
- `ECOS`

`requirements.txt` includes `clarabel` and `scs`. If solver installation is the
only failing part of the diagnostic, try:

```bash
python3 -m pip install clarabel scs ecos
```

`CLARABEL` is the solver to prefer for the SIP runs. `SCS` is useful as a
fallback, but it requires a more careful tuning.

## Running The Tests / Plot Scripts

Use the script from the repository root:

```bash
./run_tests.sh
```

It presents a numbered menu:

```text
1) all
2) smoke
3) c2c
4) c2g
5) levy
6) sip
7) ilt-sip-plot
```

Paste one or more numbers separated by spaces, for example:

```text
3 5
```

You can also run non-interactively:

```bash
./run_tests.sh all
./run_tests.sh smoke
./run_tests.sh c2g sip
./run_tests.sh 4 6
./run_tests.sh ilt
```

The runner changes into `tests/` before executing the Python files. This matters
because the test scripts use relative imports such as:

```python
sys.path.append("../")
```

Each selected script reports its actual elapsed time, and the runner prints a
total at the end. On a MacBook Air M2 with 16GB memory, representative timings
are:

```text
smoke: 0m 04s
c2c:   10m 40s
c2g:   6m 50s
levy:  2m 04s
sip:   31m 54s
ilt-sip-plot: 4m 12s
total: 55m 34s
```

`run_tests.sh` uses `python3` by default. If you want to force a specific
interpreter, set `PYTHON`:

```bash
PYTHON=.transker/bin/python ./run_tests.sh smoke
```

## Programmatic API

The reusable pieces live in the `modules` package and can be imported from the
repository root:

```python
import numpy

from modules.bounds import BoundedData
from modules.kernels import cauchy_np
from modules.transition import TransitionKernelProblem, SIPTransition

param_grid = numpy.linspace(-1.0, 1.0, 50)
omega_grid = numpy.linspace(-2.0, 2.0, 200)
exact = numpy.array([cauchy_np(a, 0.0, 1.0) for a in param_grid])
data = BoundedData(param_grid, exact, exact * 1.1, exact * 0.9)

problem = TransitionKernelProblem(
    param_grid=param_grid,
    omega_grid=omega_grid,
    data=data,
    target_func=lambda w: cauchy_np(w, 0.0, 1.0),
    basis_func=lambda w, a: cauchy_np(w, a, 1.0),
)

interval = SIPTransition(problem).solve_interval()
print(interval.lower.rigorous, interval.upper.rigorous)
```

If the exact values are not known, construct the data directly from the
admissible interval. In that case `exact` is set to the midpoint `bar`:

```python
data = BoundedData.from_bounds(param_grid, upper_values, lower_values)
```

For regulated RK-style reconstructions, use:

```python
from modules.transition import RegulatedRKTransition

_, interval = RegulatedRKTransition(problem).optimize_log_alpha(bounds=(-8, 1))
print(interval.lower_bound, interval.upper_bound)
```

By default, regulated RK bounds use `RK_method="asymmetric"`, which optimizes
the upper and lower bounds separately. Use `RK_method="symmetric"` to use one
regulator optimized by total error for both bounds.

## Running A Single Script Manually

If you do not use `run_tests.sh`, run scripts from inside `tests/`:

```bash
cd tests
python3 c2g_test.py
python3 levy_test.py
python3 sip_test.py
python3 c2c_regulated_test.py
python3 ilt_sip_plot.py
```

Running these files from the repository root is not guaranteed to work because of their relative
import setup.

## LaTeX And Matplotlib

The plotting scripts enable TeX rendering only when a `latex` executable is
available:

```python
plt.rc("text", usetex=shutil.which("latex") is not None)
```

LaTeX is therefore optional. With LaTeX installed, plots use paper-style TeX
rendering. Without LaTeX, the scripts fall back to Matplotlib's built-in
mathtext and should still run. On macOS, MacTeX or BasicTeX are typical options
if you want full TeX rendering. `check_install.py` reports whether a `latex`
executable is visible on `PATH`.

If Matplotlib reports that its cache directory is not writable, set:

```bash
export MPLCONFIGDIR=/tmp/transker-matplotlib
```

before running the plotting scripts.

## License

This code is distributed under the GNU General Public License version 2 or, at
your option, any later version; see `LICENSE` for the GPL v2 terms. The files in
`modules/` carry matching GPL source headers and the standard no-warranty notice.

## Typical Workflow

```bash
source .transker/bin/activate
python3 check_install.py
./run_tests.sh
```

For a quick check, run only the shorter script first:

```bash
./run_tests.sh smoke
```
