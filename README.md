# transker

`transker` contains numerical experiments and plotting scripts for transition
kernels between smeared spectral densities. The repository includes reusable
Python modules plus reproducibility scripts that generate the plots of
the paper [Kernel Transformations etc...](https://arxiv.org/abs/1504.00108).
[INSERT CORRECT LINK!!!!!]

The main workflows currently cover:

- Cauchy-to-Gaussian transition kernels (`modules/c2g.py`)
- Gaussian-to-Cauchy from Levy kernels (`modules/g2c.py`)
- Semi-infinite-programming bounds and certificates (`modules/sip.py`)
- Plot-generation tests under `tests/`

## Quick Start Checklist

For a fresh checkout, the practical path is:

1. Create and activate a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Run `python3 check_install.py` and confirm that the environment passes.
4. Run a short test first, for example `./run_tests.sh levy`.
5. To reproduce the paper plots, run the scripts through `./run_tests.sh`.

The sections below give the exact commands and troubleshooting notes.

## Repository Layout

```text
modules/
  c2g.py                  Cauchy-to-Gaussian transition kernel utilities
  g2c.py                  Gaussian-to-Cauchy Levy kernel utilities
  sip.py                  SIP exchange solver and certificate computation

tests/
  c2g_test.py             Cauchy-to-Gaussian plots, with RK bounds propagation
  levy_test.py            Levy Gaussian-to-Cauchy plots
  sip_test.py             SIP stability and reconstruction plots for Cauchy-to-Gaussian transitions
  c2c_regulated_test.py   regulated Cauchy-to-Cauchy plots

check_install.py          environment and solver diagnostic
run_tests.sh              interactive test/plot runner
requirements.txt          Python dependencies
```

Generated figures are automatically written under `paperplots/`.
The scripts under `plots/` can be run through `run_tests.sh`.

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
2) c2c
3) c2g
4) levy
5) sip
```

Paste one or more numbers separated by spaces, for example:

```text
3 5
```

You can also run non-interactively:

```bash
./run_tests.sh all
./run_tests.sh c2g sip
./run_tests.sh 3 5
```

The runner changes into `tests/` before executing the Python files. This matters
because the test scripts use relative imports such as:

```python
sys.path.append("../")
```

Each selected script reports its actual elapsed time, and the runner prints a
total at the end. Some runs are intentionally long; SIP and regulated C2C runs
can take tens of minutes (10 and 40, respectively).

## Running A Single Script Manually

If you do not use `run_tests.sh`, run scripts from inside `tests/`:

```bash
cd tests
python3 c2g_test.py
python3 levy_test.py
python3 sip_test.py
python3 c2c_regulated_test.py
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

## Typical Workflow

```bash
source .transker/bin/activate
python3 check_install.py
./run_tests.sh
```

For a quick check, run only the shorter script first:

```bash
./run_tests.sh levy
```
