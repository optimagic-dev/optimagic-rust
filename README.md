# optimagic-rust

> [!WARNING]
> **This project is experimental.** It is at a very early stage. The API, the package
> names and the build setup can change at any time, without deprecation period. Do not
> depend on it. Nothing has been published on PyPI or conda-forge yet.

optimagic-rust provides fast, low-level building blocks for numerical optimizers, such
as trust-region subproblem solvers and line search routines. They are written in Rust
and exposed to Python.

It is **not** a Rust port of [optimagic](https://github.com/optimagic-dev/optimagic).
Instead, optimagic (and other Python optimizers) can depend on it to speed up their
inner loops.

Currently the repository contains a single, deliberately naive algorithm: gradient
descent with a backtracking (Armijo) line search. It exists to set up and exercise the
full toolchain (Rust crate, Python bindings, wheels, conda package, CI) as a vertical
slice.

## Contents

- [Using the package](#using-the-package)
- [How the pieces fit together](#how-the-pieces-fit-together)
- [Setting up a development environment](#setting-up-a-development-environment)
- [Everyday development](#everyday-development)
- [Adding a new function](#adding-a-new-function)
- [Building packages](#building-packages)
- [Continuous integration](#continuous-integration)
- [Releasing](#releasing)
- [Using optimagic-rust from optimagic](#using-optimagic-rust-from-optimagic)
- [Design decisions](#design-decisions)
- [Open decisions and future work](#open-decisions-and-future-work)
- [Rust primer for Python developers](#rust-primer-for-python-developers)
- [Troubleshooting](#troubleshooting)

## Using the package

### Installation

Once released, the package will be installable with any of these tools:

| Tool        | Command                                                                     |
| ----------- | --------------------------------------------------------------------------- |
| pip         | `pip install optimagic-rust`                                                |
| uv          | `uv add optimagic-rust` (project) or `uv pip install optimagic-rust` (venv) |
| conda/mamba | `conda install -c conda-forge optimagic-rust`                               |
| pixi        | `pixi add optimagic-rust` (conda-forge) or `pixi add --pypi optimagic-rust` |

pip and uv download a pre-compiled wheel from PyPI, so users don't need Rust. conda and
pixi get a pre-compiled package from conda-forge.

Until the first release, install the development version from GitHub. This compiles the
Rust code on your machine, so you need a Rust toolchain (see
[step 1](#1-install-the-tools-once-per-machine)):

```bash
pip install "optimagic-rust @ git+https://github.com/optimagic-dev/optimagic-rust"
```

### Example

```python
import numpy as np
import optimagic_rust as omr


def fun(x):
    return float(x @ x)


def jac(x):
    return 2 * x


res = omr.minimize_gradient_descent(fun, jac, x0=np.array([1.0, 2.0]))
res.x  # array([0., 0.])
res.converged  # True
```

## How the pieces fit together

```
your Python code
   │   import optimagic_rust
   ▼
python/optimagic_rust/__init__.py     pure Python; re-exports everything from _core
python/optimagic_rust/_core.pyi       type stub: tells IDEs and ty what _core contains
   │
   ▼
optimagic_rust._core                  compiled extension module (_core.abi3.so / .pyd)
 = crates/optimagic-rust-py           Rust + PyO3: converts types, maps errors
   │   plain Rust function calls
   ▼
crates/optimagic-rust                 pure Rust: the algorithms and their tests
```

What happens when you call `minimize_gradient_descent(fun, jac, x0)`:

1. PyO3 converts `x0` (a list, tuple or numpy array) into a read-only view of a float64
   numpy array (`PyArrayLike1<f64, AllowTypeChange>`), which Rust sees as an
   `ndarray::ArrayView1<f64>`.
1. The Python callables `fun` and `jac` are wrapped into Rust closures
   (`crates/optimagic-rust-py/src/callbacks.rs`). Each call hands Python a fresh numpy
   copy of the current point.
1. The pure-Rust algorithm runs and calls these closures whenever it needs a function
   value or gradient.
1. Errors come back as `OptimizationError`. Invalid input becomes a `ValueError`;
   exceptions raised inside `fun` or `jac` are re-raised unchanged, so users see their
   own traceback.
1. The Rust result struct is converted into the Python class `GradientDescentResult`,
   whose `x` attribute is a numpy array.

### Repository layout

```
.
├── Cargo.toml                 Cargo workspace: shared version, dependencies, lints
├── Cargo.lock                 exact versions of all Rust dependencies (committed)
├── rust-toolchain.toml        pins the Rust compiler version (read by rustup)
├── crates/
│   ├── optimagic-rust/        pure-Rust library (no Python dependency)
│   └── optimagic-rust-py/     PyO3 bindings, compiled into optimagic_rust._core
├── python/optimagic_rust/     Python package: __init__.py, _core.pyi stub, py.typed
├── tests/                     Python tests (pytest)
├── pyproject.toml             Python metadata, maturin, ruff, pytest, ty and pixi config
├── pixi.lock                  exact versions of all conda/PyPI packages (committed)
├── recipe/                    conda recipe for rattler-build
├── .github/workflows/         CI: tests (main.yml), wheels and PyPI release (release.yml)
└── .pre-commit-config.yaml    hooks run by prek / pre-commit.ci
```

### Tools in this repository

| Tool                                                            | Role                                                       | Python analogue        |
| --------------------------------------------------------------- | ---------------------------------------------------------- | ---------------------- |
| [rustup](https://rustup.rs)                                     | installs and switches Rust compiler versions               | pyenv                  |
| [cargo](https://doc.rust-lang.org/cargo/)                       | builds Rust code, runs Rust tests, manages Rust packages   | pip + pytest + build   |
| [clippy](https://doc.rust-lang.org/clippy/)                     | Rust linter                                                | ruff check             |
| [rustfmt](https://rust-lang.github.io/rustfmt/)                 | Rust formatter                                             | ruff format            |
| [PyO3](https://pyo3.rs)                                         | Rust library to write Python extension modules             | Cython / pybind11      |
| [rust-numpy](https://docs.rs/numpy)                             | passes numpy arrays to and from Rust                       | numpy C API            |
| [ndarray](https://docs.rs/ndarray)                              | n-dimensional arrays in Rust                               | numpy                  |
| [maturin](https://www.maturin.rs)                               | build backend: compiles the crate and packages it as wheel | hatchling / setuptools |
| [pixi](https://pixi.sh)                                         | Python environments and task runner (as in optimagic)      |                        |
| [rattler-build](https://rattler.build)                          | builds conda packages                                      | conda-build            |
| [prek](https://github.com/j178/prek)                            | runs pre-commit hooks (as in optimagic)                    |                        |
| [ruff](https://docs.astral.sh/ruff/)                            | Python linter and formatter                                |                        |
| [ty](https://docs.astral.sh/ty/)                                | Python type checker (as in optimagic)                      | mypy                   |
| [stubtest](https://mypy.readthedocs.io/en/stable/stubtest.html) | checks the type stub against the compiled module           |                        |

## Setting up a development environment

### 1. Install the tools (once per machine)

**A C toolchain.** Rust uses the system linker.

- macOS: `xcode-select --install`
- Linux: install `gcc` (e.g. `sudo apt install build-essential`)
- Windows: install the
  [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
  with the workload "Desktop development with C++".

**Rust, via rustup.** rustup installs the Rust compiler (`rustc`), `cargo`, `clippy` and
`rustfmt`, and switches between versions.

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
# Windows: download and run rustup-init.exe from https://rustup.rs
```

Accept the defaults. Open a new terminal afterwards, so that `~/.cargo/bin` is on your
`PATH`. You don't need to choose a Rust version: the first time you run `cargo` in this
repository, rustup reads `rust-toolchain.toml` and installs the pinned version. To
update rustup itself later, run `rustup self update`.

**pixi**, as for optimagic:

```bash
# macOS / Linux
curl -fsSL https://pixi.sh/install.sh | sh
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
```

**prek**, the pre-commit hook runner used by optimagic:

```bash
pixi global install prek  # or: uv tool install prek
```

Check that everything is there:

```bash
cargo --version && rustup --version && pixi --version && prek --version
```

### 2. Clone the repository and run the tests

```bash
git clone https://github.com/optimagic-dev/optimagic-rust.git
cd optimagic-rust
pixi run tests      # first run: installs Rust 1.99.1 and all dependencies, compiles
prek install        # run the hooks automatically on every commit
```

The first `pixi run tests` takes a minute or two. It creates the environment
`.pixi/envs/default`, downloads and compiles the Rust dependencies into `target/`,
installs `optimagic-rust` into the environment and runs the Rust and Python tests.
Afterwards a full test run takes a few seconds.

### 3. Set up your editor

In VS Code, install the extensions **rust-analyzer**, **Python** and **Even Better
TOML**, and select `.pixi/envs/default/bin/python` (Windows:
`.pixi\envs\default\python.exe`) as Python interpreter.

rust-analyzer compiles the code in the background. Like cargo, it needs a Python >= 3.12
to compile the PyO3 bindings (see [below](#run-cargo-through-pixi)). The simplest way is
to start VS Code from the pixi environment:

```bash
pixi run code .
```

Alternatively, add this to `.vscode/settings.json`, with the absolute path to your
checkout:

```json
{
  "rust-analyzer.cargo.extraEnv": {
    "PYO3_PYTHON": "/absolute/path/to/optimagic-rust/.pixi/envs/default/bin/python"
  }
}
```

## Everyday development

### Run cargo through pixi

To compile the bindings, PyO3 needs a Python interpreter, which it finds via
`VIRTUAL_ENV`, `CONDA_PREFIX` or `PATH`. If you run plain `cargo build` in a shell where
e.g. a conda base environment with Python 3.10 is active, it fails with:

```
error: cannot set a minimum Python version 3.12 higher than the interpreter version 3.10
```

Therefore, run Rust commands through pixi (`pixi run cargo ...`, or the tasks below), or
inside `pixi shell`. The pure-Rust crate does not need Python, so
`cargo test -p optimagic-rust` and `cargo fmt` work anywhere.

### Commands

| What                                                    | Command                                               |
| ------------------------------------------------------- | ----------------------------------------------------- |
| Run all tests (Rust + Python)                           | `pixi run tests`                                      |
| Run the Rust tests (incl. doctests)                     | `pixi run tests-rust`                                 |
| Compile the extension and run the Python tests          | `pixi run tests-python`                               |
| Python tests with another Python version                | `pixi run -e tests-py312 tests-python`                |
| Compile the extension and install it into the env       | `pixi run develop`                                    |
| Run selected Python tests (after `develop`)             | `pixi run pytest tests/test_line_search.py -k ascent` |
| Run selected Rust tests                                 | `pixi run cargo test -p optimagic-rust line_search`   |
| Lint Rust                                               | `pixi run clippy`                                     |
| Format Rust                                             | `cargo fmt --all`                                     |
| Type check Python (incl. tests)                         | `pixi run -e type-checking ty`                        |
| Check the type stub against the compiled module         | `pixi run -e type-checking stubtest`                  |
| Run all pre-commit hooks (ruff, cargo fmt, clippy, ...) | `prek run --all-files`                                |
| Open the Rust API docs                                  | `pixi run docs-rust`                                  |
| List all tasks                                          | `pixi task list`                                      |

### The edit, compile, test loop

- **Python files** (`python/optimagic_rust/*.py`, `tests/`): `pixi run develop` installs
  the package in editable mode, so changes are visible immediately.
- **Rust files**: Python only sees the changes after recompiling with
  `pixi run develop`. `pixi run tests-python` does this automatically. cargo only
  recompiles what changed, so this is fast when nothing changed.
- `develop` compiles in debug mode: fast to compile, slow to run, with extra checks such
  as integer overflow detection. For benchmarks, use `pixi run develop-release`.

### Testing strategy

- **Rust unit tests** live next to the code they test, in a `#[cfg(test)] mod tests`
  block at the end of each file (see `crates/optimagic-rust/src/line_search.rs`). They
  test the algorithms. Code examples in `///` doc comments are compiled and run as
  doctests.
- **Python tests** in `tests/` test the binding layer: conversion of inputs (lists, int
  arrays), types of results, keyword-only options and the mapping of errors to Python
  exceptions. Plus one end-to-end check per function that it computes the right thing.
- **Stub tests**: `stubtest` checks that names, signatures and default values in
  `_core.pyi` match the compiled module; `tests/test_stubs.py` checks that the
  docstrings match and that everything is exported.
- Warnings are turned into errors (`filterwarnings = ["error"]` in `pyproject.toml`).

### Code quality checks

`prek install` sets up a git hook that runs on every commit (on changed files only):
ruff (lint + format), yamllint/yamlfix, mdformat for the README, standard checks (TOML,
YAML, trailing whitespace, ...), `cargo fmt` and `cargo clippy`. As in optimagic, a hook
refuses commits to `main`; work on branches.

Rust lints are configured in the workspace `Cargo.toml`: every public item needs a doc
comment (`missing_docs`) and `unsafe` code is forbidden. CI runs clippy with
`-D warnings`, so any warning fails the build.

## Adding a new function

Say you want to add `solve_trust_region_subproblem`.

1. **Implement it in pure Rust**: create `crates/optimagic-rust/src/trust_region.rs` and
   add `pub mod trust_region;` to `crates/optimagic-rust/src/lib.rs`. Take user-provided
   functions as closures returning `Result<_, E>` and return
   `Result<_, OptimizationError<E>>`. Write unit tests in the same file. Run
   `pixi run tests-rust`.

1. **Write the binding**: create `crates/optimagic-rust-py/src/trust_region.rs` with a
   `#[pyfunction]` (and a `#[pyclass(frozen, get_all)]` for the result). Its `///` doc
   comment becomes the Python docstring, so write it in Google style for Python users.
   Use `#[pyo3(signature = (...))]` to make options keyword-only with defaults. Export
   it in `crates/optimagic-rust-py/src/lib.rs`:

   ```rust
   #[pymodule_export]
   use crate::trust_region::{TrustRegionResult, solve_trust_region_subproblem};
   ```

1. **Declare it in the stub** `python/optimagic_rust/_core.pyi`: signature with type
   hints, the same docstring, and add the name to `__all__`.

1. **Re-export it** in `python/optimagic_rust/__init__.py` (import and `__all__`).

1. **Test the binding** in `tests/test_trust_region.py`.

1. **Check everything**: `pixi run tests`, `pixi run -e type-checking ty`,
   `pixi run -e type-checking stubtest`, `prek run --all-files`.

1. Add an entry to `CHANGES.md`.

### Binding cheat sheet

| Python                    | Rust parameter type in a `#[pyfunction]`            | In the core crate        |
| ------------------------- | --------------------------------------------------- | ------------------------ |
| float                     | `f64`                                               | `f64`                    |
| int (non-negative)        | `usize` (negative ints raise `OverflowError`)       | `usize`                  |
| 1d array-like (any dtype) | `PyArrayLike1<f64, AllowTypeChange>`, `.as_array()` | `ArrayView1<f64>`        |
| 2d array-like             | `PyArrayLike2<f64, AllowTypeChange>`, `.as_array()` | `ArrayView2<f64>`        |
| callable                  | `&Bound<'_, PyAny>`, wrap with `callbacks.rs`       | `FnMut(...) -> Result<>` |
| returned array            | `array.into_pyarray(py)`                            | `Array1<f64>`            |
| returned result object    | `#[pyclass(frozen, get_all)]` struct                | plain struct             |
| raised exception          | `PyValueError::new_err(msg)`, see `errors.rs`       | `OptimizationError`      |

For long pure-Rust computations that don't call back into Python (e.g. a subproblem
solver on matrices), release the Python interpreter with `py.detach(|| ...)` so other
Python threads can run. Learn more in the [PyO3 guide](https://pyo3.rs).

## Building packages

### Rust only

```bash
pixi run cargo build              # debug build into target/debug
pixi run cargo build --release    # optimized build into target/release
```

You rarely need this; the commands below build what you need.

### Python wheel and source distribution

```bash
pixi run build-wheel    # dist/optimagic_rust-0.1.0-cp312-abi3-macosx_11_0_arm64.whl
pixi run build-sdist    # dist/optimagic_rust-0.1.0.tar.gz
```

- A **wheel** contains the compiled extension and is specific to an operating system and
  CPU architecture. The tag `cp312-abi3` means it uses Python's stable ABI and works on
  every CPython >= 3.12. `pixi run build-wheel` only builds the wheel for your own
  platform; CI builds the others.
- An **sdist** (source distribution) contains the Rust and Python sources and the tests.
  Installing from it compiles the Rust code, which needs a Rust toolchain. conda-forge
  builds its packages from the sdist.

To try a wheel in a fresh environment, outside the repository:

```bash
uv venv -p 3.12 /tmp/try-omr
uv pip install -p /tmp/try-omr dist/*.whl
/tmp/try-omr/bin/python -c "import optimagic_rust; print(optimagic_rust.__version__)"
```

### Conda package

```bash
pixi run -e conda-build build-conda
```

This runs [rattler-build](https://rattler.build) with `recipe/recipe.yaml`: it builds
the package in an isolated environment with conda-forge's Rust compiler, then installs
it in a fresh environment and runs the tests. The result lands in `output/<platform>/`.
To try it:

```bash
pixi exec -c ./output -c conda-forge -s optimagic-rust -s python \
  python -c "import optimagic_rust; print(optimagic_rust.__version__)"
```

## Continuous integration

| Workflow                               | Trigger                | What it does                                                                                                                                                                |
| -------------------------------------- | ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `.github/workflows/main.yml`           | PRs, pushes to `main`  | rustfmt, clippy, Rust tests and docs; Python tests on Linux/macOS/Windows × Python 3.12–3.15; ty and stubtest; builds and tests the conda package                           |
| `.github/workflows/release.yml`        | PRs, `main`, tags `v*` | builds wheels (manylinux and musllinux x86_64/aarch64, macOS x86_64/arm64, Windows x64) and the sdist; tests the wheels on Python 3.12 and 3.15; publishes to PyPI for tags |
| [pre-commit.ci](https://pre-commit.ci) | PRs                    | runs the prek hooks (except the Rust ones) and updates their versions monthly                                                                                               |
| Dependabot (`.github/dependabot.yml`)  | weekly                 | PRs to update GitHub Actions and Rust dependencies                                                                                                                          |

As in optimagic, actions are pinned to commit hashes, and CI installs exactly the
versions in `pixi.lock` (`frozen: true`). After changing dependencies in
`pyproject.toml`, run `pixi lock` and commit `pixi.lock`.

## Releasing

### One-time setup

1. **PyPI trusted publishing.** Log in at [pypi.org](https://pypi.org), go to *Your
   account → Publishing → Add a new pending publisher* and enter: PyPI project name
   `optimagic-rust`, owner `optimagic-dev`, repository `optimagic-rust`, workflow
   `release.yml`, environment `pypi`. No API token or secret is needed.
1. **GitHub environment.** In the repository settings, under *Environments*, create an
   environment `pypi`. Optionally, add yourself as required reviewer, so each release
   waits for your approval.
1. **pre-commit.ci.** Enable the repository at [pre-commit.ci](https://pre-commit.ci)
   (if the optimagic-dev installation does not already cover all repositories).
1. **Branch protection.** Protect `main` and require the CI checks to pass.

### Each release

1. Bump `version` in the `[workspace.package]` section of `Cargo.toml` (the only place
   the version is defined; maturin passes it to the Python package) and run
   `pixi run cargo check` to update `Cargo.lock`.
1. Update `CHANGES.md`, open a PR and merge it.
1. Tag the merge commit and push the tag:
   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```
1. `release.yml` builds and tests all wheels and the sdist, checks that the tag matches
   the version and uploads everything to PyPI.

### conda-forge

conda-forge packages are built by conda-forge, from the sdist on PyPI.

- **First release:** fork
  [conda-forge/staged-recipes](https://github.com/conda-forge/staged-recipes) and add
  `recipes/optimagic-rust/recipe.yaml`, based on `recipe/recipe.yaml` with two changes:
  `source` becomes the PyPI sdist URL with its sha256, and `version` becomes a literal
  string. `recipe/variants.yaml` is not needed. After the PR is merged, conda-forge
  creates the repository `conda-forge/optimagic-rust-feedstock`, with you as maintainer.
- **Later releases:** a conda-forge bot notices new PyPI releases and opens a PR on the
  feedstock. Merge it once CI passes.

### crates.io

The Rust crates are not published (`publish = false` in their `Cargo.toml`). Revisit
this if Rust users want to use the algorithms directly.

## Using optimagic-rust from optimagic

After the first release, add `"optimagic-rust>=0.1"` to `dependencies` in optimagic's
`pyproject.toml` and `optimagic-rust = ">=0.1"` to `[tool.pixi.dependencies]` (from
conda-forge).

Before that, or to work on both projects at once, point optimagic's pixi environment to
your local checkout, in optimagic's `pyproject.toml`:

```toml
[tool.pixi.pypi-dependencies]
optimagic-rust = { path = "../optimagic-rust", editable = true }
```

`pixi install` in optimagic then compiles optimagic-rust once (this needs rustup). The
editable install imports the package directly from `../optimagic-rust/python`, so after
changing Rust code, run `pixi run develop` in optimagic-rust: it replaces the compiled
module there, and optimagic picks it up. Thanks to abi3, the same compiled module works
for all Python versions of optimagic's environments.

## Design decisions

- **Two crates.** `optimagic-rust` contains the algorithms in pure Rust. It compiles and
  tests without Python and could be published to crates.io. `optimagic-rust-py` contains
  only the conversion between Python and Rust. This is the common structure of larger
  PyO3 projects (e.g. polars, tokenizers).
- **Stable ABI (abi3).** The extension is built for Python's stable ABI, starting at
  3.12. That gives one wheel per platform instead of one per platform and Python
  version, and new Python versions work without a new release. The cost: it does not
  support free-threaded Python builds, and a few PyO3 operations are slightly slower.
- **Rust from rustup, not from conda-forge.** rustup is the standard way to install
  Rust, works with rust-analyzer and other Rust tools, and installs the version pinned
  in `rust-toolchain.toml` automatically. pixi only manages the Python side.
  (conda-forge's builds use conda-forge's Rust compiler.)
- **Hand-written type stub, checked automatically.** PyO3 cannot yet generate reliable
  `.pyi` stubs. The stub is hand-written; stubtest and `tests/test_stubs.py` keep it in
  sync with the compiled module.
- **Callback errors pass through unchanged.** The core crate is generic over the error
  type of user-provided functions, so a Python exception travels through the Rust code
  and is re-raised as is.
- **Callbacks receive copies.** User functions get a fresh numpy array on each call, so
  they cannot corrupt the algorithm's state by modifying their argument.
- **Aligned with optimagic:** MIT license, Python 3.12 and newer, pixi environments and
  task names, ruff rules, ty, prek hooks, pinned GitHub Actions, trusted publishing.

## Open decisions and future work

- **Linear algebra.** ndarray has no matrix decompositions (Cholesky, eigenvalues),
  which trust-region solvers need. Options: [faer](https://docs.rs/faer) (pure Rust,
  fast, no system dependencies), [nalgebra](https://docs.rs/nalgebra) (pure Rust), or
  ndarray-linalg (needs BLAS/LAPACK, which complicates building wheels). faer is
  probably the best fit.
- **Releasing the GIL** with `py.detach` in functions that don't call back into Python.
- **Free-threaded Python** (3.14t): requires separate, non-abi3 wheels.
- **Benchmarks** (e.g. criterion for Rust, pytest-benchmark for Python).
- **Coverage** (cargo-llvm-cov for Rust, pytest-cov for Python, uploaded to Codecov).
- **cargo-deny** to check licenses and security advisories of Rust dependencies.
- **Documentation website** once there is more than one algorithm.

## Rust primer for Python developers

| Rust                                     | Python equivalent                                                   |
| ---------------------------------------- | ------------------------------------------------------------------- |
| crate                                    | package                                                             |
| `Cargo.toml`                             | `pyproject.toml`                                                    |
| `Cargo.lock`                             | `pixi.lock`                                                         |
| workspace                                | several packages in one repository, sharing one lock file           |
| [crates.io](https://crates.io)           | PyPI                                                                |
| [docs.rs](https://docs.rs)               | Read the Docs, generated automatically for every crate on crates.io |
| edition (`2024`)                         | language version for syntax; all editions interoperate              |
| `Result<T, E>` and the `?` operator      | return value or exception; `?` re-raises the error to the caller    |
| `Option<T>`                              | `T \| None`                                                         |
| `&x` / `&mut x` (borrowing)              | passing a reference; the borrow checker prevents data races         |
| `struct` + `impl`                        | dataclass with methods                                              |
| trait                                    | protocol / abstract base class                                      |
| closure `\|x\| x + 1`                    | `lambda x: x + 1`                                                   |
| `Array1<f64>` / `ArrayView1<f64>`        | numpy array that owns its data / numpy view                         |
| `#[...]` attributes, `name!(...)` macros | decorators; code generation at compile time                         |
| `///` doc comments                       | docstrings                                                          |
| debug vs release build                   | no equivalent; debug compiles fast but runs often 10x slower        |

Reading list, roughly in order:

1. [The Rust Book](https://doc.rust-lang.org/book/), chapters 1–10: the language.
1. [ndarray for NumPy users](https://docs.rs/ndarray/latest/ndarray/doc/ndarray_for_numpy_users/index.html).
1. [The PyO3 user guide](https://pyo3.rs): exposing Rust to Python.
1. [The maturin user guide](https://www.maturin.rs): building and packaging.
1. [rust-numpy docs](https://docs.rs/numpy): passing numpy arrays.
1. [Rust by Example](https://doc.rust-lang.org/rust-by-example/) and
   [Rustlings](https://rustlings.rust-lang.org) for practice.

## Troubleshooting

- **`cannot set a minimum Python version 3.12 higher than the interpreter version`**:
  cargo found an old Python. Run the command through pixi, see
  [Run cargo through pixi](#run-cargo-through-pixi).
- **Python does not see my Rust changes**: run `pixi run develop` (or
  `pixi run tests-python`, which does it for you).
- **`ImportError: dynamic module does not define module export function`**: the module
  name in `#[pymodule(name = "_core")]` (`crates/optimagic-rust-py/src/lib.rs`) must
  match `module-name` in `[tool.maturin]` (`pyproject.toml`).
- **`linker 'cc' not found`** (Linux/macOS) or **`link.exe not found`** (Windows):
  install the C toolchain, see [step 1](#1-install-the-tools-once-per-machine).
- **`stubtest` reports an error**: the stub `_core.pyi` differs from the compiled
  module. Fix the stub (or the Rust signature) as the message describes.
- **Strange errors after switching branches**: rebuild with `pixi run develop`. As a
  last resort, `cargo clean` deletes all compiled Rust code in `target/`.
- **`target/` is large**: it holds all compiled Rust code and can grow to several GB
  over time. `cargo clean` deletes it; it is rebuilt on demand.
