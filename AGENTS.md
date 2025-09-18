# Repository Guidelines

## Project Structure & Module Organization
The Python package lives in `chroma/`, which houses detector geometry, photon propagation, GPU kernels, and helper modules. CUDA source used by PyCUDA resides in `chroma/cuda/`, and command-line entry points are in `bin/` for running simulations, cameras, and BVH utilities. Tests are collected in `test/`, pairing Python harnesses with `.cu` fixtures under `test/data/`. Supporting material lives in `doc/` (Sphinx sources and whitepaper) and `installation/` (Dockerfiles for CUDA-capable environments); `images/` stores reference renders used by docs and demos.

## Build, Test, and Development Commands
- `python -m pip install -e .` installs Chroma in editable mode with scripts from `bin/` on your PATH.
- `python -m pytest test` runs the full unit suite; add `-k <pattern>` to focus on specific detectors or kernels.
- `PYCUDA_CACHE_DIR=.pycuda-cache python -m pytest test/test_bvh.py` demonstrates how to pin the CUDA cache locally when debugging GPU-specific failures.

## Coding Style & Naming Conventions
Follow PEP 8 with four-space indentation, `snake_case` functions, and CapWords classes to match the existing modules. Keep imports explicit and localize heavy CUDA/PyCUDA helpers near their call sites so kernels stay discoverable. For `.cu` files, mirror the surrounding style: lower_snake_case functions, uppercase constants, and const-correct pointers.

## Testing Guidelines
New features need a Python unit test in `test/` named `test_<feature>.py`; reuse the companion `.cu` harness when exercising kernels. PyTest discovers both unittest-style classes and free functions, so prefer clear assertions from `numpy.testing` for numeric checks. Ensure GPU-dependent tests guard on `pycuda.driver.Context.get_current()` or mark-skip so CPU-only contributors can run the suite.

## Commit & Pull Request Guidelines
Recent history favors concise, imperative commit subjects (for example, "remove unused computations for debugging"). Keep bodies brief but note physics-impacting changes or required driver versions. Pull requests should summarize detector or kernel impacts, link related issues or notebooks, and confirm `python -m pytest test` passes on a CUDA-enabled host. Include screenshots or log snippets when visual or performance regressions are possible.

## GPU & Environment Notes
Use the Docker recipes in `installation/` or Singularity descriptors when you need a controlled CUDA stack; match container driver versions with the host as described in `README.md`. Local development expects the NVIDIA toolchain and PyCUDA headers; run `nvidia-smi` before long simulations to confirm device availability. When rendering via `chroma-cam`, ensure X11 forwarding or `DISPLAY` is configured, and commit only deterministic assets to `images/`.
