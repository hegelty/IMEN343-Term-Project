# HavenFeng/photometric_optimization placeholder

This directory is intentionally separated from local wrapper code.

Current status: the upstream repository is **not vendored yet** in this checkout.
The local wrapper in `src/eyewear/methods/photometric/runner.py` therefore emits
clearly labeled proxy artifacts so the shared handoff, comparison, and CLI layers
can be tested before the fragile photometric stack is installed.

Planned upstream:

- Repository: https://github.com/HavenFeng/photometric_optimization
- Integration style: submodule or vendored copy under this directory
- Local patches: none yet
- Required manual assets: FLAME model files and any pretrained assets required by
  the upstream project
- Known risk: PyTorch/CUDA/version coupling can be brittle, so keep the upstream
  code isolated from the common schema and CLI layers.
