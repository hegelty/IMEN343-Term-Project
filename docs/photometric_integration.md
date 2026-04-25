# photometric_optimization integration note

- Upstream target repo: `https://github.com/HavenFeng/photometric_optimization`
- Integration style: separated third-party directory + thin wrapper in `src/eyewear/methods/photometric/runner.py`
- Current status: upstream is documented but not vendored/submodule-linked in this checkout
- Local patches applied: none
- Upstream modified files: none in this repo
- Manual assets likely required:
  - FLAME model files
  - pretrained checkpoints
- Fragile dependencies:
  - CUDA / PyTorch / legacy package version coupling
- Remaining blockers:
  - Clone or add the upstream project under `third_party/photometric_optimization/`
  - Install compatible PyTorch/CUDA/OpenCV dependencies
  - Download FLAME and upstream-required assets under paths allowed by the upstream license
  - Replace the current proxy mesh writer with a call into the upstream fitting entry point
  - Map fitted mesh vertices/FLAME parameters into the existing canonical schema
  - Run iris-based or other defensible post-hoc metric calibration before marking Method B `metric_ready=true`

Suggested submodule command:

```bash
git submodule add https://github.com/HavenFeng/photometric_optimization third_party/photometric_optimization
git submodule update --init --recursive
```

Do not edit upstream files directly unless a small patch is unavoidable. If patches
are added later, document the upstream commit and each modified file here.
