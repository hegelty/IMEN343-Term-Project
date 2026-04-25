# photometric_optimization integration note

- Upstream target repo: `https://github.com/HavenFeng/photometric_optimization`
- Integration style: separated third-party directory + thin wrapper in `src/eyewear/methods/photometric/runner.py`
- Current status: the local wrapper can stage inputs and call upstream when the
  code and manual assets are present; this checkout still contains only the
  placeholder directory unless you add the upstream project
- Local patches applied: none
- Upstream modified files: none in this repo
- Manual assets likely required:
  - FLAME model files
  - pretrained checkpoints
- Fragile dependencies:
  - CUDA / PyTorch / PyTorch3D legacy package version coupling
  - Upstream README reports Python 3.8.3, PyTorch 1.5, CUDA 10.2, PyTorch3D 0.2
  - This is a PyTorch/PyTorch3D stack, not TensorFlow
- Remaining blockers:
  - Clone or add the upstream project under `third_party/photometric_optimization/`
  - Install compatible PyTorch/CUDA/OpenCV dependencies
  - Download FLAME and upstream-required assets under paths allowed by the upstream license
  - Add `data/generic_model.pkl` and `data/FLAME_texture.npz`
  - Replace coarse staged landmarks/masks with real FAN landmarks and face segmentation masks
  - Map fitted mesh vertices/FLAME parameters into the existing canonical schema with a validated vertex/region map
  - Run iris-based or other defensible post-hoc metric calibration before marking Method B `metric_ready=true`

Suggested submodule command:

```bash
git submodule add https://github.com/HavenFeng/photometric_optimization third_party/photometric_optimization
git submodule update --init --recursive
```

Current wrapper behavior:

```bash
python -m eyewear.cli run photometric --input data/front.jpg --subject-id s01 --photometric-device cpu
```

If upstream is ready, the wrapper stages:

- `third_party/photometric_optimization/FFHQ/{subject_id}.png`
- `third_party/photometric_optimization/FFHQ/{subject_id}.npy`
- `third_party/photometric_optimization/FFHQ_seg/{subject_id}.npy`

Then it runs:

```bash
python photometric_fitting.py {subject_id} cpu
```

and imports:

- `test_results/{subject_id}.obj` -> `outputs/{subject_id}/photometric/raw_mesh.obj`
- `test_results/{subject_id}.npy` -> `outputs/{subject_id}/photometric/flame_params.npz`

Until real landmark/segmentation preprocessing and metric calibration are added,
the common semantic landmarks remain proxy mapped and `metric_ready=false`.

Do not edit upstream files directly unless a small patch is unavoidable. If patches
are added later, document the upstream commit and each modified file here.
