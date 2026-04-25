# RGB-only Eyewear Face Modeling Prototype

Accessible custom-fit eyewear prototype for comparing two RGB-only upstream face
modeling approaches under one downstream handoff schema.

- Method A: MediaPipe Face Mesh/Face Landmarker-style landmarks + Iris scaling
- Method B: HavenFeng/photometric_optimization wrapper + post-hoc calibration plan

The goal is a semester-realistic technical prototype, not a polished product,
mobile app, or manufacturing pipeline.

## Repository Design

```text
docs/
third_party/photometric_optimization/
src/eyewear/
  common/
  methods/
  cli/
tests/
outputs/
```

Method-specific code lives under `src/eyewear/methods/`. Shared canonical
coordinates, schema, measurements, handoff serialization, previews, and
comparison live under `src/eyewear/common/`.

## Requirement Documents

The requested requirement documents are present at the repository root:

- `eyewear_face_modeling_handoff_spec_v1.docx`
- `0403 교수님 면담.docx`
- `3D Face Modeling.docx`
- `0322.docx`
- `Proposal.pdf`

See `docs/requirements_trace.md` for the implementation trace.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,mediapipe]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev,mediapipe]"
```

Optional photometric dependency stack:

```bash
python -m pip install -r requirements-photometric.txt
```

Method B is PyTorch/PyTorch3D-based, not TensorFlow. Real dense fitting also
requires upstream code, FLAME assets, and a compatible CUDA/PyTorch3D setup.

`third_party/photometric_optimization` is connected as a Git submodule pointing
to HavenFeng's upstream repository. The upstream code is MIT licensed, so keeping
it as a submodule pointer is acceptable for this school project. FLAME model and
texture assets are separately licensed and must not be committed.

## Docker

Build the base image:

```bash
docker build -t imenhfe-eyewear .
```

Run an always-on development container:

```bash
docker run -d --name imenhfe-eyewear-dev -v "$PWD:/workspace" -w /workspace imenhfe-eyewear sleep infinity
docker exec -it imenhfe-eyewear-dev bash
```

Windows PowerShell:

```powershell
docker run -d --name imenhfe-eyewear-dev -v "${PWD}:/workspace" -w /workspace imenhfe-eyewear sleep infinity
docker exec -it imenhfe-eyewear-dev bash
```

For real Method B fitting, prefer the Linux/CUDA-oriented Dockerfile:
This image uses the PyTorch 2.1 / CUDA 11.8 / PyTorch3D 0.7.5 wheel and pins
`numpy<2`, `opencv-python<4.13`, and `chumpy` for upstream compatibility.

```bash
docker build -f Dockerfile.photometric -t imenhfe-eyewear:method-b .
docker run --gpus all --rm -v "$PWD:/workspace" -w /workspace imenhfe-eyewear:method-b \
  python -m eyewear.cli run photometric --input data/sample_me.jpg --subject-id sample_me --photometric-device cuda
```

If Docker cannot access a GPU, CPU execution also works. It is slow, but it runs
the real upstream fitting path.

```bash
docker run --rm -v "$PWD:/workspace" -w /workspace imenhfe-eyewear:method-b \
  python -m eyewear.cli run photometric --input data/sample_me.jpg --subject-id sample_me --photometric-device cpu --photometric-timeout-sec 1800
```

## One-command Pipeline

Run Method A, Method B, and comparison together:

```bash
python -m eyewear.cli pipeline --input data/sample_me.jpg --subject-id sample_me
```

Docker:

```bash
docker exec imenhfe-eyewear-dev python -m eyewear.cli pipeline --input data/sample_me.jpg --subject-id sample_me
```

Individual commands:

```bash
python -m eyewear.cli run mediapipe --input data/front.jpg --subject-id s01 --input-mode single_image
python -m eyewear.cli run photometric --input data/front.jpg --subject-id s01 --input-mode single_image
python -m eyewear.cli compare --subject-id s01
```

## Method A Iris Scaling

Method A uses MediaPipe refined Face Mesh/Iris landmarks when `mediapipe` is
installed and face/iris detection succeeds. Check `metadata.json`:

```json
{
  "backend_status": "mediapipe_face_mesh_refine_landmarks",
  "scale_source": "iris_depth",
  "metric_ready": true
}
```

If MediaPipe is missing or detection fails, the pipeline emits
`backend_status="template_proxy"` and `metric_ready=false`. That output is only
for smoke testing, not real measurement.

Method B runs post-hoc MediaPipe Iris calibration after successful upstream
photometric fitting. In `calibration.json`, `status="validated"`,
`scale_source="iris_posthoc_calibration"`, and `metric_ready=true` mean the
eye/nose anchor residual passed the configured threshold. The current semantic
mapping is still a mesh bounding-box proxy, so temple, forehead, and
ear/back-of-ear values must remain proxy/estimated interpretations.

## FLAME Assets For Method B

1. Create an account at https://flame.is.tue.mpg.de/.
2. Accept the license and download the items below.

| Download item | Required? | Purpose | Local path |
| --- | --- | --- | --- |
| `FLAME 2023 Open` | Required | Face geometry model. Use the `.pkl` file inside the archive. | `third_party/photometric_optimization/data/generic_model.pkl` |
| `FLAME texture space (for non-commercial use only)` | Required for the original photometric fitter | Texture space used by HavenFeng's `FLAMETex`. The archive usually contains an `.npz` file. | `third_party/photometric_optimization/data/FLAME_texture.npz` |
| `FLAME Mediapipe Landmark Embedding` | Optional | Useful for MediaPipe-to-FLAME landmark/vertex mapping experiments. Not required by the current wrapper. | `third_party/photometric_optimization/data/` if needed |
| `FLAME Vertex Masks` | Optional | Useful for future semantic region mapping. | `third_party/photometric_optimization/data/` if needed |
| `FLAME Blender Add-on`, `FLAME 2020/2019/2017` | Not needed | Not used by the current wrapper. | You can skip these. |

3. Rename/place the files like this. Renaming is fine if the archive uses a
different filename.

```text
third_party/photometric_optimization/data/generic_model.pkl
third_party/photometric_optimization/data/FLAME_texture.npz
```

4. Fetch the submodule code:

```bash
git submodule update --init --recursive
```

Do not commit FLAME model/texture files. They are licensed assets. `.gitignore`
excludes `third_party/photometric_optimization/data/*.pkl` and
`third_party/photometric_optimization/data/*.npz`.

Until these assets are present, Method B still writes the common handoff package
but marks the backend as blocked or proxy-level. Do not use proxy-level Method B
outputs as real mm-accurate geometry.

## Outputs

```text
outputs/{subject_id}/{method_name}/
  metadata.json
  eyewear_landmarks.json
  measurements.json
  preview_front.png
  preview_side.png
  face_transform.json
  design_curves.json
```

Method A extra:

```text
raw_landmarks.csv
```

Method B extra:

```text
raw_mesh.obj
flame_params.npz
calibration.json
```

Comparison output:

```text
outputs/{subject_id}/comparison/
  comparison_summary.json
  comparison_report.md
```

## Known Limitations

- RGB-only input cannot precisely recover back-of-ear geometry.
- Temple/back-of-ear outputs are visible-ear proxies or estimated values.
- Single portrait input is weak for side/head-breadth measurements.
- Method B dense fitting requires upstream code and licensed FLAME assets.
- Method B `metric_ready=true` means the handoff unit is iris-calibrated mm; it
  does not mean every FLAME vertex-to-anatomical landmark mapping is validated.
- Do not use proxy or residual-too-high outputs for fabrication or mm-accuracy claims.

## Smoke Test

```bash
python -m pytest -q
```
