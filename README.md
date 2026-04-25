# IMEN343 Term Project - RGB-only Eyewear Face Modeling Prototype

Accessible custom-fit eyewear prototype for comparing two RGB-only upstream face
modeling approaches under one downstream handoff schema.

- Method A: MediaPipe Face Mesh/Face Landmarker-style landmarks + Iris scaling
- Method B: HavenFeng/photometric_optimization wrapper + post-hoc calibration plan

The goal is not a polished product or manufacturing pipeline. The goal is a
semester-realistic technical prototype that can produce a structured handoff
package for downstream parametric eyewear design and compare accuracy/cost/time
tradeoffs honestly.

## Repository Design

```text
docs/
third_party/photometric_optimization/
src/eyewear/
  common/
    canonicalization/
    evaluation/
    geometry/
    io/
    measurements/
    schema/
    visualization/
  methods/
    mediapipe/
    photometric/
  cli/
tests/
outputs/
```

Method-specific code stops at raw inference/wrapping. Canonical coordinates,
semantic landmarks, measurements, handoff serialization, previews, and comparison
live in `src/eyewear/common/`.

## Requirement Documents

Found at repository root:

- `eyewear_face_modeling_handoff_spec_v1.docx`
- `0403 교수님 면담.docx`
- `3D Face Modeling.docx`
- `0322.docx`
- `Proposal.pdf`

No requested requirement document is missing. See `docs/requirements_trace.md`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional MediaPipe runtime:

```bash
python -m pip install -e ".[mediapipe]"
```

Optional photometric dependency stack:

```bash
python -m pip install -r requirements-photometric.txt
```

The photometric backend is PyTorch/PyTorch3D-based, not TensorFlow. It also
needs upstream code, FLAME assets, and compatible PyTorch/CUDA setup before it
should be treated as subject-specific dense fitting.

## Docker / Dev Container

Build the base image:

```bash
docker build -t imenhfe-eyewear .
docker run --rm imenhfe-eyewear
```

Build with optional photometric Python dependencies:

```bash
docker build --build-arg INSTALL_PHOTOMETRIC=true -t imenhfe-eyewear:photometric .
```

For real Method B fitting, prefer the separate Linux/CUDA-oriented Dockerfile:

```bash
docker build -f Dockerfile.photometric -t imenhfe-eyewear:method-b .
docker run --gpus all --rm -v "$PWD:/workspace" imenhfe-eyewear:method-b \
  python -m eyewear.cli run photometric --input data/front.jpg --subject-id s01 --photometric-device cuda
```

VS Code users can reopen the repository in the included `.devcontainer/`.

## CLI

```bash
python -m eyewear.cli run mediapipe --input data/front.jpg --subject-id s01 --input-mode single_image
python -m eyewear.cli run mediapipe --input data/s01_photos --subject-id s01 --input-mode photo_set
python -m eyewear.cli run mediapipe --input data/face.mp4 --subject-id s01 --input-mode video
python -m eyewear.cli run photometric --input data/front.jpg --subject-id s01 --input-mode single_image
python -m eyewear.cli run photometric --input data/front.jpg --subject-id s01 --photometric-device cuda --photometric-timeout-sec 3600
python -m eyewear.cli run photometric --input data/front.jpg --subject-id s01 --skip-photometric-upstream
python -m eyewear.cli compare --subject-id s01
python -m eyewear.cli evaluate --subject-id s01
```

Method B currently supports `single_image` and `photo_set`; video fitting is not
claimed. If `third_party/photometric_optimization` has the upstream code and
manual FLAME assets, the wrapper stages the image/landmarks/mask, runs
`photometric_fitting.py`, and imports the resulting OBJ/NPY files. Otherwise it
emits clearly labeled proxy outputs so the shared handoff/comparison pipeline can
still be tested.

## Outputs

Common output folder:

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

## Schema Examples

`eyewear_landmarks.json`:

```json
{
  "unit": "mm",
  "coordinate_system": {
    "origin": "midpoint_between_iris_centers",
    "x": "subject_right",
    "y": "up",
    "z": "forward",
    "handedness": "right-handed"
  },
  "points": {
    "left_iris_center": {
      "xyz": [31.0, 0.0, 0.0],
      "source": "observed",
      "method": "mediapipe_iris",
      "confidence": 0.9
    }
  }
}
```

`measurements.json`:

```json
{
  "unit": "mm",
  "pd_mm": {"value": 62.0, "status": "observed", "confidence": 0.95, "basis": null},
  "bridge_width_mm_at_eye_line": {"value": 14.0, "status": "derived", "confidence": 0.7, "basis": null},
  "temple_length_estimated_mm": {"value": 140.0, "status": "estimated", "confidence": 0.4, "basis": "template-rule"},
  "face_roll_deg": 0.0,
  "face_pitch_deg": 0.0,
  "face_yaw_deg": 0.0
}
```

`metadata.json` includes runtime, dependency burden, backend status, missing
fields, estimated fields, metric scale source, and whether the run is
`metric_ready`.

## Comparison Workflow

1. Run both methods for the same `subject_id`.
2. Confirm both output the same canonical schema.
3. Run `python -m eyewear.cli compare --subject-id s01`.
4. Review measurement deltas, estimated fields, runtime, setup burden, GPU need,
   metric readiness, and backend status.

Accuracy requires scanner/manual ground truth. Without ground truth, the
comparison report should be treated as schema completeness and engineering
tradeoff evidence, not validated anatomical accuracy.

## Known Limitations

- RGB-only input cannot precisely recover back-of-ear geometry.
- Temple/back-of-ear outputs are visible-ear proxies or estimated values.
- Single portrait input is weak for side/head-breadth measurements.
- MediaPipe output is metric only when iris scaling succeeds; otherwise the CLI
  labels the result as a template proxy.
- Method B can invoke HavenFeng upstream when code/assets are present, but its
  current staged landmarks/masks are coarse proxies and dense mesh metric scale
  is still marked unverified.
- Do not use proxy outputs for fabrication or mm-accuracy claims.

## Smoke Test

```bash
python -m pytest -q
```
