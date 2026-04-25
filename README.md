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

One-command full pipeline:

```bash
python -m eyewear.cli pipeline --input data/sample_me.jpg --subject-id sample_me
```

Docker one-command full pipeline:

```bash
docker exec imenhfe-eyewear-dev python -m eyewear.cli pipeline --input data/sample_me.jpg --subject-id sample_me
```

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

## Method A Iris Scaling / Method A 홍채 기반 스케일링

English: Method A tries to use MediaPipe refined Face Mesh/Iris landmarks when
the `mediapipe` package is installed. If face/iris detection succeeds,
`metadata.json` reports `backend_status="mediapipe_face_mesh_refine_landmarks"`
and `scale_source="iris_depth"`. If MediaPipe is missing or detection fails, the
pipeline emits `backend_status="template_proxy"` and `metric_ready=false`; that
output is only a smoke-test handoff, not a real face measurement.

한국어: Method A는 `mediapipe` 패키지가 설치되어 있고 얼굴/홍채 검출이
성공하면 MediaPipe refined Face Mesh/Iris landmark를 사용합니다. 성공 시
`metadata.json`에 `backend_status="mediapipe_face_mesh_refine_landmarks"`,
`scale_source="iris_depth"`가 기록됩니다. MediaPipe가 없거나 검출 실패 시에는
`backend_status="template_proxy"`, `metric_ready=false`로 표시되며, 이 결과는
실측값이 아니라 파이프라인 확인용입니다.

## FLAME Assets For Method B / Method B용 FLAME 파일 받는 법

English:

1. Create an account at the FLAME project site: https://flame.is.tue.mpg.de/
2. Request/download the FLAME model after accepting the license.
3. Download the FLAME texture space if available for your license/project.
4. Place the files here:

```text
third_party/photometric_optimization/data/generic_model.pkl
third_party/photometric_optimization/data/FLAME_texture.npz
```

5. Add the upstream code if it is not present yet:

```bash
git submodule add https://github.com/HavenFeng/photometric_optimization third_party/photometric_optimization
git submodule update --init --recursive
```

6. Use the Method B Docker image for real fitting:

```bash
docker build -f Dockerfile.photometric -t imenhfe-eyewear:method-b .
docker run --gpus all --rm -v "$PWD:/workspace" -w /workspace imenhfe-eyewear:method-b \
  python -m eyewear.cli run photometric --input data/sample_me.jpg --subject-id sample_me --photometric-device cuda
```

한국어:

1. FLAME 사이트에서 계정을 만듭니다: https://flame.is.tue.mpg.de/
2. 라이선스에 동의한 뒤 FLAME model 파일을 다운로드합니다.
3. 프로젝트/라이선스에서 허용되는 경우 FLAME texture space도 다운로드합니다.
4. 파일을 아래 위치에 둡니다.

```text
third_party/photometric_optimization/data/generic_model.pkl
third_party/photometric_optimization/data/FLAME_texture.npz
```

5. upstream 코드가 아직 없다면 추가합니다.

```bash
git submodule add https://github.com/HavenFeng/photometric_optimization third_party/photometric_optimization
git submodule update --init --recursive
```

6. 실제 Method B fitting은 PyTorch/PyTorch3D/CUDA 환경이 필요하므로 전용 Docker를
사용하는 편이 안전합니다. TensorFlow Docker가 아니라 Linux + CUDA + PyTorch3D
환경입니다.

```bash
docker build -f Dockerfile.photometric -t imenhfe-eyewear:method-b .
docker run --gpus all --rm -v "$PWD:/workspace" -w /workspace imenhfe-eyewear:method-b \
  python -m eyewear.cli run photometric --input data/sample_me.jpg --subject-id sample_me --photometric-device cuda
```

Until these assets are present, Method B still writes the common handoff package
but marks the backend as blocked or proxy-level. Do not use proxy-level Method B
outputs as real mm-accurate geometry.

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
