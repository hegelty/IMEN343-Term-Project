# RGB-only 안경 얼굴 모델링 프로토타입

이 저장소는 접근성 높은 맞춤형 안경 제작을 위해 RGB-only 얼굴 모델링
방법 2가지를 하나의 공통 handoff schema로 비교하는 학기 프로젝트용
프로토타입입니다.

- Method A: MediaPipe Face Mesh/Face Landmarker 계열 landmark + Iris scaling
- Method B: HavenFeng/photometric_optimization wrapper + 사후 metric calibration 계획

목표는 완성형 제품, 모바일 앱, 제조 자동화가 아니라 학기 안에 가능한 기술
프로토타입과 공정한 비교입니다.

## 저장소 구조

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

방법별 코드는 `src/eyewear/methods/`에 있고, canonical 좌표계, schema,
측정치, 저장, preview, 비교 로직은 `src/eyewear/common/`에 모았습니다.

## 요구사항 문서

요청된 문서는 모두 저장소 루트에 있습니다.

- `eyewear_face_modeling_handoff_spec_v1.docx`
- `0403 교수님 면담.docx`
- `3D Face Modeling.docx`
- `0322.docx`
- `Proposal.pdf`

구현 추적은 `docs/requirements_trace.md`를 참고하세요.

## 설치

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

선택 사항인 photometric 의존성:

```bash
python -m pip install -r requirements-photometric.txt
```

Method B는 TensorFlow가 아니라 PyTorch/PyTorch3D 기반입니다. 실제 dense
fitting에는 upstream 코드, FLAME asset, 호환되는 CUDA/PyTorch3D 환경이
필요합니다.

## Docker

기본 이미지 빌드:

```bash
docker build -t imenhfe-eyewear .
```

개발용 컨테이너를 계속 띄워두기:

```bash
docker run -d --name imenhfe-eyewear-dev -v "$PWD:/workspace" -w /workspace imenhfe-eyewear sleep infinity
docker exec -it imenhfe-eyewear-dev bash
```

Windows PowerShell:

```powershell
docker run -d --name imenhfe-eyewear-dev -v "${PWD}:/workspace" -w /workspace imenhfe-eyewear sleep infinity
docker exec -it imenhfe-eyewear-dev bash
```

실제 Method B fitting은 Linux/CUDA용 Dockerfile을 쓰는 것이 안전합니다.

```bash
docker build -f Dockerfile.photometric -t imenhfe-eyewear:method-b .
docker run --gpus all --rm -v "$PWD:/workspace" -w /workspace imenhfe-eyewear:method-b \
  python -m eyewear.cli run photometric --input data/sample_me.jpg --subject-id sample_me --photometric-device cuda
```

## 한 번에 실행

Method A, Method B, comparison을 한 번에 실행:

```bash
python -m eyewear.cli pipeline --input data/sample_me.jpg --subject-id sample_me
```

Docker:

```bash
docker exec imenhfe-eyewear-dev python -m eyewear.cli pipeline --input data/sample_me.jpg --subject-id sample_me
```

개별 실행:

```bash
python -m eyewear.cli run mediapipe --input data/front.jpg --subject-id s01 --input-mode single_image
python -m eyewear.cli run photometric --input data/front.jpg --subject-id s01 --input-mode single_image
python -m eyewear.cli compare --subject-id s01
```

## Method A 홍채 기반 스케일링

`mediapipe`가 설치되어 있고 얼굴/홍채 검출이 성공하면 Method A는 MediaPipe
refined Face Mesh/Iris landmark를 사용합니다. `metadata.json`에서 아래처럼
확인할 수 있습니다.

```json
{
  "backend_status": "mediapipe_face_mesh_refine_landmarks",
  "scale_source": "iris_depth",
  "metric_ready": true
}
```

MediaPipe가 없거나 검출에 실패하면 `backend_status="template_proxy"`,
`metric_ready=false`로 표시됩니다. 이 경우 결과는 pipeline smoke test용이지
실측값이 아닙니다.

## Method B용 FLAME 파일 받는 법

1. https://flame.is.tue.mpg.de/ 에서 계정을 만듭니다.
2. 라이선스에 동의한 뒤 FLAME model을 요청/다운로드합니다.
3. 프로젝트/라이선스에서 허용되면 FLAME texture space도 다운로드합니다.
4. 파일을 아래 위치에 둡니다.

```text
third_party/photometric_optimization/data/generic_model.pkl
third_party/photometric_optimization/data/FLAME_texture.npz
```

5. upstream 코드가 없다면 추가합니다.

```bash
git submodule add https://github.com/HavenFeng/photometric_optimization third_party/photometric_optimization
git submodule update --init --recursive
```

이 asset들이 없으면 Method B는 공통 handoff package는 만들지만 backend를
blocked/proxy 상태로 표시합니다. proxy-level Method B 결과를 실제 mm 정확도나
제작용 geometry로 사용하면 안 됩니다.

## 출력

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

Method A 추가 출력:

```text
raw_landmarks.csv
```

Method B 추가 출력:

```text
raw_mesh.obj
flame_params.npz
calibration.json
```

비교 출력:

```text
outputs/{subject_id}/comparison/
  comparison_summary.json
  comparison_report.md
```

## 알려진 한계

- RGB-only 입력으로 귀 뒤쪽 geometry를 정밀 복원할 수 없습니다.
- temple/back-of-ear 값은 visible-ear proxy 또는 estimated 값입니다.
- 단일 정면 사진은 측면/머리폭 측정에 약합니다.
- Method B dense fitting은 upstream 코드와 라이선스가 필요한 FLAME asset을
  설치해야 실제로 동작합니다.
- proxy 출력은 제작이나 mm 정확도 주장에 사용하면 안 됩니다.

## Smoke Test

```bash
python -m pytest -q
```
