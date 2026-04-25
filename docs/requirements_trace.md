# Requirements Trace

This file records the repository-document requirements used to shape the current
prototype. Priority order follows the handoff instruction:

1. `eyewear_face_modeling_handoff_spec_v1.docx`
2. `0403 교수님 면담.docx`
3. `3D Face Modeling.docx`
4. `0322.docx`
5. `Proposal.pdf`

## Found Documents

All five requested requirement documents were found at the repository root.

## Hard Requirements Reflected In Code

- One repository contains both Method A and Method B.
- Method-specific raw inference lives under `src/eyewear/methods/`.
- Canonical schema, coordinate handling, measurements, writers, previews, and
  comparison logic live under `src/eyewear/common/`.
- Handoff output uses mm, origin at midpoint between iris centers, +X subject
  right, +Y up, +Z forward, right-handed.
- Ear and temple-back geometry are marked as estimated/proxy, not precise
  RGB-only recovery.
- Both methods emit the same common files under
  `outputs/{subject_id}/{method_name}/`.
- Comparison is first-class through `python -m eyewear.cli compare`.

## Conflict Resolution

- Early proposal material mentions manufacturing/wearing-study validation, but
  the professor meeting notes de-emphasize full validation for this semester.
  The repository therefore prioritizes technical prototype, comparison, and
  honest limitations.
- Printed markers are mentioned as a possible calibration method, but later
  modeling notes prefer MediaPipe Iris for accessibility. The default code path
  therefore records iris-based scale strategy and treats marker calibration as a
  future alternative.
- Back-of-ear geometry appears in eyewear design needs, but the handoff spec and
  modeling notes both state RGB cannot recover it reliably. The code outputs
  visible-ear proxy fields as estimated.

## Current Implementation Boundary

Method A can optionally use MediaPipe Face Mesh refined landmarks when the
`mediapipe` extra is installed; otherwise it emits a clearly labeled template
proxy for smoke testing.

Method B currently provides the shared wrapper, output contract, and proxy
artifacts while the HavenFeng upstream code and FLAME assets remain uninstalled.
Its metadata keeps `metric_ready=false` until a real dense fitting and calibration
path is wired.
