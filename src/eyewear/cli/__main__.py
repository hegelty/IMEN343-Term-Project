from __future__ import annotations

import argparse
import json
from pathlib import Path

from eyewear.common.evaluation.compare import compare_subject
from eyewear.methods.mediapipe.runner import run_mediapipe
from eyewear.methods.photometric.runner import run_photometric


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eyewear")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run")
    run.add_argument("method", choices=["mediapipe", "photometric"])
    run.add_argument("--input", required=True)
    run.add_argument("--subject-id", required=True)
    run.add_argument("--output-root", default="outputs")
    run.add_argument("--input-mode", default="single_image", choices=["single_image", "photo_set", "video"])
    run.add_argument("--photometric-device", default="cpu", choices=["cpu", "cuda"], help="Device passed to HavenFeng photometric_optimization when Method B is runnable.")
    run.add_argument("--photometric-timeout-sec", type=int, default=1800, help="Timeout for Method B upstream fitting.")
    run.add_argument("--skip-photometric-upstream", action="store_true", help="Generate Method B handoff/proxy outputs without invoking the upstream fitter.")

    pipe = sub.add_parser("pipeline", help="Run Method A, Method B, and comparison in one command.")
    pipe.add_argument("--input", required=True)
    pipe.add_argument("--subject-id", required=True)
    pipe.add_argument("--output-root", default="outputs")
    pipe.add_argument("--input-mode", default="single_image", choices=["single_image", "photo_set", "video"])
    pipe.add_argument("--photometric-device", default="cpu", choices=["cpu", "cuda"], help="Device passed to HavenFeng photometric_optimization when Method B is runnable.")
    pipe.add_argument("--photometric-timeout-sec", type=int, default=1800, help="Timeout for Method B upstream fitting.")
    pipe.add_argument("--skip-photometric-upstream", action="store_true", help="Generate Method B handoff/proxy outputs without invoking the upstream fitter.")

    comp = sub.add_parser("compare")
    comp.add_argument("--subject-id", required=True)
    comp.add_argument("--output-root", default="outputs")

    evalp = sub.add_parser("evaluate")
    evalp.add_argument("--subject-id", required=True)
    evalp.add_argument("--output-root", default="outputs")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "run":
        out_root = Path(args.output_root)
        if args.method == "mediapipe":
            result = run_mediapipe(args.subject_id, args.input, out_root, input_mode=args.input_mode)
        else:
            result = run_photometric(
                args.subject_id,
                args.input,
                out_root,
                input_mode=args.input_mode,
                device=args.photometric_device,
                timeout_sec=args.photometric_timeout_sec,
                run_upstream=not args.skip_photometric_upstream,
            )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.cmd == "pipeline":
        out_root = Path(args.output_root)
        mediapipe_result = run_mediapipe(args.subject_id, args.input, out_root, input_mode=args.input_mode)
        photometric_result = run_photometric(
            args.subject_id,
            args.input,
            out_root,
            input_mode="single_image" if args.input_mode == "video" else args.input_mode,
            device=args.photometric_device,
            timeout_sec=args.photometric_timeout_sec,
            run_upstream=not args.skip_photometric_upstream,
        )
        comparison_result = compare_subject(out_root, args.subject_id)
        result = {
            "subject_id": args.subject_id,
            "output_root": str(out_root),
            "mediapipe": mediapipe_result,
            "photometric": photometric_result,
            "comparison": comparison_result,
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.cmd in {"compare", "evaluate"}:
        result = compare_subject(Path(args.output_root), args.subject_id)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return


if __name__ == "__main__":
    main()
