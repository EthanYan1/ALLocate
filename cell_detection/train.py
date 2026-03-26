
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> int:
    print("Running:", " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def _ultralytics_cmd() -> list[str]:
    """
    Prefer `python -m ultralytics` to avoid PATH issues with `yolo`.
    """
    return [sys.executable, "-m", "ultralytics"]


def build_args(ns: argparse.Namespace) -> list[str]:
    args: list[str] = []
    # Ultralytics CLI pattern:
    #   python -m ultralytics detect train key=value key=value ...
    args += ["detect", "train"]
    args += [f"data={ns.data}"]
    args += [f"model={ns.model}"]
    args += [f"epochs={ns.epochs}"]
    args += [f"imgsz={ns.imgsz}"]
    args += [f"batch={ns.batch}"]
    args += [f"device={ns.device}"]
    if ns.workers is not None:
        args += [f"workers={ns.workers}"]
    if ns.project:
        args += [f"project={ns.project}"]
    if ns.name:
        args += [f"name={ns.name}"]
    if ns.seed is not None:
        args += [f"seed={ns.seed}"]
    if ns.patience is not None:
        args += [f"patience={ns.patience}"]
    if ns.resume:
        args += ["resume=True"]
    if ns.exist_ok:
        args += ["exist_ok=True"]
    if ns.amp is not None:
        args += [f"amp={str(ns.amp)}"]
    if ns.pretrained is not None:
        args += [f"pretrained={str(ns.pretrained)}"]

    # Any extra `key=value` args are passed through as-is.
    args += ns.extra
    return args


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Train YOLOv8 cell detector (Ultralytics CLI wrapper)."
    )
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to YOLO `data.yaml` exported from Roboflow (or equivalent).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Base model or checkpoint path (e.g. yolov8s.pt or runs/.../best.pt).",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Ultralytics device string (e.g. cpu, 0, 0,1).",
    )
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--resume", action="store_true", help="Resume last run.")
    p.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting an existing run directory.",
    )
    p.add_argument(
        "--amp",
        type=str,
        default=None,
        choices=["True", "False"],
        help="Mixed precision (leave unset to use Ultralytics default).",
    )
    p.add_argument(
        "--pretrained",
        type=str,
        default=None,
        choices=["True", "False"],
        help="Whether to use pretrained weights (leave unset for default).",
    )
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra Ultralytics `key=value` args (e.g. lr0=0.001 optimizer=SGD).",
    )
    ns = p.parse_args(argv)

    # Basic hygiene: helpful early errors for common mispaths.
    if not Path(ns.data).exists():
        raise SystemExit(f"--data not found: {ns.data}")

    cmd = _ultralytics_cmd() + build_args(ns)
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
