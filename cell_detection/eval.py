"""
Evaluate the ALLocate YOLOv8 cell detector (Ultralytics CLI wrapper).

"""

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
    return [sys.executable, "-m", "ultralytics"]


def build_args(ns: argparse.Namespace) -> list[str]:
    args: list[str] = []
    args += ["detect", "val"]
    args += [f"data={ns.data}"]
    args += [f"model={ns.model}"]
    args += [f"imgsz={ns.imgsz}"]
    args += [f"batch={ns.batch}"]
    args += [f"device={ns.device}"]
    if ns.conf is not None:
        args += [f"conf={ns.conf}"]
    if ns.iou is not None:
        args += [f"iou={ns.iou}"]
    if ns.project:
        args += [f"project={ns.project}"]
    if ns.name:
        args += [f"name={ns.name}"]
    if ns.save_json:
        args += ["save_json=True"]
    if ns.save_txt:
        args += ["save_txt=True"]
    args += ns.extra
    return args


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Validate YOLOv8 cell detector (Ultralytics CLI wrapper)."
    )
    p.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to YOLO `data.yaml` (should define val/valid split).",
    )
    p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Checkpoint to validate (e.g. runs/detect/train/weights/best.pt).",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=None, help="IoU threshold for NMS.")
    p.add_argument("--project", type=str, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--save-json", action="store_true")
    p.add_argument("--save-txt", action="store_true")
    p.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra Ultralytics `key=value` args (e.g. plots=True half=False).",
    )
    ns = p.parse_args(argv)

    if not Path(ns.data).exists():
        raise SystemExit(f"--data not found: {ns.data}")
    if not Path(ns.model).exists():
        raise SystemExit(f"--model not found: {ns.model}")

    cmd = _ultralytics_cmd() + build_args(ns)
    return _run(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
