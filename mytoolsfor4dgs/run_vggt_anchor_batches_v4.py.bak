#!/usr/bin/env python3
"""Run VGGT per synchronized time step for anchor-based 4DGS preparation.

Input image naming convention:
    camera_name_000123.png
    camera_name_000123.jpg

The camera name can contain underscores. The final numeric token is treated as
frame id.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_name(path: Path) -> Tuple[str, int]:
    stem = path.stem
    m = re.match(r"^(?P<cam>.+)[_-](?P<frame>\d+)$", stem)
    if not m:
        raise ValueError(
            f"Cannot parse camera/frame from {path.name}. Expected camera_000001.png"
        )
    return m.group("cam"), int(m.group("frame"))


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def index_images(all_images: Path) -> Dict[int, Dict[str, Path]]:
    result: Dict[int, Dict[str, Path]] = {}
    for p in sorted(all_images.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        cam, frame = parse_name(p)
        result.setdefault(frame, {})[cam] = p
    if not result:
        raise RuntimeError(f"No images found in {all_images}")
    return result


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def run_one(cmd: List[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all_images", required=True, type=Path,
                        help="Flat image folder containing camera_frame images.")
    parser.add_argument("--work_root", required=True, type=Path,
                        help="Output working folder for per-frame VGGT scenes.")
    parser.add_argument("--vggt_repo", required=True, type=Path,
                        help="Path to VGGT repository containing demo_colmap.py.")
    parser.add_argument("--fixed_cams", required=True,
                        help="Comma separated fixed camera names, e.g. cctv1,cctv2,cctv3.")
    parser.add_argument("--moving_cam", required=True,
                        help="The one moving camera name, e.g. phone1 or uav1.")
    parser.add_argument("--frame_ids", default="",
                        help="Optional comma separated frame ids. Empty means all complete frames.")
    parser.add_argument("--use_ba", action="store_true", help="Pass --use_ba to VGGT demo_colmap.py.")
    parser.add_argument("--camera_type", default="PINHOLE", help="COLMAP camera type for VGGT BA mode.")
    parser.add_argument("--query_frame_num", type=int, default=0,
                        help="0 means number of images in each per-frame batch.")
    parser.add_argument("--max_query_pts", type=int, default=4096)
    parser.add_argument("--link_mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    fixed_cams = parse_csv(args.fixed_cams)
    moving_cam = args.moving_cam.strip()
    required_cams = fixed_cams + [moving_cam]
    if len(fixed_cams) < 2:
        raise ValueError("At least two fixed cameras are recommended for anchor alignment.")

    demo = args.vggt_repo / "demo_colmap.py"
    if not demo.exists():
        raise FileNotFoundError(f"Cannot find {demo}")

    by_frame = index_images(args.all_images)
    if args.frame_ids.strip():
        frame_ids = [int(x) for x in parse_csv(args.frame_ids)]
    else:
        frame_ids = sorted(fid for fid, cams in by_frame.items() if all(c in cams for c in required_cams))

    if not frame_ids:
        raise RuntimeError("No complete frames found for requested cameras.")

    args.work_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "all_images": str(args.all_images.resolve()),
        "fixed_cams": fixed_cams,
        "moving_cam": moving_cam,
        "required_cams": required_cams,
        "frames": [],
    }

    for frame_id in frame_ids:
        cams = by_frame.get(frame_id, {})
        missing = [c for c in required_cams if c not in cams]
        if missing:
            print(f"[SKIP] frame {frame_id}: missing {missing}")
            continue

        scene_dir = args.work_root / f"frame_{frame_id:06d}"
        image_dir = scene_dir / "images"
        sparse_dir = scene_dir / "sparse"
        if args.skip_existing and (sparse_dir / "images.bin").exists() and (sparse_dir / "cameras.bin").exists():
            print(f"[SKIP] existing VGGT result for frame {frame_id}")
        else:
            image_dir.mkdir(parents=True, exist_ok=True)
            for cam in required_cams:
                src = cams[cam]
                dst = image_dir / src.name
                link_or_copy(src, dst, args.link_mode)

            qfn = args.query_frame_num if args.query_frame_num > 0 else len(required_cams)
            cmd = [sys.executable, str(demo), "--scene_dir", str(scene_dir)]
            if args.use_ba:
                cmd += ["--use_ba", "--camera_type", args.camera_type,
                        "--query_frame_num", str(qfn), "--max_query_pts", str(args.max_query_pts)]
            run_one(cmd, cwd=args.vggt_repo)

        manifest["frames"].append({
            "frame_id": frame_id,
            "scene_dir": str(scene_dir.resolve()),
            "images": {cam: str(cams[cam].resolve()) for cam in required_cams},
        })

    with open(args.work_root / "anchor_batches_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] wrote {args.work_root / 'anchor_batches_manifest.json'}")


if __name__ == "__main__":
    main()
