#!/usr/bin/env python3
"""Run VGGT per synchronized time step for anchor-based 4DGS preparation.

This version supports two input layouts:

1) Flat layout, backward compatible:
       all_images/CCTV_01_000000.png
       all_images/CCTV_02_000000.png
       all_images/Drone_Main_000000.png

2) Per-camera folder layout:
       dataset_root/cctv_01/CCTV_01_000000.png
       dataset_root/cctv_02/CCTV_02_000000.png
       dataset_root/try_drone/Drone_Main_000000.png

Camera names are parsed from the image filename by default. For example,
CCTV_01_000000.png -> camera=CCTV_01, frame=0.
The final numeric token is treated as the synchronized frame id.

For phone/drone/portrait images, use the default --normalize_images behavior.
It physically applies EXIF orientation and writes clean RGB images into each
temporary VGGT scene, so VGGT and 4DGS read the same pixel orientation.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_IGNORE_DIRS = {"depth", "depths", "mask", "masks", "seg", "segs", "semantic", "semantics"}


def parse_name(path: Path) -> Tuple[str, int]:
    """Parse camera name and frame id from a filename.

    The camera name may contain underscores. The last underscore/dash followed
    by digits is treated as the frame id separator.
    """
    stem = path.stem
    m = re.match(r"^(?P<cam>.+)[_-](?P<frame>\d+)$", stem)
    if not m:
        raise ValueError(f"Cannot parse camera/frame from {path.name}. Expected camera_000001.png")
    return m.group("cam"), int(m.group("frame"))


def parse_csv(value: str) -> List[str]:
    if value is None:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def image_size_after_exif(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img)
        return int(img.size[0]), int(img.size[1])


def _add_image(result: Dict[int, Dict[str, Path]], path: Path, camera_name_from: str = "filename") -> None:
    if path.suffix.lower() not in IMAGE_EXTS:
        return
    parsed_cam, frame = parse_name(path)
    cam = parsed_cam if camera_name_from == "filename" else path.parent.name
    if frame in result and cam in result[frame]:
        old = result[frame][cam]
        raise RuntimeError(f"Duplicate image for camera={cam}, frame={frame}: {old} and {path}")
    result.setdefault(frame, {})[cam] = path


def index_flat_images(all_images: Path, camera_name_from: str = "filename") -> Dict[int, Dict[str, Path]]:
    result: Dict[int, Dict[str, Path]] = {}
    for p in sorted(all_images.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        _add_image(result, p, camera_name_from=camera_name_from)
    if not result:
        raise RuntimeError(f"No images found in flat folder: {all_images}")
    return result


def index_camera_folder_dataset(
    dataset_root: Path,
    ignore_dirs: List[str],
    camera_name_from: str = "filename",
) -> Dict[int, Dict[str, Path]]:
    """Index images from dataset_root/<camera_folder>/*.png.

    Only immediate child folders are scanned. This matches the common Unreal
    export layout where each camera has its own folder and each folder may also
    contain a depth/ subfolder that should be ignored.
    """
    result: Dict[int, Dict[str, Path]] = {}
    ignore = {x.lower() for x in ignore_dirs}

    # Also accept a dataset root that already contains images directly.
    direct_images = [p for p in sorted(dataset_root.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    for p in direct_images:
        _add_image(result, p, camera_name_from="filename")

    for cam_dir in sorted(dataset_root.iterdir()):
        if not cam_dir.is_dir():
            continue
        if cam_dir.name.startswith(".") or cam_dir.name.lower() in ignore:
            continue
        for p in sorted(cam_dir.iterdir()):
            if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
                continue
            try:
                _add_image(result, p, camera_name_from=camera_name_from)
            except ValueError as e:
                print(f"[WARN] skip unparsable image: {p} ({e})")

    if not result:
        raise RuntimeError(f"No images found in per-camera dataset root: {dataset_root}")
    return result


def list_all_cameras(by_frame: Dict[int, Dict[str, Path]]) -> List[str]:
    cams = set()
    for frame_cams in by_frame.values():
        cams.update(frame_cams.keys())
    return sorted(cams)


def link_copy_or_normalize(src: Path, dst: Path, mode: str, normalize_images: bool) -> Tuple[int, int, bool]:
    """Place src at dst and return width, height, normalized flag.

    If normalize_images is true, EXIF orientation is applied and an RGB image is
    saved to dst. This avoids phone portrait orientation mismatches between VGGT
    and 4DGS. Otherwise, the file is copied or symlinked unchanged.
    """
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if normalize_images:
        with Image.open(src) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")
            elif img.mode == "RGBA":
                # VGGT converts to RGB internally. Store RGB here so downstream
                # tools see the same image size and orientation.
                bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
                img = Image.alpha_composite(bg, img).convert("RGB")
            else:
                img = img.convert("RGB")
            width, height = img.size
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.suffix.lower() in {".jpg", ".jpeg"}:
                img.save(dst, quality=95)
            else:
                img.save(dst)
            return int(width), int(height), True

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(f"Unknown link mode: {mode}")
    width, height = image_size_after_exif(src)
    return width, height, False


def run_one(cmd: List[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def sparse_exists(scene_dir: Path) -> bool:
    sparse = scene_dir / "sparse"
    return (sparse / "images.bin").exists() and (sparse / "cameras.bin").exists()


def main() -> None:
    parser = argparse.ArgumentParser()

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--all_images", type=Path, help="Old flat image folder containing camera_frame images.")
    input_group.add_argument("--dataset_root", type=Path, help="New per-camera folder dataset root.")

    parser.add_argument("--work_root", required=True, type=Path,
                        help="Output working folder for per-frame VGGT scenes.")
    parser.add_argument("--vggt_repo", required=True, type=Path,
                        help="Path to VGGT repository containing demo_colmap.py.")
    parser.add_argument("--fixed_cams", default="",
                        help="Comma separated fixed camera names. Empty means infer all cameras except moving cameras.")
    parser.add_argument("--moving_cam", default="",
                        help="Deprecated single moving camera name. Prefer --moving_cams.")
    parser.add_argument("--moving_cams", default="",
                        help="Comma separated moving camera names, e.g. Drone_Main,Phone_Main. If empty, use --moving_cam if provided.")
    parser.add_argument("--frame_ids", default="",
                        help="Optional comma separated frame ids. Empty means all complete frames.")
    parser.add_argument("--camera_name_from", choices=["filename", "folder"], default="filename",
                        help="Use filename prefix or parent folder as camera name. filename is safer for your shown dataset.")
    parser.add_argument("--ignore_dirs", default=",".join(sorted(DEFAULT_IGNORE_DIRS)),
                        help="Comma separated subfolder names to ignore under dataset_root.")
    parser.add_argument("--use_ba", action="store_true", help="Pass --use_ba to VGGT demo_colmap.py.")
    parser.add_argument("--camera_type", default="PINHOLE", help="COLMAP camera type for VGGT BA mode. Use PINHOLE for mixed resolution cameras.")
    parser.add_argument("--query_frame_num", type=int, default=0,
                        help="0 means number of images in each per-frame batch.")
    parser.add_argument("--max_query_pts", type=int, default=4096)
    parser.add_argument("--link_mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--normalize_images", dest="normalize_images", action="store_true", default=True,
                        help="Apply EXIF orientation and save clean RGB images into each temporary scene. Default: on.")
    parser.add_argument("--no_normalize_images", dest="normalize_images", action="store_false",
                        help="Disable image normalization and use copy/symlink instead.")
    args = parser.parse_args()

    demo = args.vggt_repo / "demo_colmap.py"
    if not demo.exists():
        raise FileNotFoundError(f"Cannot find {demo}")

    if args.dataset_root is not None:
        by_frame = index_camera_folder_dataset(
            args.dataset_root,
            ignore_dirs=parse_csv(args.ignore_dirs),
            camera_name_from=args.camera_name_from,
        )
        input_root = args.dataset_root
        input_layout = "per_camera_folders"
    else:
        by_frame = index_flat_images(args.all_images, camera_name_from=args.camera_name_from)
        input_root = args.all_images
        input_layout = "flat"

    all_cams = list_all_cameras(by_frame)
    moving_cams = parse_csv(args.moving_cams)
    legacy_moving_cam = args.moving_cam.strip()
    if legacy_moving_cam and not moving_cams:
        moving_cams = [legacy_moving_cam]
    moving_cams = list(dict.fromkeys(moving_cams))

    missing_moving = [c for c in moving_cams if c not in all_cams]
    if missing_moving:
        raise ValueError(f"moving_cams not found: {missing_moving}. Available cameras: {all_cams}")

    fixed_cams = parse_csv(args.fixed_cams)
    if not fixed_cams:
        moving_set = set(moving_cams)
        fixed_cams = [c for c in all_cams if c not in moving_set] if moving_cams else all_cams.copy()
        print(f"[INFO] inferred fixed_cams: {fixed_cams}")
    else:
        missing_fixed = [c for c in fixed_cams if c not in all_cams]
        if missing_fixed:
            raise ValueError(f"fixed_cams not found: {missing_fixed}. Available cameras: {all_cams}")

    overlap = sorted(set(fixed_cams).intersection(moving_cams))
    if overlap:
        raise ValueError(f"Camera(s) cannot be both fixed and moving: {overlap}")

    required_cams = list(dict.fromkeys(fixed_cams + moving_cams))

    if moving_cams and len(fixed_cams) < 2:
        print("[WARN] Fewer than two fixed cameras. build_vggtview_scene_v4.py requires at least two fixed anchors.")

    if args.frame_ids.strip():
        frame_ids = [int(x) for x in parse_csv(args.frame_ids)]
    else:
        frame_ids = sorted(fid for fid, cams in by_frame.items() if all(c in cams for c in required_cams))

    if not frame_ids:
        raise RuntimeError(f"No complete frames found for requested cameras: {required_cams}")

    args.work_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_layout": input_layout,
        "input_root": str(input_root.resolve()),
        "all_cameras": all_cams,
        "fixed_cams": fixed_cams,
        "moving_cams": moving_cams,
        "moving_cam": moving_cams[0] if len(moving_cams) == 1 else "",
        "required_cams": required_cams,
        "normalize_images": bool(args.normalize_images),
        "camera_name_from": args.camera_name_from,
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
        if args.skip_existing and sparse_exists(scene_dir):
            print(f"[SKIP] existing VGGT result for frame {frame_id}")
        else:
            image_dir.mkdir(parents=True, exist_ok=True)
            image_records = {}
            for cam in required_cams:
                src = cams[cam]
                dst = image_dir / src.name
                width, height, normalized = link_copy_or_normalize(src, dst, args.link_mode, args.normalize_images)
                image_records[cam] = {
                    "source_path": str(src.resolve()),
                    "scene_image_path": str(dst.resolve()),
                    "width": int(width),
                    "height": int(height),
                    "normalized": bool(normalized),
                }

            qfn = args.query_frame_num if args.query_frame_num > 0 else len(required_cams)
            cmd = [sys.executable, str(demo), "--scene_dir", str(scene_dir)]
            if args.use_ba:
                cmd += [
                    "--use_ba",
                    "--camera_type", args.camera_type,
                    "--query_frame_num", str(qfn),
                    "--max_query_pts", str(args.max_query_pts),
                ]
            run_one(cmd, cwd=args.vggt_repo)
        
        manifest["frames"].append({
            "frame_id": int(frame_id),
            "scene_dir": str(scene_dir.resolve()),
            "images": {
                cam: {
                    "source_path": str(cams[cam].resolve()),
                    "scene_image_path": str((image_dir / cams[cam].name).resolve()),
                }
                for cam in required_cams
            },
        })

    with open(args.work_root / "anchor_batches_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[OK] cameras: {required_cams}")
    print(f"[OK] frames: {len(manifest['frames'])}")
    print(f"[OK] wrote {args.work_root / 'anchor_batches_manifest.json'}")


if __name__ == "__main__":
    main()
