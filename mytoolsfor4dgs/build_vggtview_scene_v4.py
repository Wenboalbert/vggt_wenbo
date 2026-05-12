#!/usr/bin/env python3
"""Build a 4DGS VGGTView scene from per-frame VGGT COLMAP outputs.

This script aligns each per-frame VGGT local reconstruction to a reference
frame using all fixed cameras as anchors. It then writes:
    final_scene/images/
    final_scene/sparse/0/points3D.txt
    final_scene/vggtview_meta.json

Optionally, it also writes COLMAP-style cameras.txt/images.txt for debugging.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image as PILImage

from colmap_io_minimal import (
    Camera,
    Image,
    camera_center_from_image,
    copy_points_to_text,
    qvec2rotmat,
    read_model_sparse,
    rotmat2qvec,
    write_cameras_text,
    write_images_text,
)


def parse_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_cam_frame(name: str) -> Tuple[str, int]:
    stem = Path(name).stem
    m = re.match(r"^(?P<cam>.+)[_-](?P<frame>\d+)$", stem)
    if not m:
        raise ValueError(f"Cannot parse camera/frame from {name}; expected camera_000001.png")
    return m.group("cam"), int(m.group("frame"))


def find_image_by_camera(images: Dict[int, Image], cam_name: str) -> Image:
    matches = []
    for im in images.values():
        try:
            c, _ = parse_cam_frame(im.name)
        except ValueError:
            continue
        if c == cam_name:
            matches.append(im)
    if len(matches) != 1:
        names = [m.name for m in matches]
        raise RuntimeError(f"Expected exactly one image for camera {cam_name}, got {len(matches)}: {names}")
    return matches[0]


def rotation_angle_deg(R: np.ndarray) -> float:
    x = (np.trace(R) - 1.0) / 2.0
    x = float(np.clip(x, -1.0, 1.0))
    return math.degrees(math.acos(x))


def project_to_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def estimate_alignment_from_anchors(
    local_anchor_images: Dict[str, Image],
    global_anchor_images: Dict[str, Image],
    with_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray, Dict[str, float]]:
    """Estimate X_global = scale * R_align * X_local + t_align.

    Rotation is estimated mainly from anchor camera orientations, making the
    two-anchor case well constrained. Scale is estimated from all pairwise
    anchor-center distances. Translation is estimated from anchor centers.
    """
    cams = sorted(local_anchor_images.keys())
    if len(cams) < 2:
        raise ValueError("At least two fixed anchors are required.")

    # Rotation candidates from camera orientation relation:
    # Rcw_global = Rcw_local @ R_align.T
    # R_align = Rcw_global.T @ Rcw_local
    rotations = []
    for cam in cams:
        Rcw_l = qvec2rotmat(local_anchor_images[cam].qvec)
        Rcw_g = qvec2rotmat(global_anchor_images[cam].qvec)
        rotations.append(Rcw_g.T @ Rcw_l)
    R_align = project_to_rotation(np.mean(rotations, axis=0))

    Cl = np.stack([camera_center_from_image(local_anchor_images[c]) for c in cams], axis=0)
    Cg = np.stack([camera_center_from_image(global_anchor_images[c]) for c in cams], axis=0)

    if with_scale:
        ratios = []
        for i in range(len(cams)):
            for j in range(i + 1, len(cams)):
                dl = np.linalg.norm(Cl[i] - Cl[j])
                dg = np.linalg.norm(Cg[i] - Cg[j])
                if dl > 1e-8:
                    ratios.append(dg / dl)
        scale = float(np.median(ratios)) if ratios else 1.0
    else:
        scale = 1.0

    t_align = Cg.mean(axis=0) - scale * (R_align @ Cl.mean(axis=0))

    pred = (scale * (R_align @ Cl.T)).T + t_align
    center_rms = float(np.sqrt(np.mean(np.sum((pred - Cg) ** 2, axis=1))))

    rot_errs = []
    for cam in cams:
        Rcw_l = qvec2rotmat(local_anchor_images[cam].qvec)
        Rcw_g = qvec2rotmat(global_anchor_images[cam].qvec)
        Rcw_pred = Rcw_l @ R_align.T
        rot_errs.append(rotation_angle_deg(Rcw_pred @ Rcw_g.T))

    stats = {
        "center_rms": center_rms,
        "rotation_mean_deg": float(np.mean(rot_errs)),
        "rotation_max_deg": float(np.max(rot_errs)),
        "scale": float(scale),
    }
    return scale, R_align, t_align, stats


def transform_image_pose_to_global(local_image: Image, scale: float, R_align: np.ndarray, t_align: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Rcw_l = qvec2rotmat(local_image.qvec)
    C_l = camera_center_from_image(local_image)
    C_g = scale * (R_align @ C_l) + t_align
    Rcw_g = Rcw_l @ R_align.T
    tvec_g = -Rcw_g @ C_g
    qvec_g = rotmat2qvec(Rcw_g)
    return qvec_g, tvec_g


def camera_params_to_pinhole(cam: Camera) -> Tuple[str, List[float]]:
    model = cam.model
    p = [float(x) for x in cam.params]
    if model == "PINHOLE":
        return model, p[:4]
    if model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
        f = p[0]
        cx = p[1] if len(p) > 1 else cam.width / 2.0
        cy = p[2] if len(p) > 2 else cam.height / 2.0
        return "PINHOLE", [f, f, cx, cy]
    if model == "OPENCV":
        return "PINHOLE", p[:4]
    raise ValueError(f"Unsupported camera model {model}; use VGGT --camera_type PINHOLE if possible.")


def choose_camera_record(
    src_cam: Camera,
    camera_name: str,
    image_name: str,
    mode: str,
    camera_registry: Dict[str, int],
    cameras_out: Dict[int, Camera],
) -> int:
    model, params = camera_params_to_pinhole(src_cam)
    if mode == "per_image":
        key = image_name
    elif mode == "per_prefix":
        key = camera_name
    else:
        raise ValueError(f"Unknown intrinsic mode {mode}")

    if key in camera_registry:
        return camera_registry[key]

    cam_id = len(camera_registry) + 1
    camera_registry[key] = cam_id
    cameras_out[cam_id] = Camera(cam_id, model, int(src_cam.width), int(src_cam.height), np.array(params, dtype=np.float64))
    return cam_id


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        raise ValueError(mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_root", required=True, type=Path)
    parser.add_argument("--all_images", required=True, type=Path)
    parser.add_argument("--out_scene", required=True, type=Path)
    parser.add_argument("--fixed_cams", required=True, help="Comma separated fixed camera names.")
    parser.add_argument("--moving_cam", required=True, help="The one moving camera name.")
    parser.add_argument("--reference_frame", type=int, default=None,
                        help="Default: first available complete frame.")
    parser.add_argument("--intrinsic_mode", choices=["per_prefix", "per_image"], default="per_prefix")
    parser.add_argument("--with_scale", action="store_true", default=True)
    parser.add_argument("--no_scale", action="store_true", help="Disable Sim3 scale, use SE3 only.")
    parser.add_argument("--link_mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument("--write_colmap_text", action="store_true", default=True,
                        help="Also write cameras.txt/images.txt for debugging and conservative COLMAP route.")
    parser.add_argument("--warn_center_rms", type=float, default=0.05)
    parser.add_argument("--warn_rotation_deg", type=float, default=5.0)
    args = parser.parse_args()

    fixed_cams = parse_csv(args.fixed_cams)
    moving_cam = args.moving_cam.strip()
    required = fixed_cams + [moving_cam]
    if len(fixed_cams) < 2:
        raise ValueError("Need at least two fixed cameras.")

    frame_dirs = []
    for p in sorted(args.work_root.glob("frame_*")):
        if not p.is_dir():
            continue
        m = re.search(r"(\d+)$", p.name)
        if m and (p / "sparse" / "images.bin").exists():
            frame_dirs.append((int(m.group(1)), p))
    if not frame_dirs:
        raise RuntimeError(f"No frame_* sparse outputs found in {args.work_root}")

    frame_ids = [fid for fid, _ in frame_dirs]
    ref_frame = args.reference_frame if args.reference_frame is not None else frame_ids[0]
    ref_dir = dict(frame_dirs).get(ref_frame)
    if ref_dir is None:
        raise RuntimeError(f"Reference frame {ref_frame} not found in work_root")

    ref_cameras, ref_images, ref_points = read_model_sparse(str(ref_dir / "sparse"))
    global_anchor_images = {cam: find_image_by_camera(ref_images, cam) for cam in fixed_cams}

    # Use reference-frame fixed cameras as fixed global poses. The moving camera at
    # reference frame also gets its aligned global pose from the reference model.
    images_meta = []
    cameras_out: Dict[int, Camera] = {}
    images_out: Dict[int, Image] = {}
    camera_registry: Dict[str, int] = {}

    out_images = args.out_scene / "images"
    out_sparse = args.out_scene / "sparse" / "0"
    out_images.mkdir(parents=True, exist_ok=True)
    out_sparse.mkdir(parents=True, exist_ok=True)

    min_frame = min(frame_ids)
    max_frame = max(frame_ids)
    denom = max(max_frame - min_frame, 1)

    image_id_next = 1
    alignment_report = []

    for frame_id, frame_dir in frame_dirs:
        cameras, images, _points = read_model_sparse(str(frame_dir / "sparse"))
        local_fixed = {cam: find_image_by_camera(images, cam) for cam in fixed_cams}
        scale, R_align, t_align, stats = estimate_alignment_from_anchors(
            local_fixed, global_anchor_images, with_scale=(args.with_scale and not args.no_scale)
        )
        stats["frame_id"] = int(frame_id)
        alignment_report.append(stats)
        if stats["center_rms"] > args.warn_center_rms or stats["rotation_max_deg"] > args.warn_rotation_deg:
            print(f"[WARN] frame {frame_id}: center_rms={stats['center_rms']:.6f}, "
                  f"rot_max={stats['rotation_max_deg']:.3f} deg, scale={stats['scale']:.6f}")

        for cam_name in required:
            im_local = find_image_by_camera(images, cam_name)
            src_cam = cameras[im_local.camera_id]
            qvec_g, tvec_g = transform_image_pose_to_global(im_local, scale, R_align, t_align)

            # For fixed cameras, enforce exactly the reference-frame global pose.
            if cam_name in fixed_cams:
                im_ref = global_anchor_images[cam_name]
                qvec_g = np.asarray(im_ref.qvec, dtype=np.float64)
                tvec_g = np.asarray(im_ref.tvec, dtype=np.float64)

            image_src = args.all_images / im_local.name
            if not image_src.exists():
                # fallback to per-frame VGGT images folder
                image_src = frame_dir / "images" / im_local.name
            if not image_src.exists():
                raise FileNotFoundError(f"Cannot find source image {im_local.name}")
            image_dst = out_images / im_local.name
            link_or_copy(image_src, image_dst, args.link_mode)

            # Get actual image size if available; keep VGGT camera otherwise.
            width, height = int(src_cam.width), int(src_cam.height)
            try:
                with PILImage.open(image_src) as pil:
                    width, height = pil.size
            except Exception:
                pass

            # If source camera dimensions differ from actual, keep params as produced
            # by VGGT demo_colmap, which already rescales to original resolution.
            cam_id = choose_camera_record(src_cam, cam_name, im_local.name, args.intrinsic_mode, camera_registry, cameras_out)
            cameras_out[cam_id] = cameras_out[cam_id]._replace(width=width, height=height)

            time = float((frame_id - min_frame) / denom)
            entry = {
                "image_name": im_local.name,
                "image_path": f"images/{im_local.name}",
                "camera_name": cam_name,
                "frame_id": int(frame_id),
                "time": time,
                "camera_id": int(cam_id),
                "camera_model": cameras_out[cam_id].model,
                "width": int(cameras_out[cam_id].width),
                "height": int(cameras_out[cam_id].height),
                "params": [float(x) for x in cameras_out[cam_id].params],
                "qvec": [float(x) for x in qvec_g],
                "tvec": [float(x) for x in tvec_g],
                "is_fixed": cam_name in fixed_cams,
                "is_moving": cam_name == moving_cam,
            }
            images_meta.append(entry)

            images_out[image_id_next] = Image(
                image_id_next,
                np.asarray(qvec_g, dtype=np.float64),
                np.asarray(tvec_g, dtype=np.float64),
                cam_id,
                im_local.name,
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0,), dtype=np.int64),
            )
            image_id_next += 1

    # Initial point cloud: use the reference frame sparse point cloud.
    copy_points_to_text(str(ref_dir / "sparse"), str(out_sparse))

    if args.write_colmap_text:
        write_cameras_text(cameras_out, str(out_sparse / "cameras.txt"))
        write_images_text(images_out, str(out_sparse / "images.txt"))

    meta = {
        "format": "VGGTView",
        "version": 4,
        "description": "Anchor-aligned VGGT camera poses for 4DGS.",
        "fixed_cams": fixed_cams,
        "moving_cam": moving_cam,
        "reference_frame": int(ref_frame),
        "frame_min": int(min_frame),
        "frame_max": int(max_frame),
        "intrinsic_mode": args.intrinsic_mode,
        "images": sorted(images_meta, key=lambda x: (x["frame_id"], x["camera_name"])),
        "alignment_report": alignment_report,
        "point_cloud": "sparse/0/points3D.txt",
    }
    with open(args.out_scene / "vggtview_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] wrote scene: {args.out_scene}")
    print(f"[OK] images: {out_images}")
    print(f"[OK] meta: {args.out_scene / 'vggtview_meta.json'}")
    print(f"[OK] sparse: {out_sparse}")


if __name__ == "__main__":
    main()
