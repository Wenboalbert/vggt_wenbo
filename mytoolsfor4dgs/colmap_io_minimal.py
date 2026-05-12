#!/usr/bin/env python3
"""Minimal COLMAP binary/text IO for VGGT-to-4DGS utilities.

This file is intentionally standalone so the conversion tools do not need to
import either VGGT or 4DGS internals.
"""

from __future__ import annotations

import collections
import os
import struct
from typing import Dict, List, Tuple

import numpy as np

CameraModel = collections.namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

CAMERA_MODELS = {
    CameraModel(0, "SIMPLE_PINHOLE", 3),
    CameraModel(1, "PINHOLE", 4),
    CameraModel(2, "SIMPLE_RADIAL", 4),
    CameraModel(3, "RADIAL", 5),
    CameraModel(4, "OPENCV", 8),
    CameraModel(5, "OPENCV_FISHEYE", 8),
    CameraModel(6, "FULL_OPENCV", 12),
    CameraModel(7, "FOV", 5),
    CameraModel(8, "SIMPLE_RADIAL_FISHEYE", 4),
    CameraModel(9, "RADIAL_FISHEYE", 5),
    CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}
CAMERA_MODEL_IDS = {m.model_id: m for m in CAMERA_MODELS}
CAMERA_MODEL_NAMES = {m.model_name: m for m in CAMERA_MODELS}


def read_next_bytes(fid, num_bytes: int, fmt: str, endian: str = "<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def qvec2rotmat(qvec):
    qvec = np.asarray(qvec, dtype=np.float64)
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2],
    ], dtype=np.float64)


def rotmat2qvec(R):
    R = np.asarray(R, dtype=np.float64)
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ], dtype=np.float64) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    qvec /= max(np.linalg.norm(qvec), 1e-12)
    return qvec


def read_cameras_binary(path: str) -> Dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = int(camera_properties[0])
            model_id = int(camera_properties[1])
            width = int(camera_properties[2])
            height = int(camera_properties[3])
            model = CAMERA_MODEL_IDS[model_id].model_name
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[camera_id] = Camera(camera_id, model, width, height, np.array(params, dtype=np.float64))
    return cameras


def read_images_binary(path: str) -> Dict[int, Image]:
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            image_id = int(props[0])
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            camera_id = int(props[8])
            name = ""
            ch = read_next_bytes(fid, 1, "c")[0]
            while ch != b"\x00":
                name += ch.decode("utf-8")
                ch = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            xys = np.zeros((0, 2), dtype=np.float64)
            point3D_ids = np.zeros((0,), dtype=np.int64)
            if num_points2D > 0:
                elems = read_next_bytes(fid, 24 * num_points2D, "ddq" * num_points2D)
                xys = np.column_stack([elems[0::3], elems[1::3]]).astype(np.float64)
                point3D_ids = np.array(elems[2::3], dtype=np.int64)
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, point3D_ids)
    return images


def read_points3D_binary(path: str) -> Dict[int, Point3D]:
    points = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = read_next_bytes(fid, 43, "QdddBBBd")
            point_id = int(props[0])
            xyz = np.array(props[1:4], dtype=np.float64)
            rgb = np.array(props[4:7], dtype=np.uint8)
            error = float(props[7])
            track_len = read_next_bytes(fid, 8, "Q")[0]
            image_ids = []
            point2D_idxs = []
            if track_len > 0:
                track = read_next_bytes(fid, 8 * track_len, "ii" * track_len)
                image_ids = list(map(int, track[0::2]))
                point2D_idxs = list(map(int, track[1::2]))
            points[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2D_idxs)
    return points


def read_model_sparse(path: str):
    cam_bin = os.path.join(path, "cameras.bin")
    img_bin = os.path.join(path, "images.bin")
    pts_bin = os.path.join(path, "points3D.bin")
    if not (os.path.exists(cam_bin) and os.path.exists(img_bin)):
        raise FileNotFoundError(f"Missing cameras.bin/images.bin in {path}")
    cameras = read_cameras_binary(cam_bin)
    images = read_images_binary(img_bin)
    points = read_points3D_binary(pts_bin) if os.path.exists(pts_bin) else {}
    return cameras, images, points


def camera_center_from_image(image: Image) -> np.ndarray:
    Rcw = qvec2rotmat(image.qvec)
    return -Rcw.T @ image.tvec


def write_cameras_text(cameras: Dict[int, Camera], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam_id in sorted(cameras):
            cam = cameras[cam_id]
            params = " ".join(str(float(x)) for x in cam.params)
            f.write(f"{cam.id} {cam.model} {cam.width} {cam.height} {params}\n")


def write_images_text(images: Dict[int, Image], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, IMAGE_NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        for img_id in sorted(images):
            im = images[img_id]
            q = " ".join(str(float(x)) for x in im.qvec)
            t = " ".join(str(float(x)) for x in im.tvec)
            f.write(f"{im.id} {q} {t} {im.camera_id} {im.name}\n")
            if len(im.xys) == 0:
                f.write("\n")
            else:
                pts = []
                for xy, pid in zip(im.xys, im.point3D_ids):
                    pts.extend([str(float(xy[0])), str(float(xy[1])), str(int(pid))])
                f.write(" ".join(pts) + "\n")


def write_points3D_text(points: Dict[int, Point3D], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")
        for pid in sorted(points):
            p = points[pid]
            track = []
            for iid, idx in zip(p.image_ids, p.point2D_idxs):
                track.extend([str(int(iid)), str(int(idx))])
            line = [str(int(p.id))] + [str(float(x)) for x in p.xyz] + [str(int(x)) for x in p.rgb] + [str(float(p.error))] + track
            f.write(" ".join(line) + "\n")


def copy_points_to_text(src_sparse: str, dst_sparse: str) -> None:
    os.makedirs(dst_sparse, exist_ok=True)
    src_bin = os.path.join(src_sparse, "points3D.bin")
    src_txt = os.path.join(src_sparse, "points3D.txt")
    dst_txt = os.path.join(dst_sparse, "points3D.txt")
    if os.path.exists(src_bin):
        pts = read_points3D_binary(src_bin)
        write_points3D_text(pts, dst_txt)
    elif os.path.exists(src_txt):
        import shutil
        shutil.copy2(src_txt, dst_txt)
    else:
        raise FileNotFoundError(f"No points3D.bin or points3D.txt in {src_sparse}")
