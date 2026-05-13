# VGGTView to 4DGS v4 - multi-moving-camera version

This toolkit prepares 4DGS training data for a multi-camera dynamic scene.
It supports:

- A flat image folder.
- A dataset root with one folder per camera.
- An unknown number of fixed cameras.
- One or more moving cameras.
- Mixed landscape / portrait images, when image normalization is enabled.

The core idea is unchanged: VGGT is run for every synchronized time step, then all per-frame VGGT coordinate systems are aligned into one global coordinate system using the fixed cameras as anchors. The final output is a 4DGS `VGGTView` scene.

## Input layouts

### Layout A: flat folder

```text
all_images/
  CCTV_01_000000.png
  CCTV_02_000000.png
  Drone_Main_000000.png
  Phone_Main_000000.png
  CCTV_01_000001.png
  CCTV_02_000001.png
  Drone_Main_000001.png
  Phone_Main_000001.png
```

### Layout B: one folder per camera

```text
city_1_datasets/
  cctv_01/
    CCTV_01_000000.png
    CCTV_01_000001.png
    depth/
      ... ignored ...
  cctv_02/
    CCTV_02_000000.png
    CCTV_02_000001.png
  try_drone/
    Drone_Main_000000.png
    Drone_Main_000001.png
  phone/
    Phone_Main_000000.png
    Phone_Main_000001.png
```

By default the camera name is parsed from the image filename, not from the folder name. For example:

- `CCTV_01_000000.png` -> camera name `CCTV_01`, frame id `0`.
- `Drone_Main_000000.png` -> camera name `Drone_Main`, frame id `0`.

This is important when the folder name is `try_drone` but the image prefix is `Drone_Main`.

Subfolders named `depth`, `depths`, `mask`, `masks`, `seg`, `semantic`, etc. are ignored.

## Step 1: run VGGT per synchronized frame

For one moving camera:

```bash
python mytoolsfor4dgs/run_vggt_anchor_batches_v4.py \
  --dataset_root "/home/wenbo/Documents/Unreal Projects/city_1_datasets" \
  --work_root /data/vggt_anchor_work \
  --vggt_repo /path/to/vggt_wenbo \
  --moving_cams Drone_Main \
  --use_ba \
  --camera_type PINHOLE \
  --query_frame_num 0 \
  --skip_existing
```

For multiple moving cameras:

```bash
python mytoolsfor4dgs/run_vggt_anchor_batches_v4.py \
  --dataset_root "/home/wenbo/Documents/Unreal Projects/city_1_datasets" \
  --work_root /data/vggt_anchor_work \
  --vggt_repo /path/to/vggt_wenbo \
  --moving_cams Drone_Main,Phone_Main \
  --use_ba \
  --camera_type PINHOLE \
  --query_frame_num 0 \
  --skip_existing
```

If `--fixed_cams` is not given, the script infers:

```text
fixed_cams = all detected cameras - moving_cams
```

For example, if all detected cameras are:

```text
CCTV_01, CCTV_02, CCTV_03, Drone_Main, Phone_Main
```

and you pass:

```text
--moving_cams Drone_Main,Phone_Main
```

then the fixed anchors are inferred as:

```text
CCTV_01, CCTV_02, CCTV_03
```

You can still manually specify fixed cameras:

```bash
--fixed_cams CCTV_01,CCTV_02,CCTV_03 --moving_cams Drone_Main,Phone_Main
```

`--normalize_images` is on by default. It applies EXIF orientation and writes clean RGB images into every temporary VGGT scene. Keep this on for phone / drone / portrait images.

## Step 2: build the final VGGTView scene

For one moving camera:

```bash
python mytoolsfor4dgs/build_vggtview_scene_v4.py \
  --work_root /data/vggt_anchor_work \
  --dataset_root "/home/wenbo/Documents/Unreal Projects/city_1_datasets" \
  --out_scene /data/final_scene \
  --moving_cams Drone_Main \
  --intrinsic_mode per_image \
  --with_scale
```

For multiple moving cameras:

```bash
python mytoolsfor4dgs/build_vggtview_scene_v4.py \
  --work_root /data/vggt_anchor_work \
  --dataset_root "/home/wenbo/Documents/Unreal Projects/city_1_datasets" \
  --out_scene /data/final_scene \
  --moving_cams Drone_Main,Phone_Main \
  --intrinsic_mode per_image \
  --with_scale
```

Again, if `--fixed_cams` is not provided, fixed cameras are inferred as:

```text
all cameras - moving_cams
```

The fixed cameras are used only as anchors for alignment. Their poses are forced to the reference-frame global poses. All moving cameras are transformed by the same alignment for each frame, so they keep their frame-varying trajectories.

Outputs:

```text
final_scene/
  images/
  sparse/0/points3D.txt
  sparse/0/cameras.txt
  sparse/0/images.txt
  vggtview_meta.json
```

`vggtview_meta.json` stores every image's global pose, camera intrinsics, camera name, frame id, normalized time, `is_fixed`, and `is_moving`.

## Step 3: train 4DGS

Make sure your 4DGaussians repository has the VGGTView reader patch. Then:

```bash
cd /path/to/4DGaussians_wenbo
python train.py \
  -s /data/final_scene \
  --configs arguments/hypernerf/default.py \
  --model_path /data/output_model
```

4DGS reads `final_scene/vggtview_meta.json`, so it does not need to know the original `dataset_root` structure.

## Important options

### `--moving_cams`

Comma-separated moving camera names. Use image filename prefixes, not folder names. Example:

```text
--moving_cams Drone_Main,Phone_Main
```

The old `--moving_cam` option is still accepted for one moving camera, but `--moving_cams` is recommended.

### `--fixed_cams`

Comma-separated fixed anchor camera names. Optional. If omitted, fixed cameras are inferred automatically.

### `--intrinsic_mode per_image`

Recommended for mixed camera types, phone/drone images, portrait/landscape mixes, or any case where image size/intrinsics may vary.

### `--intrinsic_mode per_prefix`

Only use this if each camera prefix always has the same resolution and stable intrinsics.

### `--camera_type PINHOLE`

Recommended for VGGT BA mode because it preserves `fx`, `fy`, `cx`, and `cy` separately.

## Files changed for multi-moving-camera support

- `run_vggt_anchor_batches_v4.py`: accepts `--moving_cams`, infers fixed anchors from all detected cameras minus all moving cameras, and includes all moving cameras in every per-frame VGGT batch.
- `build_vggtview_scene_v4.py`: accepts `--moving_cams`, aligns using only fixed anchors, writes all moving cameras into `vggtview_meta.json`, and marks each image with `is_moving`.
- `README_vggtview_4dgs_v4.md`: updated usage examples.

`colmap_io_minimal.py` does not need changes.

## Current limitations

At least two fixed cameras are still required for stable anchor alignment. Multiple moving cameras are supported, but they must all appear in the synchronized frames you want to process.
