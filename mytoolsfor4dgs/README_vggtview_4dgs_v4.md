# VGGTView to 4DGS v4

This toolkit prepares 4DGS training data for a multi-camera dynamic scene where most cameras are fixed and one camera moves.

It uses VGGT per synchronized frame group, aligns all per-frame VGGT coordinate systems using the fixed cameras, then writes a VGGTView scene for 4DGS.

## Inputs

A flat image folder:

- cctv1_000000.png
- cctv2_000000.png
- cctv3_000000.png
- phone1_000000.png
- cctv1_000001.png
- cctv2_000001.png
- cctv3_000001.png
- phone1_000001.png

The final numeric token is the frame id. The prefix before it is the camera name.

## Step 1: patch VGGT image ordering

python patch_vggt_demo_colmap_sort.py --demo_colmap /path/to/vggt_wenbo/demo_colmap.py

This only changes the image glob line to a sorted glob, making each per-frame VGGT batch deterministic.

## Step 2: run VGGT per synchronized frame

python run_vggt_anchor_batches_v4.py \
  --all_images /data/all_images \
  --work_root /data/vggt_anchor_work \
  --vggt_repo /path/to/vggt_wenbo \
  --fixed_cams cctv1,cctv2,cctv3,cctv4 \
  --moving_cam phone1 \
  --use_ba \
  --camera_type PINHOLE \
  --query_frame_num 0 \
  --skip_existing

For every frame id, the script creates a temporary VGGT scene containing all fixed cameras plus the moving camera, then runs VGGT demo_colmap.py.

## Step 3: build the final VGGTView scene

python build_vggtview_scene_v4.py \
  --work_root /data/vggt_anchor_work \
  --all_images /data/all_images \
  --out_scene /data/final_scene \
  --fixed_cams cctv1,cctv2,cctv3,cctv4 \
  --moving_cam phone1 \
  --intrinsic_mode per_prefix \
  --with_scale

Outputs:

- /data/final_scene/images
- /data/final_scene/sparse/0/points3D.txt
- /data/final_scene/sparse/0/cameras.txt
- /data/final_scene/sparse/0/images.txt
- /data/final_scene/vggtview_meta.json

The vggtview_meta.json file stores each image's global pose, intrinsics, camera name, frame id, and normalized time.

## Step 4: patch 4DGS to add VGGTView reader

python patch_4dgs_add_vggtview_reader_v4.py --four_dgs_repo /path/to/4DGaussians_wenbo

This modifies only:

- scene/dataset_readers.py
- scene/__init__.py

Backups are created with suffix .bak_vggtview_v4.

## Step 5: train 4DGS

cd /path/to/4DGaussians_wenbo
python train.py -s /data/final_scene --configs arguments/hypernerf/default.py --model_path /data/output_model

Because final_scene contains vggtview_meta.json, 4DGS will use the VGGTView reader.

## Conservative COLMAP route

build_vggtview_scene_v4.py also writes cameras.txt and images.txt in sparse/0. This is mainly for inspection and conservative debugging. Standard COLMAP has no dynamic time field, so for real dynamic training VGGTView is safer.

## Important options

--fixed_cams
Comma separated list of fixed cameras. Use at least two. More fixed cameras improve alignment stability.

--moving_cam
The one moving camera. v4 supports one moving camera.

--intrinsic_mode per_prefix
All frames from the same camera prefix share one camera_id. Good for fixed focal length cameras.

--intrinsic_mode per_image
Every image gets its own camera_id. Safer if phone/uav intrinsics may change.

--with_scale
Use Sim3 alignment. This is recommended because independent VGGT runs may have slightly different scale.

--no_scale
Use rotation and translation only.

## What the code does not solve

It does not segment dynamic objects. If moving objects dominate the fixed-camera views, VGGT pose estimation may be unstable. Masking or choosing cleaner keyframes can improve results.

It supports one moving camera by default. Multiple moving cameras require extending --moving_cam to a moving camera list.
