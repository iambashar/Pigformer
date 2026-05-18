# Preprocessing pipeline

End-to-end path from raw Azure Kinect ROS2 bags to the two HDF5 artifacts
consumed by `dataset.py`:

```
.db3 bag files
      │  stage 1: rosbag_to_h5.py          (extract synced color + depth + camera intrinsics)
      ▼
per-bag HDF5  (color, depth, timestamps, camera_info)
      │  stage 2: maskdino/infer_pig_depth_h5.py   (pig + upper-body instance segmentation on depth)
      ▼
per-bag HDF5  (+ pig / upper_body masks, endpoints, scores)
      │  stage 3: build_height_dataset.py  (RANSAC ground, BEV height map, min-area-rect + upper-body heading)
      ▼
dataset.h5   (height_maps, bag_names, source_names)

slaughter_lab.csv
      │  stage 4: parse_labels.py
      ▼
label.h5     (bag_names, unique_ids, fat_rib12, loin_rib12, total_rib)
```

These are the exact scripts used for the paper. They were developed against
the MSU/UNL data directory layouts and hard-code a few paths inside `main()` —
when running on your own data, pass the CLI flags explicitly. Expect to
spend a little time wiring paths; this is research code.

---

## Stage 1 — ROS bag → per-bag HDF5

`rosbag_to_h5.py` parses each `.db3` ROS2 bag, streams color and depth frames
to an HDF5, and keeps nanosecond timestamps so the two streams can be synced
by nearest match. Handles raw `Image`, `CompressedImage`, and `CameraInfo`
topics.

```bash
python preprocessing/rosbag_to_h5.py \
    --input_dir  /path/to/rosbags/ \
    --output_dir /path/to/hdf5_out/ \
    --workers 8
```

Output per bag:

```
color/images         (N, H, W, 3) uint8   LZF-compressed
color/timestamps_ns  (N,)         int64
depth/images         (M, H, W)    uint16  LZF-compressed
depth/timestamps_ns  (M,)         int64
color/camera_info    K, D, R, P
depth/camera_info    K, D, R, P
```

## Stage 2 — MaskDINO depth segmentation (pig + upper-body)

The paper's segmentation stage is a MaskDINO-R50 instance-segmentation model
trained on depth frames (3-channel encoding: normalized depth, valid mask,
gradient magnitude). It outputs two classes per frame — `pig` and
`upper_body` — plus ordered head/tail endpoints. The pipeline at inference
time runs MaskDINO only; it needs no RGB or prompt-based segmenter.

Prerequisites:

```bash
git clone https://github.com/IDEA-Research/MaskDINO external/MaskDINO
cd external/MaskDINO && pip install -r requirements.txt && pip install -e .
```

Configs and inference script in `preprocessing/maskdino/`:

- `maskdino_R50_depth_instance_endpoint_pig_upper_body.yaml` — the
  2-instance-class config (pig + upper-body) with the endpoint regression
  head turned on. Inherits from the two `_BASE_` configs also shipped here.
- `maskdino_R50_depth_instance_endpoint.yaml` — endpoint-head base.
- `maskdino_R50_depth_instance.yaml` — depth-input MaskDINO base.
- `infer_pig_depth_h5.py` — batched inference that reads a per-bag HDF5
  and appends `pig_masks`, `upper_body_masks`, `pig_scores`, and
  `pig_endpoints_xy` back into the same file.

```bash
python preprocessing/maskdino/infer_pig_depth_h5.py \
    --config-file preprocessing/maskdino/maskdino_R50_depth_instance_endpoint_pig_upper_body.yaml \
    --weights    /path/to/maskdino_pig_upper_body.pth \
    --input-h5   /path/to/bag.h5 \
    --output-h5  /path/to/bag.h5 \
    --batch-size 8 --score-threshold 0.2
```

## Stage 3 — height-map extraction (`build_height_dataset.py`)

`build_height_dataset.py` is the paper's reference implementation. Per pig
bag it:

1. Loads the per-bag HDF5 from stages 1 + 2.
2. Syncs each depth frame to its nearest color frame by timestamp.
3. Undistorts depth with `DepthIntrinsic` / `DepthDistortion`.
4. Projects to a 3D point cloud; fits or loads a RANSAC ground plane
   (`msu_ground_plane.py`); rotates so ground is z=0.
5. Uses the MaskDINO `pig` mask to select pig points and the `upper_body`
   mask to resolve the head-to-tail heading (the minimum-area rectangle of
   the pig mask gives the long axis; the upper-body centroid picks which
   end is the head).
6. Max-z-per-pixel BEV projection at a 1 cm × 1 cm grid, centered and
   cropped to the final 96×224 dorsal height map.
7. Writes one row per frame to `dataset.h5` under `height_maps` /
   `bag_names` / `source_names`.

Dependencies that ship alongside the script:

- `msu_ground_plane.py` — per-date ground-plane fitting utilities.

MSU-specific paths (`BAGFILES_DIR`, `CSV_PATH`, `CAMERA_INI`,
`DEFAULT_MASKDINO_CONFIG`, `DEFAULT_MASKDINO_WEIGHTS`, `OUTPUT_DIR`) are
set at the top of `build_height_dataset.py`; override via the matching
CLI flags when running on your data.

## Stage 4 — labels from slaughter CSV

```bash
python preprocessing/parse_labels.py \
    --csv path/to/slaughter_labels.csv \
    --hdf5_dir path/to/per_bag_h5_frames/ \
    --output data/label.h5
```

Expected CSV columns: `RosBagPath, UniqueID, Fat_r, Loin_r`. Multiple rows
per `UniqueID` are averaged. See `parse_labels.py` for details.
