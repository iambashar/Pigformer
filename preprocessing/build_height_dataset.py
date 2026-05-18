#!/usr/bin/env python3
"""
PigFormer — Stage 3: MaskDINO depth masks → 96x224 dorsal height map.

Per pig bag: RANSAC ground-plane fit, point-cloud rotation, min-area-rectangle
+ upper-body-guided heading, BEV max-z projection at 1 cm grid, lateral crop.

Mask acceptance criteria:
    a) Full mask must lie inside the central window
    b) Central window is image with a 15 px border on all sides

Usage:
    python build_height_dataset.py --target_bag rosbag2_2025_08_01-10_23_15
    python build_height_dataset.py --max_frames_per_bag 20
"""

import os
import sys
import csv
import re
import argparse
import configparser
from os.path import join, isfile, isdir, basename, dirname, abspath
from glob import glob
from collections import defaultdict

import numpy as np
import cv2
import h5py
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

import msu_ground_plane as ground_plane_utils

# MaskDINO depth-inference integration.
MASKDINO_ROOT = '/mnt/scratch/basharmk/data/body_condition/unl/MaskDINO'
MASKDINO_TOOLS = join(MASKDINO_ROOT, 'tools')
if MASKDINO_TOOLS not in sys.path:
    sys.path.append(MASKDINO_TOOLS)

try:
    import infer_pig_depth_h5 as maskdino_depth_infer  # type: ignore[import-not-found]
except ImportError:
    import traceback
    traceback.print_exc()
    print("ERROR: MaskDINO depth inference helper not found.")
    print(f"  Expected script: {join(MASKDINO_TOOLS, 'infer_pig_depth_h5.py')}")
    sys.exit(1)

# ---------------------------------------------------------------------------
SCRIPT_DIR = dirname(abspath(__file__))
DATA_ROOT = dirname(SCRIPT_DIR)  # swine_rgbd_data/
SHARED_DIR = join(DATA_ROOT, 'body-condition-shared')

# from convnext_ag import ConvNeXt # Removed as requested

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BAGFILES_DIR = ground_plane_utils.resolve_bagfiles_dir(join(DATA_ROOT, 'bagfiles'))
CSV_PATH = join(SCRIPT_DIR, 'csv', 'Body_Condition_Score_Dataset - MSU_All_Final.csv')
DEFAULT_DATE_GROUND_PLANE_JSON = ground_plane_utils.DEFAULT_DATE_GROUND_PLANE_JSON
CAMERA_INI = join(SHARED_DIR, 'CameraParam_Color1920x1080_Depth640x576.ini')
# ANGLE_CHECKPOINT = join(SCRIPT_DIR, 'angle_model_checkpoint.tar') # Removed
OUTPUT_DIR = join(SCRIPT_DIR, 'preprocessed_height_maps')
DATASET_H5_PATH = join(OUTPUT_DIR, 'dataset.h5')
VIS_DIR = join(OUTPUT_DIR, 'vis')
DEFAULT_MASKDINO_CONFIG = join(
    '/mnt/scratch/basharmk/data/body_condition/unl',
    'MaskDINO',
    'configs',
    'pig_depth',
    'instance-segmentation',
    'maskdino_R50_depth_instance_endpoint_pig_upper_body.yaml',
)
DEFAULT_MASKDINO_WEIGHTS = join(
    '/mnt/scratch/basharmk/data/body_condition/unl',
    'MaskDINO',
    'output',
    'pig_depth_endpoint_pig_upper_body_full_20260321_4gpu',
    'model_best.pth',
)

# Fixed ground plane
GROUND_A = 0.024
GROUND_B = 0.088
GROUND_C = 2.353

load_date_ground_planes = ground_plane_utils.load_date_ground_planes
resolve_ground_plane_for_bag = ground_plane_utils.resolve_ground_plane_for_bag

# Height map params
H_MAP_SIZE = 224
PIXEL_LEN = 0.01
H_MAX = 1.0
VIS_H_MAX = 1.0
GROUND_HEIGHT_THRESH_M = 0.03
FINAL_CROP_HEIGHT = 96
FINAL_CROP_WIDTH = 224
DEPTH_UNDISTORT_FX_SCALE = 0.85
_DEPTH_UNDISTORT_CACHE = {}
_XY_TABLE_CACHE = {}

# GT target names
TARGET_NAMES = ['fat_rib12', 'loin_rib12', 'fat_lumbar1', 'loin_lumbar1', 'caliper_score']

# Frame is accepted only when the full pig mask is inside this inner window border.
MASK_CENTER_BORDER_PX = 0  # was 15; rejected full-body pigs that touch FOV edge
                            # (e.g. 06_11 cam setup where pigs often reach right edge,
                            # 11_05 setup where they reach left edge). v1 dataset.h5
                            # included these bags so the check was either looser or
                            # bypassed in v1's build.
AREA_OUTLIER_IQR_SCALE = 1.5
AREA_OUTLIER_MIN_SAMPLES = 5
ANGLE_SELECTION_REFERENCE_OFFSETS = {
    '20250611': 180.0,
}

# Session-specific rotation angles (month → degrees)
SESSION_ROTATION_ANGLES = {
    '02': 0,     # 20250212
    '06': 270,    # 20250611
    '08': 0,     # 20250801
    '11': 180,   # 20251105
    '12': 180,   # 20251204
}

SESSION_CAMERA_PARAM_FALLBACKS = {
    '20250213': '20250212',
}

def get_rotation_angle_for_bag(bag_name):
    """Return the fixed rotation angle (degrees) based on session month."""
    name = re.sub(r'^rosbag2_', '', bag_name)
    flat = name.replace('_', '').replace('-', '')
    m = re.search(r'20\d{2}(\d{2})\d{2}', flat)
    if m:
        month = m.group(1)
        return SESSION_ROTATION_ANGLES.get(month, 0)
    return 0


def extract_bag_date_yyyymmdd(bag_name):
    name = basename(str(bag_name)).replace('.h5', '')
    rosbag_match = re.search(r'rosbag2_(\d{4})_(\d{2})_(\d{2})', name)
    if rosbag_match:
        return ''.join(rosbag_match.groups())
    flat = name.replace('_', '').replace('-', '')
    date_match = re.search(r'(20\d{6})', flat)
    if date_match:
        return date_match.group(1)
    return None


def choose_angle_closest_to_reference(predicted_deg, reference_deg):
    pred = float(predicted_deg)
    ref = float(reference_deg)
    candidates = [pred, pred + 180.0]
    return min(candidates, key=lambda angle: _angle_distance_deg(angle, ref))


def get_angle_selection_reference_deg(bag_name):
    base_reference = float(get_rotation_angle_for_bag(bag_name))
    bag_date = extract_bag_date_yyyymmdd(bag_name)
    extra_offset = 0.0 if bag_date is None else float(ANGLE_SELECTION_REFERENCE_OFFSETS.get(bag_date, 0.0))
    return float(base_reference + extra_offset)


def _compute_iqr_inlier_mask(values, iqr_scale=AREA_OUTLIER_IQR_SCALE, min_samples=AREA_OUTLIER_MIN_SAMPLES):
    vals = np.asarray(values, dtype=np.float32)
    keep_mask = np.ones(vals.shape, dtype=bool)
    if vals.size < int(min_samples):
        return keep_mask, None

    finite_vals = vals[np.isfinite(vals)]
    if finite_vals.size < int(min_samples):
        return keep_mask, None

    q1, q3 = np.percentile(finite_vals, [25.0, 75.0])
    iqr = float(q3 - q1)
    if iqr <= 0.0:
        lo = float(np.min(finite_vals))
        hi = float(np.max(finite_vals))
    else:
        lo = float(q1 - float(iqr_scale) * iqr)
        hi = float(q3 + float(iqr_scale) * iqr)
    keep_mask = np.isfinite(vals) & (vals >= lo) & (vals <= hi)
    return keep_mask, (lo, hi)


def filter_area_outlier_entries(entries, bag_name):
    if len(entries) < AREA_OUTLIER_MIN_SAMPLES:
        return list(entries)

    depth_mask_areas = [entry['depth_mask_area'] for entry in entries]
    height_map_areas = [entry['height_map_area'] for entry in entries]
    keep_depth, depth_bounds = _compute_iqr_inlier_mask(depth_mask_areas)
    keep_height, height_bounds = _compute_iqr_inlier_mask(height_map_areas)
    keep_mask = keep_depth & keep_height

    if not np.any(keep_mask):
        print(f"  Area-outlier filter removed every frame for {bag_name}; keeping all candidates")
        return list(entries)

    kept_entries = [entry for entry, keep in zip(entries, keep_mask) if keep]
    dropped = len(entries) - len(kept_entries)
    if dropped > 0:
        depth_lo, depth_hi = depth_bounds if depth_bounds is not None else (float('-inf'), float('inf'))
        height_lo, height_hi = height_bounds if height_bounds is not None else (float('-inf'), float('inf'))
        print(
            f"  Dropped {dropped}/{len(entries)} area-outlier frames for {bag_name} "
            f"(depth_area in [{depth_lo:.1f}, {depth_hi:.1f}], "
            f"height_area in [{height_lo:.1f}, {height_hi:.1f}])"
        )
    return kept_entries


def _float_or_nan(val):
    val = val.strip() if isinstance(val, str) else val
    try:
        return float(val)
    except (ValueError, TypeError, AttributeError):
        return float('nan')


def _flat_bag_to_rosbag2_name(bag_name):
    match = re.fullmatch(r'(\d{8})_(\d{6})', bag_name)
    if not match:
        return None
    date_part, time_part = match.groups()
    return (
        f"rosbag2_{date_part[:4]}_{date_part[4:6]}_{date_part[6:8]}"
        f"-{time_part[:2]}_{time_part[2:4]}_{time_part[4:6]}"
    )


def _rosbag2_to_flat_bag_name(bag_name):
    match = re.fullmatch(r'rosbag2_(\d{4})_(\d{2})_(\d{2})-(\d{2})_(\d{2})_(\d{2})', bag_name)
    if not match:
        return None
    year, month, day, hour, minute, second = match.groups()
    return f"{year}{month}{day}_{hour}{minute}{second}"


def build_bagfile_index(bagfiles_dir):
    bagfile_index = {}
    for path in sorted(glob(join(bagfiles_dir, '*.h5'))):
        exact_name = basename(path).replace('.h5', '')
        bagfile_index[exact_name] = path
        flat_name = _rosbag2_to_flat_bag_name(exact_name)
        if flat_name:
            bagfile_index.setdefault(flat_name, path)
        rosbag2_name = _flat_bag_to_rosbag2_name(exact_name)
        if rosbag2_name:
            bagfile_index.setdefault(rosbag2_name, path)
    return bagfile_index


def build_bagfile_time_index(bagfiles_dir):
    time_index = []
    for path in sorted(glob(join(bagfiles_dir, '*.h5'))):
        try:
            with h5py.File(path, 'r') as f:
                color_ts = f['color/timestamps_ns'][:]
            if len(color_ts) == 0:
                continue
            time_index.append({
                'path': path,
                'bag_name': basename(path).replace('.h5', ''),
                'start_s': int(color_ts[0] // 1_000_000_000),
                'end_s': int(color_ts[-1] // 1_000_000_000),
            })
        except Exception:
            continue
    return time_index


def resolve_bagfile_path(bag_name, bagfile_index, bagfile_time_index=None, start_time=None, end_time=None):
    candidate = bag_name.replace('.h5', '')
    path = bagfile_index.get(candidate)
    if path is None:
        rosbag2_name = _flat_bag_to_rosbag2_name(candidate)
        if rosbag2_name is not None:
            path = bagfile_index.get(rosbag2_name)
    if path is None:
        flat_name = _rosbag2_to_flat_bag_name(candidate)
        if flat_name is not None:
            path = bagfile_index.get(flat_name)
    if path is None and bagfile_time_index is not None and not np.isnan(start_time) and not np.isnan(end_time):
        matches = [
            item for item in bagfile_time_index
            if item['start_s'] <= int(start_time) and int(end_time) <= item['end_s']
        ]
        if len(matches) == 1:
            path = matches[0]['path']
    if path is None:
        return None, None
    resolved_name = basename(path).replace('.h5', '')
    return resolved_name, path


# ===========================================================================
# Camera utilities (same as preprocess_new.py)
# ===========================================================================

def load_camera_parameters(ini_file):
    if not isfile(ini_file):
        return None
    config = configparser.ConfigParser()
    config.read(ini_file)
    params = {}
    for prefix in ['Color', 'Depth']:
        section = f'{prefix}Intrinsic'
        params.update({
            section: {
                'fx': config.getfloat(section, 'fx'),
                'fy': config.getfloat(section, 'fy'),
                'cx': config.getfloat(section, 'cx'),
                'cy': config.getfloat(section, 'cy'),
                'width': config.getint(section, 'width') if config.has_option(section, 'width') else (1920 if prefix == 'Color' else 640),
                'height': config.getint(section, 'height') if config.has_option(section, 'height') else (1080 if prefix == 'Color' else 576),
            },
            f'{prefix}Distortion': {
                'k1': config.getfloat(f'{prefix}Distortion', 'k1'),
                'k2': config.getfloat(f'{prefix}Distortion', 'k2'),
                'p1': config.getfloat(f'{prefix}Distortion', 'p1'),
                'p2': config.getfloat(f'{prefix}Distortion', 'p2'),
                'k3': config.getfloat(f'{prefix}Distortion', 'k3'),
                'k4': config.getfloat(f'{prefix}Distortion', 'k4') if config.has_option(f'{prefix}Distortion', 'k4') else 0.0,
                'k5': config.getfloat(f'{prefix}Distortion', 'k5') if config.has_option(f'{prefix}Distortion', 'k5') else 0.0,
                'k6': config.getfloat(f'{prefix}Distortion', 'k6') if config.has_option(f'{prefix}Distortion', 'k6') else 0.0,
            }
        })

    ext_section = 'D2CTransformParam' if config.has_section('D2CTransformParam') else 'Extrinsic'
    for i in range(9):
        params[f'rot{i}'] = config.getfloat(ext_section, f'rot{i}')
    for i in range(3):
        params[f'trans{i}'] = config.getfloat(ext_section, f'trans{i}')
    return params


def get_camera_params_for_bag(bag_name, hdf5_path=None):
    """Pick the correct session-specific .ini file. If `hdf5_path` is
    provided and the INI lookup fails (e.g. UNL bags), fall back to the
    `depth/camera_info` group inside the bag HDF5."""
    cam = ground_plane_utils.get_camera_params_for_bag(bag_name)
    if cam is not None:
        return cam
    if hdf5_path is None:
        return None
    intrinsic, distortion = ground_plane_utils.load_depth_camera_info_from_hdf5(hdf5_path)
    if intrinsic is None:
        return None
    return {
        'DepthIntrinsic': intrinsic,
        'DepthDistortion': ground_plane_utils.distortion_array_to_dict(distortion),
    }


def make_scaled_camera_matrix(camera_matrix, fx_scale=DEPTH_UNDISTORT_FX_SCALE):
    scaled = np.asarray(camera_matrix, dtype=np.float32).copy()
    scaled[0, 0] = float(scaled[0, 0] * fx_scale)
    scaled[1, 1] = float(scaled[1, 1] * fx_scale)
    return scaled


def _intrinsic_cache_key(intrinsic):
    return (
        float(intrinsic['fx']),
        float(intrinsic['fy']),
        float(intrinsic['cx']),
        float(intrinsic['cy']),
    )


def _distortion_cache_key(distortion):
    return tuple(
        float(distortion.get(name, 0.0))
        for name in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')
    )


def _get_depth_undistort_cache_entry(image_shape, intrinsic, distortion):
    height, width = int(image_shape[0]), int(image_shape[1])
    cache_key = (
        height,
        width,
        float(DEPTH_UNDISTORT_FX_SCALE),
        _intrinsic_cache_key(intrinsic),
        _distortion_cache_key(distortion),
    )
    cached = _DEPTH_UNDISTORT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    camera_matrix = np.array([
        [intrinsic['fx'], 0, intrinsic['cx']],
        [0, intrinsic['fy'], intrinsic['cy']],
        [0, 0, 1],
    ], dtype=np.float32)
    dist_coeffs = np.array([
        distortion['k1'], distortion['k2'], distortion['p1'], distortion['p2'],
        distortion['k3'], distortion.get('k4', 0), distortion.get('k5', 0), distortion.get('k6', 0),
    ], dtype=np.float32)
    camera_matrix2 = make_scaled_camera_matrix(camera_matrix, DEPTH_UNDISTORT_FX_SCALE)
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix2, (width, height), cv2.CV_32FC1)
    intrinsic2 = {
        **intrinsic,
        'fx': float(camera_matrix2[0, 0]),
        'fy': float(camera_matrix2[1, 1]),
        'cx': float(camera_matrix2[0, 2]),
        'cy': float(camera_matrix2[1, 2]),
        'width': width,
        'height': height,
    }
    cached = (map1, map2, intrinsic2)
    _DEPTH_UNDISTORT_CACHE[cache_key] = cached
    return cached


def undistort_depth_image2(image, intrinsic, distortion):
    map1, map2, intrinsic2 = _get_depth_undistort_cache_entry(image.shape[:2], intrinsic, distortion)
    undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_NEAREST)
    return undistorted_image, intrinsic2


def undistort_color_image(image, intrinsic, distortion):
    camera_matrix = np.array([
        [intrinsic['fx'], 0, intrinsic['cx']],
        [0, intrinsic['fy'], intrinsic['cy']],
        [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([
        distortion['k1'], distortion['k2'], distortion['p1'], distortion['p2'],
        distortion['k3'], distortion.get('k4', 0), distortion.get('k5', 0), distortion.get('k6', 0)
    ])
    height, width = image.shape[:2]
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix, (width, height), cv2.CV_32FC1)
    undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)
    return undistorted_image


def create_xy_table(depth_intrinsic):
    width = int(depth_intrinsic['width'])
    height = int(depth_intrinsic['height'])
    fx = float(depth_intrinsic['fx'])
    fy = float(depth_intrinsic['fy'])
    cx = float(depth_intrinsic['cx'])
    cy = float(depth_intrinsic['cy'])
    cache_key = (width, height, fx, fy, cx, cy)
    cached = _XY_TABLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    x = np.arange(width, dtype=np.float32) - cx
    y = np.arange(height, dtype=np.float32) - cy
    xx, yy = np.meshgrid(x / fx, y / fy)
    mesh = np.dstack((xx, yy)).astype(np.float32, copy=False)
    _XY_TABLE_CACHE[cache_key] = mesh
    return mesh


def generate_point_cloud(depth_image, xy_table):
    valid_mask = (depth_image > 0.)
    point_cloud_data = np.empty((*depth_image.shape, 3), dtype=np.float32)
    point_cloud_data[..., 0] = xy_table[..., 0] * depth_image
    point_cloud_data[..., 1] = xy_table[..., 1] * depth_image
    point_cloud_data[..., 2] = depth_image
    pcd = point_cloud_data[valid_mask]
    return pcd


def proj2plane(x_pts, y_pts, z_pts, a, b, c):
    k = (-c + z_pts - a * x_pts - b * y_pts) / (a**2 + b**2 + 1)
    x_pts1 = x_pts + k * a
    y_pts1 = y_pts + k * b
    z_pts1 = z_pts + k * (-1)
    return x_pts1, y_pts1, z_pts1


def compute_rotation_matrix(a, b, c):
    x_pts = np.array([0, 1])
    y_pts = np.array([0, 0])
    z_pts = np.array([0, 0])
    xs_proj, ys_proj, zs_proj = proj2plane(x_pts, y_pts, z_pts, a, b, c)
    vect_x_bev = np.array([xs_proj[1] - xs_proj[0],
                           ys_proj[1] - ys_proj[0],
                           zs_proj[1] - zs_proj[0]])
    vect_x_bev = vect_x_bev / np.linalg.norm(vect_x_bev)
    vect_z_bev = np.array([a, b, -1])
    vect_z_bev = vect_z_bev / np.linalg.norm(vect_z_bev)
    vect_y_bev = np.cross(vect_x_bev, vect_z_bev)
    rot_mat = np.stack([vect_x_bev, vect_y_bev, vect_z_bev], axis=1).T
    return rot_mat


# ===========================================================================
# Step 1: MaskDINO pig detection on depth frames
# ===========================================================================

def mask_inside_central_window(mask_bool, border=MASK_CENTER_BORDER_PX):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return False
    h, w = mask_bool.shape
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return (
        x_min >= int(border)
        and y_min >= int(border)
        and x_max <= int(w - 1 - border)
        and y_max <= int(h - 1 - border)
    )


def init_maskdino_depth_model(config_file, weights, score_threshold=0.2,
                              encoding='depth_valid_gradient', device_override='',
                              torch_compile=False):
    if not isfile(config_file):
        raise FileNotFoundError(f"MaskDINO config not found: {config_file}")
    if not isfile(weights):
        raise FileNotFoundError(f"MaskDINO weights not found: {weights}")

    args = argparse.Namespace(
        config_file=config_file,
        weights=weights,
        device=device_override,
        score_threshold=float(score_threshold),
    )
    cfg = maskdino_depth_infer.setup_cfg(args)
    model, augment = maskdino_depth_infer.build_model_and_augment(cfg)
    device = torch.device(cfg.MODEL.DEVICE)

    # Optional inference-speed knobs (the 2.8x compile win we benchmarked).
    if device.type == 'cuda':
        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    if torch_compile and device.type == 'cuda':
        try:
            # dynamic=True so variable batch sizes (e.g. last partial batch
            # smaller than nominal) don't trigger recompilation. Also raise
            # recompile_limit since detectron2 has multiple shape branches.
            torch._dynamo.config.recompile_limit = 32
            model = torch.compile(model, mode='reduce-overhead', dynamic=True)
        except Exception as e:
            print(f"[init_maskdino_depth_model] torch.compile failed: {e}; running uncompiled.")

    return model, augment, device, str(encoding)


def select_best_masks_by_class(output, shape, class_indices, score_threshold):
    instances = output["instances"].to("cpu")
    empty_mask = np.zeros(shape, dtype=np.uint8)
    if len(instances) == 0:
        return {name: empty_mask.copy() for name in class_indices}

    pred_classes = instances.pred_classes.numpy()
    pred_scores = instances.scores.numpy()
    pred_masks = instances.pred_masks.numpy().astype(np.uint8)

    selected_masks = {}
    for class_name, class_index in class_indices.items():
        class_matches = np.where(pred_classes == int(class_index))[0]
        if class_matches.size == 0:
            selected_masks[class_name] = empty_mask.copy()
            continue
        best_position = class_matches[int(np.argmax(pred_scores[class_matches]))]
        best_score = float(pred_scores[best_position])
        if best_score < float(score_threshold):
            selected_masks[class_name] = empty_mask.copy()
            continue
        selected_masks[class_name] = pred_masks[best_position]
    return selected_masks


def detect_pigs_maskdino_depth(hdf5_path, maskdino_model, maskdino_augment, maskdino_device,
                               start_ns, end_ns, batch_size=8, score_threshold=0.2,
                               pig_class_index=0, upper_body_class_index=1,
                               encoding='depth_valid_gradient',
                               camera_params=None,
                               central_window_border=None,
                               min_pig_px=0):
    """Detect pigs directly from depth frames.

    Returns list of
    (color_idx, depth_idx, pig_mask_bool, upper_body_mask_bool, bbox_xywhn_in_depth_frame).

    For encoding='raw_meters_1ch' (MaskDINO v2), depth is undistorted using
    `camera_params` before being fed to the model — the v2 weights were
    trained against undistorted depth + GT polygons, so the input must
    match that coordinate system. Mask outputs are in undistorted coords;
    downstream BEV projection in this script also operates in undistorted
    coords (`undistort_depth_image2`) so shapes line up.
    """
    undistort_for_maskdino = (encoding == 'raw_meters_1ch')
    if undistort_for_maskdino and camera_params is None:
        raise RuntimeError("encoding 'raw_meters_1ch' requires camera_params (v2 model "
                           "trained on undistorted input).")

    with h5py.File(hdf5_path, 'r') as f:
        n_depth = int(f.attrs.get('n_depth', 0))
        if n_depth == 0:
            return []

        # Build undistortion remap once if needed. newCameraMatrix uses the
        # SAME DEPTH_UNDISTORT_FX_SCALE as `undistort_depth_image2`, so the
        # depth grid the model sees here matches the depth grid that BEV
        # projection in `generate_height_map` operates on later. Mixing scales
        # silently misaligns masks vs depth — see CLAUDE.md "fx_scale".
        depth_remap_x = depth_remap_y = None
        if undistort_for_maskdino:
            di = camera_params['DepthIntrinsic']
            dd = camera_params['DepthDistortion']
            K = np.array([[di['fx'], 0, di['cx']],
                          [0, di['fy'], di['cy']],
                          [0, 0, 1]], dtype=np.float64)
            D = np.array([dd[k] for k in ('k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6')],
                         dtype=np.float64)
            K_new = K.copy()
            K_new[0, 0] *= DEPTH_UNDISTORT_FX_SCALE
            K_new[1, 1] *= DEPTH_UNDISTORT_FX_SCALE
            shape = f['depth/images'].shape  # (N, H, W)
            H_d, W_d = int(shape[1]), int(shape[2])
            depth_remap_x, depth_remap_y = cv2.initUndistortRectifyMap(
                K, D, np.eye(3), K_new, (W_d, H_d), cv2.CV_32FC1)

        depth_ts = f['depth/timestamps_ns'][:]
        color_ts = f['color/timestamps_ns'][:] if 'color/timestamps_ns' in f else None

        if np.isnan(start_ns) or np.isnan(end_ns):
            start_ns_int = int(depth_ts[0])
            end_ns_int = int(depth_ts[-1])
        else:
            start_ns_int = int(start_ns * 1e9)
            end_ns_int = int(end_ns * 1e9)

        depth_indices = np.where((depth_ts >= start_ns_int) & (depth_ts <= end_ns_int))[0]
        if len(depth_indices) == 0:
            return []

        # Bulk-read all needed depth frames in one HDF5 access (avoids
        # per-batch random-access overhead which was ~20% of profile time).
        # ~1.5 MB per frame × N frames; safe to materialize for typical
        # bag sizes (< 500 frames → < 750 MB).
        all_depth = np.asarray(f['depth/images'][list(depth_indices)], dtype=np.float32)

        detections = []
        for batch_start in range(0, len(depth_indices), int(batch_size)):
            batch_idx = depth_indices[batch_start:batch_start + int(batch_size)]
            depth_batch = all_depth[batch_start:batch_start + int(batch_size)]
            if depth_remap_x is not None:
                depth_batch = np.stack([
                    cv2.remap(d.astype(np.uint16), depth_remap_x, depth_remap_y,
                              cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for d in depth_batch
                ]).astype(np.float32)
            h_d, w_d = depth_batch.shape[1], depth_batch.shape[2]

            batch_inputs = maskdino_depth_infer.prepare_batch_inputs(
                depth_batch=depth_batch,
                augment=maskdino_augment,
                device=maskdino_device,
                encoding=encoding,
            )

            with torch.inference_mode():
                with torch.autocast(
                    device_type='cuda',
                    dtype=torch.float16,
                    enabled=bool(maskdino_device.type == 'cuda'),
                ):
                    batch_outputs = maskdino_model(batch_inputs)

            for i, di in enumerate(batch_idx):
                selected_masks = select_best_masks_by_class(
                    output=batch_outputs[i],
                    shape=(h_d, w_d),
                    class_indices={
                        'pig': int(pig_class_index),
                        'upper_body': int(upper_body_class_index),
                    },
                    score_threshold=float(score_threshold),
                )
                pig_mask_u8 = selected_masks['pig']
                upper_body_mask_u8 = selected_masks['upper_body']
                if int(np.count_nonzero(pig_mask_u8)) == 0:
                    continue

                pig_mask_bool = pig_mask_u8.astype(bool)
                upper_body_mask_bool = upper_body_mask_u8.astype(bool)
                if pig_mask_bool.ndim != 2:
                    continue
                _border = MASK_CENTER_BORDER_PX if central_window_border is None else int(central_window_border)
                if not mask_inside_central_window(pig_mask_bool, border=_border):
                    continue
                if min_pig_px and int(pig_mask_bool.sum()) < int(min_pig_px):
                    continue

                ys, xs = np.where(pig_mask_bool)
                if len(xs) == 0:
                    continue

                x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
                bbox_xywhn = np.array([
                    (x1 + x2) / 2.0 / float(w_d),
                    (y1 + y2) / 2.0 / float(h_d),
                    (x2 - x1) / float(w_d),
                    (y2 - y1) / float(h_d),
                ], dtype=np.float32)

                if color_ts is not None and len(color_ts) > 0:
                    dt = np.abs(color_ts.astype(np.int64) - int(depth_ts[int(di)]))
                    ci = int(np.argmin(dt))
                else:
                    ci = int(di)

                detections.append((
                    ci,
                    int(di),
                    pig_mask_bool,
                    upper_body_mask_bool,
                    bbox_xywhn,
                ))

    return detections


# ===========================================================================
# UNet stage-1: same contract as detect_pigs_maskdino_depth, just faster.
# ===========================================================================

def detect_pigs_unet_depth(hdf5_path, unet_model, hdf5_device,
                           start_ns, end_ns, camera_params,
                           batch_size=16, score_threshold=0.5,
                           min_pig_px=1000, central_window_border=None):
    """Run the UNet on every depth frame in [start_ns, end_ns] of the bag's
    HDF5. Returns (color_idx, depth_idx, pig_mask_bool, upper_body_mask_bool,
    bbox_xywhn) tuples — same contract as detect_pigs_maskdino_depth.

    Frames where the predicted pig mask has fewer than min_pig_px pixels
    above threshold, or where the mask touches the central-window border,
    are dropped (mirrors the MaskDINO pipeline's filtering).
    """
    from unet_depth import predict_masks, undistort_for_unet

    with h5py.File(hdf5_path, 'r') as f:
        n_depth = int(f.attrs.get('n_depth', 0))
        if n_depth == 0:
            return []
        depth_ts = f['depth/timestamps_ns'][:]
        color_ts = f['color/timestamps_ns'][:] if 'color/timestamps_ns' in f else None

        if np.isnan(start_ns) or np.isnan(end_ns):
            start_ns_int = int(depth_ts[0]); end_ns_int = int(depth_ts[-1])
        else:
            start_ns_int = int(start_ns * 1e9); end_ns_int = int(end_ns * 1e9)
        depth_indices = np.where((depth_ts >= start_ns_int) & (depth_ts <= end_ns_int))[0]
        if len(depth_indices) == 0:
            return []

        # Bulk-read all relevant raw depth frames once.
        raw_batch = np.asarray(f['depth/images'][list(depth_indices)], dtype=np.uint16)

    border = MASK_CENTER_BORDER_PX if central_window_border is None else int(central_window_border)

    detections = []
    for batch_start in range(0, len(depth_indices), int(batch_size)):
        batch_idx = depth_indices[batch_start:batch_start + int(batch_size)]
        raw_chunk = raw_batch[batch_start:batch_start + int(batch_size)]
        # Undistort each frame in the chunk (loop — cv2.initUndistortRectifyMap
        # is cached by lru behavior inside undistort_for_unet's call to cv2).
        und = np.stack([undistort_for_unet(r, camera_params) for r in raw_chunk])
        probs = predict_masks(unet_model, und, device=hdf5_device)
        for i, di in enumerate(batch_idx):
            pig_mask = probs[i, 0] > float(score_threshold)
            ub_mask  = probs[i, 1] > float(score_threshold)
            if int(pig_mask.sum()) < int(min_pig_px):
                continue
            if not mask_inside_central_window(pig_mask, border=border):
                continue

            ys, xs = np.where(pig_mask)
            if len(xs) == 0:
                continue
            h_d, w_d = pig_mask.shape
            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            bbox_xywhn = np.array([
                (x1 + x2) / 2.0 / float(w_d),
                (y1 + y2) / 2.0 / float(h_d),
                (x2 - x1) / float(w_d),
                (y2 - y1) / float(h_d),
            ], dtype=np.float32)

            if color_ts is not None and len(color_ts) > 0:
                dt = np.abs(color_ts.astype(np.int64) - int(depth_ts[int(di)]))
                ci = int(np.argmin(dt))
            else:
                ci = int(di)

            detections.append((ci, int(di), pig_mask, ub_mask, bbox_xywhn))

    return detections


# ===========================================================================
# Height map generation (depth mask driven)
# ===========================================================================

def _undistort_endpoint_to_depth_pixel(endpoint_xy, depth_intrinsic, depth_distortion, depth_shape):
    if endpoint_xy is None or len(endpoint_xy) != 4:
        return None

    h, w = depth_shape
    out = []
    for x_raw, y_raw in ((endpoint_xy[0], endpoint_xy[1]), (endpoint_xy[2], endpoint_xy[3])):
        xi = int(np.clip(round(float(x_raw)), 0, w - 1))
        yi = int(np.clip(round(float(y_raw)), 0, h - 1))
        marker = np.zeros((h, w), dtype=np.uint16)
        marker[yi, xi] = 1000
        marker_undist, _ = undistort_depth_image2(marker, depth_intrinsic, depth_distortion)
        ys, xs = np.where(marker_undist > 0)
        if len(xs) == 0:
            return None
        out.append((float(np.mean(xs)), float(np.mean(ys))))
    return np.asarray(out, dtype=np.float32)


def _nearest_valid_mask_point(point_xy, depth_mask, depth_img):
    if point_xy is None:
        return None
    valid = depth_mask & (depth_img > 0)
    ys, xs = np.where(valid)
    if len(xs) == 0:
        return None
    dx = xs.astype(np.float32) - float(point_xy[0])
    dy = ys.astype(np.float32) - float(point_xy[1])
    idx = int(np.argmin(dx * dx + dy * dy))
    return int(xs[idx]), int(ys[idx])


def _pixel_to_ground_xy(pixel_xy, depth_img, depth_intrinsic, rot_mat_al, ground_origin_grd, center_shift):
    if pixel_xy is None:
        return None
    px, py = int(pixel_xy[0]), int(pixel_xy[1])
    z = float(depth_img[py, px])
    if z <= 0:
        return None
    x = (float(px) - float(depth_intrinsic['cx'])) * z / float(depth_intrinsic['fx'])
    y = (float(py) - float(depth_intrinsic['cy'])) * z / float(depth_intrinsic['fy'])
    pt_grd = rot_mat_al @ np.array([x, y, z], dtype=np.float32)
    pt_grd = pt_grd - ground_origin_grd - center_shift
    return pt_grd[:2]


def _ground_xy_to_bev_xy(point_xy, x_min_hm, y_min_hm, pixel_len, width, height):
    if point_xy is None:
        return None
    px = (float(point_xy[0]) - x_min_hm) / pixel_len
    py = (float(point_xy[1]) - y_min_hm) / pixel_len
    if not (0.0 <= px < float(width) and 0.0 <= py < float(height)):
        return None
    return np.array([px, py], dtype=np.float32)


def generate_height_map(depth_img_raw, color_img_raw, camera_params,
                        a, b, c, rot_mat_al, pig_bbox_xywhn,
                        depth_mask_raw, upper_body_mask_raw=None,
                        masks_already_undistorted=False):
    """Generate a BEV height map from a depth-frame pig mask.

    `masks_already_undistorted=True` is set when the mask producer (e.g.
    MaskDINO v2 trained on undistorted depth) outputs masks already in
    undistorted pixel coordinates. In that case we skip the mask
    undistortion below to avoid double-undistorting (which shrinks the
    mask further).
    """
    depth_intrinsic = camera_params['DepthIntrinsic']
    depth_distortion = camera_params['DepthDistortion']
    undist_depth, intrinsic2 = undistort_depth_image2(depth_img_raw.copy(), depth_intrinsic, depth_distortion)
    depth_img = undist_depth.astype(np.float32) / 1000.0
    depth_img[depth_img > 3.5] = 0

    if masks_already_undistorted:
        depth_mask = depth_mask_raw.astype(bool)
    else:
        # Keep mask geometry aligned with the undistorted depth frame.
        pig_mask_undist_u16, _ = undistort_depth_image2(
            (depth_mask_raw.astype(np.uint16) * 1000), depth_intrinsic, depth_distortion)
        depth_mask = pig_mask_undist_u16 > 0
    if depth_mask.shape != depth_img.shape:
        depth_mask = cv2.resize(
            depth_mask.astype(np.uint8),
            (depth_img.shape[1], depth_img.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    upper_body_mask = np.zeros_like(depth_mask, dtype=bool)
    if upper_body_mask_raw is not None:
        if masks_already_undistorted:
            upper_body_mask = upper_body_mask_raw.astype(bool)
        else:
            upper_body_undist_u16, _ = undistort_depth_image2(
                (upper_body_mask_raw.astype(np.uint16) * 1000), depth_intrinsic, depth_distortion)
            upper_body_mask = upper_body_undist_u16 > 0
        if upper_body_mask.shape != depth_img.shape:
            upper_body_mask = cv2.resize(
                upper_body_mask.astype(np.uint8),
                (depth_img.shape[1], depth_img.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        upper_body_mask &= depth_mask

    xy_table = create_xy_table(intrinsic2)
    pcd_all_cam = generate_point_cloud(depth_img, xy_table)
    if len(pcd_all_cam) == 0:
        return None, None, None, None, None, None, None, None, None, None

    valid_d = depth_img > 0
    m_sam = depth_mask[valid_d]
    pcd_pig_cam = pcd_all_cam[m_sam]
    if len(pcd_pig_cam) < 100:
        return None, None, None, None, None, None, None, None, None, None

    xs_floor, ys_floor, zs_floor = proj2plane(
        pcd_pig_cam[:, 0],
        pcd_pig_cam[:, 1],
        pcd_pig_cam[:, 2],
        a,
        b,
        c,
    )
    pcd_pig_floor_cam = np.stack([xs_floor, ys_floor, zs_floor], axis=1).astype(np.float32, copy=False)
    pcd_pig_floor_grd = (rot_mat_al @ pcd_pig_floor_cam.T).T

    pig_floor_xy = pcd_pig_floor_grd[:, :2].astype(np.float32)
    if len(pig_floor_xy) >= 5:
        (pig_floor_cx, pig_floor_cy), _, _ = cv2.minAreaRect(pig_floor_xy)
    else:
        pig_floor_cx = float(np.mean(pig_floor_xy[:, 0]))
        pig_floor_cy = float(np.mean(pig_floor_xy[:, 1]))
    pig_floor_cz = float(np.median(pcd_pig_floor_grd[:, 2]))
    ground_origin_grd = np.array([pig_floor_cx, pig_floor_cy, pig_floor_cz], dtype=np.float32)

    pcd_all_grd = (rot_mat_al @ pcd_all_cam.T).T
    pcd_all_grd = pcd_all_grd - ground_origin_grd[None, :]
    masked_z = pcd_all_grd[m_sam, 2] if np.any(m_sam) else np.empty((0,), dtype=np.float32)
    height_sign = -1.0 if masked_z.size and float(np.mean(masked_z)) < 0.0 else 1.0
    if height_sign < 0.0:
        pcd_all_grd = pcd_all_grd.copy()
        pcd_all_grd[:, 2] *= height_sign
    msk_above_grd = pcd_all_grd[:, 2] > GROUND_HEIGHT_THRESH_M
    pcd_pig_grd = pcd_all_grd[m_sam & msk_above_grd]
    if len(pcd_pig_grd) < 100:
        return None, None, None, None, None, None, None, None, None, None

    center_shift = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    if len(pcd_pig_grd) >= 5:
        pig_xy = pcd_pig_grd[:, :2].astype(np.float32)
        (pig_cx, pig_cy), _, _ = cv2.minAreaRect(pig_xy)
        center_shift = np.array([pig_cx, pig_cy, 0.0], dtype=np.float32)
        pcd_all_grd = pcd_all_grd - center_shift[None, :]
        pcd_pig_grd = pcd_pig_grd - center_shift[None, :]

    pixel_len = PIXEL_LEN
    h_map, w_map = H_MAP_SIZE, H_MAP_SIZE
    x_min_hm = -w_map / 2 * pixel_len
    y_min_hm = -h_map / 2 * pixel_len
    x_max_hm = w_map / 2 * pixel_len
    y_max_hm = h_map / 2 * pixel_len

    xs_pig, ys_pig, zs_pig = pcd_pig_grd[:, 0], pcd_pig_grd[:, 1], pcd_pig_grd[:, 2]
    msk = (xs_pig > x_min_hm) & (xs_pig < x_max_hm) & (ys_pig > y_min_hm) & (ys_pig < y_max_hm)
    xs_pig, ys_pig, zs_pig = xs_pig[msk], ys_pig[msk], zs_pig[msk]
    if len(xs_pig) < 50:
        return None, None, None, None, None, None, None, None, None, None

    xs_q = np.floor((xs_pig - x_min_hm) / pixel_len).astype(int)
    ys_q = np.floor((ys_pig - y_min_hm) / pixel_len).astype(int)
    dist_to_grid = (((xs_pig - x_min_hm) % pixel_len) - 0.5) ** 2 + (((ys_pig - y_min_hm) % pixel_len) - 0.5) ** 2
    order = np.argsort(dist_to_grid)[::-1]
    xs_q, ys_q, zs_pig = xs_q[order], ys_q[order], zs_pig[order]

    height_map = np.zeros((h_map, w_map), dtype=np.float32)
    height_map[ys_q, xs_q] = zs_pig
    height_map[height_map <= GROUND_HEIGHT_THRESH_M] = 0.0

    pig_bev_mask = np.zeros((h_map, w_map), dtype=bool)
    pig_bev_mask[ys_q, xs_q] = True

    upper_body_bev_mask = np.zeros((h_map, w_map), dtype=bool)
    upper_body_point_mask = upper_body_mask[valid_d] & msk_above_grd
    if np.any(upper_body_point_mask):
        pcd_upper_body_grd = pcd_all_grd[upper_body_point_mask]
        xs_upper, ys_upper = pcd_upper_body_grd[:, 0], pcd_upper_body_grd[:, 1]
        upper_in_bounds = (
            (xs_upper > x_min_hm) & (xs_upper < x_max_hm) &
            (ys_upper > y_min_hm) & (ys_upper < y_max_hm)
        )
        if np.any(upper_in_bounds):
            xs_upper_q = np.floor((xs_upper[upper_in_bounds] - x_min_hm) / pixel_len).astype(int)
            ys_upper_q = np.floor((ys_upper[upper_in_bounds] - y_min_hm) / pixel_len).astype(int)
            xs_upper_q = np.clip(xs_upper_q, 0, w_map - 1)
            ys_upper_q = np.clip(ys_upper_q, 0, h_map - 1)
            upper_body_bev_mask[ys_upper_q, xs_upper_q] = True

    ys_m, xs_m = np.where(depth_mask)
    if len(xs_m) > 0:
        depth_bbox = (int(xs_m.min()), int(ys_m.min()), int(xs_m.max() - xs_m.min()), int(ys_m.max() - ys_m.min()))
    else:
        depth_bbox = (0, 0, 0, 0)

    min_xy = np.min(pcd_pig_grd[:, :2], axis=0)
    max_xy = np.max(pcd_pig_grd[:, :2], axis=0)
    x1_bev = (min_xy[0] - x_min_hm) / pixel_len
    y1_bev = (min_xy[1] - y_min_hm) / pixel_len
    x2_bev = (max_xy[0] - x_min_hm) / pixel_len
    y2_bev = (max_xy[1] - y_min_hm) / pixel_len
    ground_bev_box = (x1_bev, y1_bev, x2_bev - x1_bev, y2_bev - y1_bev)

    corners_grd = np.array([
        [min_xy[0], min_xy[1], 0],
        [max_xy[0], min_xy[1], 0],
        [max_xy[0], max_xy[1], 0],
        [min_xy[0], max_xy[1], 0],
    ], dtype=np.float32)
    corners_depth_cam = (rot_mat_al.T @ (corners_grd + center_shift[None] + ground_origin_grd[None]).T).T
    d_fx = float(intrinsic2['fx'])
    d_fy = float(intrinsic2['fy'])
    d_cx = float(intrinsic2['cx'])
    d_cy = float(intrinsic2['cy'])
    u_d = corners_depth_cam[:, 0] * d_fx / corners_depth_cam[:, 2] + d_cx
    v_d = corners_depth_cam[:, 1] * d_fy / corners_depth_cam[:, 2] + d_cy
    depth_ground_box = np.stack([u_d, v_d], axis=1)

    x, y, w, h = depth_bbox
    yolo_depth_poly = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h],
    ], dtype=np.float32)

    return (
        height_map,
        depth_bbox,
        ground_bev_box,
        depth_ground_box,
        undist_depth,
        depth_mask,
        yolo_depth_poly,
        upper_body_mask,
        pig_bev_mask,
        upper_body_bev_mask,
    )


# ===========================================================================
# Angle prediction + rotation
# ===========================================================================

def _rotate_points_xy(points_xy, angle_deg, center_xy):
    pts = np.asarray(points_xy, dtype=np.float32)
    if pts.ndim == 1:
        pts = pts[None, :]
    theta = np.radians(float(angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    centered = pts - np.asarray(center_xy, dtype=np.float32)[None, :]
    rotated = np.empty_like(centered)
    rotated[:, 0] = centered[:, 0] * cos_t + centered[:, 1] * sin_t
    rotated[:, 1] = -centered[:, 0] * sin_t + centered[:, 1] * cos_t
    rotated += np.asarray(center_xy, dtype=np.float32)[None, :]
    return rotated


def front_edge_from_masks(mask, upper_body_mask):
    pig_mask = np.asarray(mask, dtype=bool)
    upper_mask = np.asarray(upper_body_mask, dtype=bool)
    if not np.any(pig_mask) or not np.any(upper_mask):
        return None, None, None

    obb_corners = obb_from_mask(pig_mask)
    if obb_corners is None:
        return None, None, None

    ordered = _ordered_box_vertices(np.asarray(obb_corners, dtype=np.float32))
    center_xy = np.mean(ordered, axis=0)
    upper_ys, upper_xs = np.where(upper_mask)
    upper_centroid_xy = np.array([float(np.mean(upper_xs)), float(np.mean(upper_ys))], dtype=np.float32)

    edges = []
    for index in range(4):
        p0 = ordered[index]
        p1 = ordered[(index + 1) % 4]
        midpoint = 0.5 * (p0 + p1)
        length = float(np.linalg.norm(p1 - p0))
        edges.append((p0, p1, midpoint, length))

    if not edges:
        return None, None, None
    min_length = min(edge[3] for edge in edges)
    short_edges = [edge for edge in edges if edge[3] <= 1.02 * min_length]
    if not short_edges:
        short_edges = sorted(edges, key=lambda edge: edge[3])[:2]
    front_edge = min(short_edges, key=lambda edge: float(np.linalg.norm(edge[2] - upper_centroid_xy)))
    front_midpoint = np.asarray(front_edge[2], dtype=np.float32)
    front_vector = front_midpoint - center_xy
    if float(np.linalg.norm(front_vector)) < 1e-6:
        return None, None, None

    # PIL rotates counter-clockwise in image coordinates; use the raw front-vector
    # angle so a front endpoint below the center maps to the right after rotation.
    angle_deg = float(np.degrees(np.arctan2(front_vector[1], front_vector[0])))
    front_edge_xy = np.stack([front_edge[0], front_edge[1]], axis=0).astype(np.float32)
    return ordered.astype(np.float32), front_edge_xy, angle_deg


def rotate_height_map(height_map, pig_bev_mask=None, upper_body_bev_mask=None):
    """Rotate the height map using pig and pig_upper_body masks, keeping the front on the right."""
    _, _, ag_deg = front_edge_from_masks(pig_bev_mask, upper_body_bev_mask)
    if ag_deg is None:
        return None, None
    ht_norm = np.clip(height_map / H_MAX, 0, 1)
    img = (ht_norm * 255).astype('uint8')
    img = Image.fromarray(img, mode='L')
    rotated = np.array(img.rotate(ag_deg, resample=Image.NEAREST))

    return rotated, ag_deg


def _angle_distance_deg(a_deg, b_deg):
    return abs(((float(a_deg) - float(b_deg) + 180.0) % 360.0) - 180.0)


def stabilize_rotation_angle(current_deg, previous_deg):
    if previous_deg is None:
        return float(current_deg)
    candidates = [
        float(current_deg) - 180.0,
        float(current_deg),
        float(current_deg) + 180.0,
    ]
    best = min(candidates, key=lambda angle: _angle_distance_deg(angle, previous_deg))
    return float(best)


def render_height_map_rotation(height_map, angle_deg):
    ht_norm = np.clip(height_map / H_MAX, 0, 1)
    img = (ht_norm * 255).astype('uint8')
    pil_img = Image.fromarray(img, mode='L')
    return np.array(pil_img.rotate(float(angle_deg), resample=Image.NEAREST))


def render_height_map_rotation_float(height_map, angle_deg):
    pil_img = Image.fromarray(np.asarray(height_map, dtype=np.float32), mode='F')
    return np.asarray(
        pil_img.rotate(float(angle_deg), resample=Image.NEAREST),
        dtype=np.float32,
    )


# ===========================================================================
# Background removal + centering
# ===========================================================================

def _crop_and_center_impl(rotated_image, return_centered=False):
    image = np.asarray(rotated_image)
    height, width = image.shape
    ys, xs = np.where(image > 0)
    if len(xs) == 0:
        return (None, None) if return_centered else None

    # Center by bbox midpoint and shift with zero padding (no wrap-around).
    # 1st/99th percentile (vs raw min/max) so a single stray noise pixel
    # cannot inflate the bbox and yank the centering off by tens of rows.
    src_cx = 0.5 * (float(np.percentile(xs, 1)) + float(np.percentile(xs, 99)))
    src_cy = 0.5 * (float(np.percentile(ys, 1)) + float(np.percentile(ys, 99)))
    tgt_cx = (width - 1) * 0.5
    tgt_cy = (height - 1) * 0.5
    shift_x = float(tgt_cx - src_cx)
    shift_y = float(tgt_cy - src_cy)

    M = np.array([[1.0, 0.0, shift_x], [0.0, 1.0, shift_y]], dtype=np.float32)
    img_centered = cv2.warpAffine(
        image,
        M,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0 if np.issubdtype(image.dtype, np.floating) else 0,
    )
    if width < FINAL_CROP_WIDTH:
        pad_left = (FINAL_CROP_WIDTH - width) // 2
        pad_right = FINAL_CROP_WIDTH - width - pad_left
        img_centered = np.pad(
            img_centered,
            ((0, 0), (pad_left, pad_right)),
            mode='constant',
            constant_values=0.0 if np.issubdtype(image.dtype, np.floating) else 0,
        )
        width = img_centered.shape[1]

    x0 = max((width - FINAL_CROP_WIDTH) // 2, 0)
    y0 = max((height - FINAL_CROP_HEIGHT) // 2, 0)
    img_out = img_centered[y0:y0 + FINAL_CROP_HEIGHT, x0:x0 + FINAL_CROP_WIDTH]
    if img_out.shape != (FINAL_CROP_HEIGHT, FINAL_CROP_WIDTH):
        return (None, img_centered) if return_centered else None
    if return_centered:
        return img_out, img_centered
    return img_out


def crop_and_center(rotated_uint8, return_centered=False):
    return _crop_and_center_impl(np.asarray(rotated_uint8, dtype=np.uint8), return_centered=return_centered)


def crop_and_center_float(rotated_float, return_centered=False):
    return _crop_and_center_impl(np.asarray(rotated_float, dtype=np.float32), return_centered=return_centered)


# ===========================================================================
# CSV Parsing
# ===========================================================================

def parse_csv_ground_truth(csv_path, bagfiles_dir, target_bag=None, target_unique_id=None,
                            site='msu'):
    """site='msu' → output bag_name = unique_id (release convention).
    site='unl' → output bag_name = basename of RosBagPath (matches v1 UNL
    dataset.h5 / label.h5 naming).
    """
    bagfile_index = build_bagfile_index(bagfiles_dir)
    bagfile_time_index = build_bagfile_time_index(bagfiles_dir)
    uid_records = defaultdict(list)
    missing_bags = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            unique_id = row.get('UniqueID', '').strip()
            bag = row.get('RosBagPath', '').strip()
            if not unique_id or not bag:
                continue
            # UNL CSV stores RosBagPath as `G1/.../Recording*` — keep only basename.
            bag = basename(bag).replace('.h5', '')
            if target_unique_id and unique_id != target_unique_id:
                continue
            start_time = _float_or_nan(row.get('Start Time', ''))
            end_time = _float_or_nan(row.get('End Time', ''))
            resolved_bag_name, hdf5_path = resolve_bagfile_path(
                bag,
                bagfile_index,
                bagfile_time_index=bagfile_time_index,
                start_time=start_time,
                end_time=end_time,
            )
            if target_bag:
                target_resolved_name, _ = resolve_bagfile_path(target_bag, bagfile_index)
                bag_matches = bag == target_bag or resolved_bag_name == target_bag or resolved_bag_name == target_resolved_name
                if not bag_matches:
                    continue
            if hdf5_path is None:
                missing_bags.append((unique_id, bag))
                continue
            uid_records[unique_id].append({
                'unique_id': unique_id,
                'bag_name': bag,
                'resolved_bag_name': resolved_bag_name,
                'hdf5_path': hdf5_path,
                'fat_rib12': _float_or_nan(row.get('Fat_r', '')),
                'loin_rib12': _float_or_nan(row.get('Loin_r', '')),
                'fat_lumbar1': _float_or_nan(row.get('Fat_l', '')),
                'loin_lumbar1': _float_or_nan(row.get('Loin_l', '')),
                'caliper_score': _float_or_nan(row.get('CalipersScore', '')),
                'start_time': start_time,
                'end_time': end_time,
            })

    if missing_bags:
        preview = ', '.join(f'{uid}->{bag}' for uid, bag in missing_bags[:10])
        raise ValueError(f'CSV rows reference bagfiles that were not found: {preview}')

    processing_records = []
    for unique_id, records in sorted(uid_records.items()):
        gt = {}
        for target in TARGET_NAMES:
            vals = [r[target] for r in records if not np.isnan(r[target])]
            gt[target] = float(np.mean(vals)) if vals else float('nan')

        gt['unique_id'] = unique_id
        if site == 'unl':
            # UNL convention: dataset.h5 / label.h5 use the bag basename
            # (Recording*) as the row's bag_name, not the slaughter UniqueID.
            # Use the first record's resolved_bag_name; for UNL each
            # UniqueID maps to exactly one bag.
            gt['bag_name'] = records[0]['resolved_bag_name']
        else:
            gt['bag_name'] = unique_id
        gt['source_bags'] = sorted({r['bag_name'] for r in records})
        segment_map = {}
        for record in records:
            key = (
                record['resolved_bag_name'],
                record['hdf5_path'],
                record['start_time'],
                record['end_time'],
            )
            segment_map[key] = {
                'source_bag_name': record['bag_name'],
                'resolved_bag_name': record['resolved_bag_name'],
                'hdf5_path': record['hdf5_path'],
                'start_time': record['start_time'],
                'end_time': record['end_time'],
            }
        gt['segments'] = sorted(
            segment_map.values(),
            key=lambda item: (
                item['resolved_bag_name'],
                item['start_time'] if not np.isnan(item['start_time']) else float('-inf'),
                item['end_time'] if not np.isnan(item['end_time']) else float('inf'),
            ),
        )
        processing_records.append(gt)

    return processing_records


# ===========================================================================
# Visualization
# ===========================================================================

def _ordered_box_vertices(box):
    c = np.mean(box, axis=0)
    ang = np.arctan2(box[:, 1] - c[1], box[:, 0] - c[0])
    return box[np.argsort(ang)]


def _front_edge_from_obb(box, angle_deg):
    if box is None or angle_deg is None:
        return None
    b = _ordered_box_vertices(box)
    rad = np.radians(angle_deg)
    axis = np.array([np.cos(rad), np.sin(rad)], dtype=np.float32)
    edges = [
        (b[0], b[1]),
        (b[1], b[2]),
        (b[2], b[3]),
        (b[3], b[0]),
    ]
    lengths = [float(np.linalg.norm(e1 - e0)) for e0, e1 in edges]
    if max(lengths) <= 0:
        return None
    min_len = min(lengths)
    short_idx = [i for i, l in enumerate(lengths) if l <= 1.02 * min_len]
    if not short_idx:
        short_idx = [int(np.argmin(lengths))]
    cand_edges = [edges[i] for i in short_idx]
    midpoints = [0.5 * (e0 + e1) for e0, e1 in cand_edges]
    center = np.mean(box, axis=0)
    projs = [np.dot(m - center, axis) for m in midpoints]
    i = int(np.argmax(projs))
    return cand_edges[i]


def normalize_depth_for_vis(depth_img_raw):
    depth_mm = np.asarray(depth_img_raw, dtype=np.float32)
    valid = np.isfinite(depth_mm) & (depth_mm > 0)
    normalized = np.zeros_like(depth_mm, dtype=np.float32)
    if not np.any(valid):
        return normalized
    valid_values = depth_mm[valid]
    lo = float(np.percentile(valid_values, 2.0))
    hi = float(np.percentile(valid_values, 98.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(valid_values.min())
        hi = float(valid_values.max())
    if hi <= lo:
        return normalized
    normalized[valid] = np.clip((depth_mm[valid] - lo) / (hi - lo), 0.0, 1.0)
    return normalized


def render_depth_gray_rgb(depth_img_raw):
    depth_norm = normalize_depth_for_vis(depth_img_raw)
    depth_u8 = np.round(depth_norm * 255.0).astype(np.uint8)
    return np.stack([depth_u8, depth_u8, depth_u8], axis=-1)


def normalize_linear_for_display(image_raw, vmax=1.0):
    image = np.asarray(image_raw, dtype=np.float32)
    vmax = float(vmax)
    if vmax <= 0.0:
        raise ValueError(f"vmax must be positive, got {vmax}")
    return np.clip(image / vmax, 0.0, 1.0)


def render_jet_rgb(image_01):
    image = np.clip(np.asarray(image_01, dtype=np.float32), 0.0, 1.0)
    image_u8 = np.round(image * 255.0).astype(np.uint8)
    jet_bgr = cv2.applyColorMap(image_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)


def center_pixel_stats(image_2d, scale=1.0):
    image = np.asarray(image_2d)
    cy = int(image.shape[0] // 2)
    cx = int(image.shape[1] // 2)
    value = float(image[cy, cx]) * float(scale)
    return cx, cy, value


def annotate_center_pixel(ax, image_2d, scale=1.0, label_prefix="center",
                          display_image_2d=None, display_scale=1.0):
    cx, cy, value = center_pixel_stats(image_2d, scale=scale)
    display_value = None
    if display_image_2d is not None:
        cx, cy, display_value = center_pixel_stats(display_image_2d, scale=display_scale)
    ax.scatter([cx], [cy], s=28, c='white', edgecolors='black', linewidths=0.8)
    ax.axvline(cx, color='white', linewidth=0.8, alpha=0.7)
    ax.axhline(cy, color='white', linewidth=0.8, alpha=0.7)
    label = f"{label_prefix}=({cx},{cy}) {value:.3f}"
    if display_value is not None and abs(display_value - value) >= 5e-4:
        label = f"{label_prefix}=({cx},{cy}) raw={value:.3f} vis={display_value:.3f}"
    ax.text(
        0.02,
        0.04,
        label,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        color='white',
        fontsize=9,
        bbox=dict(facecolor='black', alpha=0.65, edgecolor='none', pad=0.25),
    )


def overlay_mask_rgb(image_rgb, mask, color=(0, 255, 0), alpha=0.45):
    rendered = image_rgb.copy()
    mask_bool = np.asarray(mask, dtype=bool)
    if not np.any(mask_bool):
        return rendered
    color_array = np.asarray(color, dtype=np.float32)
    blended = rendered[mask_bool].astype(np.float32) * (1.0 - alpha) + color_array[None, :] * alpha
    rendered[mask_bool] = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return rendered


def obb_from_mask(mask):
    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if int(mask_u8.sum()) == 0:
        return None
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if contour is None or len(contour) < 3:
        return None
    hull = cv2.convexHull(contour)
    if hull is None or len(hull) < 3:
        return None
    rect = cv2.minAreaRect(hull)
    return cv2.boxPoints(rect).astype(np.float32)


def draw_oriented_box_with_front_edge(image_rgb, obb_corners, front_edge_xy, box_color=(0, 100, 0), front_edge_color=(255, 255, 0)):
    if obb_corners is None:
        return image_rgb
    rendered = image_rgb.copy()
    corners = np.round(np.asarray(obb_corners, dtype=np.float32).reshape(4, 2)).astype(np.int32)
    cv2.polylines(rendered, [corners], True, box_color, 2, cv2.LINE_AA)
    if front_edge_xy is None:
        return rendered

    edge_points = np.round(np.asarray(front_edge_xy, dtype=np.float32).reshape(2, 2)).astype(np.int32)
    p0 = tuple(int(v) for v in edge_points[0])
    p1 = tuple(int(v) for v in edge_points[1])
    cv2.line(rendered, p0, p1, front_edge_color, 4, cv2.LINE_AA)
    return rendered


def save_single_vis(vis_data, ci, save_vis_dir):
    os.makedirs(save_vis_dir, exist_ok=True)
    (undist_depth, raw_mask_vis, raw_upper_body_vis, raw_mask_obb, raw_front_edge_xy,
     masked_depth, hmap, centered_hmap, cropped_hmap, ag_deg) = vis_data

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    depth_norm = normalize_depth_for_vis(undist_depth)
    masked_depth_norm = normalize_depth_for_vis(masked_depth)
    masked_depth_jet = render_jet_rgb(masked_depth_norm)
    hmap_vis = normalize_linear_for_display(hmap, VIS_H_MAX)
    centered_vis = normalize_linear_for_display(centered_hmap, VIS_H_MAX)
    cropped_vis = normalize_linear_for_display(cropped_hmap, VIS_H_MAX)
    hmap_jet = render_jet_rgb(hmap_vis)
    centered_jet = render_jet_rgb(centered_vis)
    cropped_jet = render_jet_rgb(cropped_vis)
    overlay = overlay_mask_rgb(render_depth_gray_rgb(undist_depth), raw_mask_vis, color=(0, 255, 0), alpha=0.35)
    overlay = overlay_mask_rgb(overlay, raw_upper_body_vis, color=(255, 0, 0), alpha=0.45)
    overlay = draw_oriented_box_with_front_edge(
        overlay,
        raw_mask_obb,
        raw_front_edge_xy,
        box_color=(0, 100, 0),
        front_edge_color=(255, 215, 0),
    )

    ax = axes[0]
    ax.imshow(depth_norm, cmap='gray', vmin=0.0, vmax=1.0)
    ax.set_title(f"1. Depth (frame {ci})")
    ax.axis('off')

    ax = axes[1]
    ax.imshow(overlay)
    ax.set_title("2. Pig + Upper Body + OBB")
    ax.axis('off')

    ax = axes[2]
    ax.imshow(masked_depth_jet)
    ax.set_title("3. Cutout Depth")
    ax.axis('off')

    ax = axes[3]
    ax.imshow(hmap_jet)
    ax.set_title("4. Height Map")
    annotate_center_pixel(ax, hmap, scale=1.0, display_image_2d=hmap_vis)
    ax.axis('off')

    ax = axes[4]
    ax.imshow(centered_jet)
    ax.set_title(f"5. Angle Corrected + Centered ({ag_deg:.1f} deg)")
    annotate_center_pixel(ax, centered_hmap, scale=1.0, display_image_2d=centered_vis)
    ax.axis('off')

    ax = axes[5]
    ax.imshow(cropped_jet)
    ax.set_title("6. Final 96x224 Crop")
    annotate_center_pixel(ax, cropped_hmap, scale=1.0, display_image_2d=cropped_vis)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(join(save_vis_dir, f'pipeline_frame_{ci:05d}.png'), dpi=150)
    plt.close()


def save_random_triplet_visualization(final_crops, save_vis_dir, seed=0):
    if not final_crops:
        return
    os.makedirs(save_vis_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    pick_n = min(3, len(final_crops))
    chosen = rng.choice(len(final_crops), size=pick_n, replace=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes = np.atleast_1d(axes)
    for ax_idx, ax in enumerate(axes):
        if ax_idx < pick_n:
            ci_val, crop_img = final_crops[int(chosen[ax_idx])]
            ax.imshow(crop_img.astype(np.float32) / 255.0 * H_MAX, cmap='jet', vmin=0, vmax=VIS_H_MAX)
            ax.set_title(f"Frame {ci_val}")
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(join(save_vis_dir, 'random_triplet_heightmaps.png'), dpi=150)
    plt.close()


# ===========================================================================
# Process a single HDF5 bag
# ===========================================================================

def process_bag(hdf5_path, camera_params, ground_plane_params, rot_mat_al,
                device, maskdino_model, maskdino_augment,
                max_frames=20, save_vis_dir=None,
                start_time=float('nan'), end_time=float('nan'),
                output_bag_name=None,
                target_depth_idx=None,
                maskdino_batch_size=8,
                maskdino_score_threshold=0.2,
                maskdino_pig_class_index=0,
                maskdino_upper_body_class_index=1,
                maskdino_encoding='depth_valid_gradient',
                mask_dilate_px=0,
                central_window_border=None,
                central_window_border_fallback=0,
                unet_model=None,
                unet_score_threshold=0.5,
                unet_min_pig_px=1000,
                maskdino_min_pig_px=0):
    resolved_bag_name = os.path.basename(hdf5_path).replace('.h5', '')
    bag_name = output_bag_name or resolved_bag_name
    a, b, c = [float(v) for v in ground_plane_params]
    need_vis = save_vis_dir is not None
    print(f"  Using pig_upper_body-driven horizontal alignment for {resolved_bag_name}")

    # Step 1: Depth-frame detection. UNet path if a UNet model is provided;
    # otherwise MaskDINO. The MaskDINO path retries with a looser central-
    # window border if the primary border yields no detections.
    def _run_detect(border):
        if unet_model is not None:
            return detect_pigs_unet_depth(
                hdf5_path,
                unet_model,
                hdf5_device=device,
                start_ns=start_time,
                end_ns=end_time,
                camera_params=camera_params,
                batch_size=int(maskdino_batch_size),
                score_threshold=float(unet_score_threshold),
                min_pig_px=int(unet_min_pig_px),
                central_window_border=border,
            )
        return detect_pigs_maskdino_depth(
            hdf5_path,
            maskdino_model,
            maskdino_augment,
            maskdino_device=device,
            start_ns=start_time,
            end_ns=end_time,
            batch_size=maskdino_batch_size,
            score_threshold=maskdino_score_threshold,
            pig_class_index=maskdino_pig_class_index,
            upper_body_class_index=maskdino_upper_body_class_index,
            encoding=maskdino_encoding,
            camera_params=camera_params,
            central_window_border=border,
            min_pig_px=int(maskdino_min_pig_px),
        )

    detections = _run_detect(central_window_border)
    if not detections and central_window_border_fallback is not None and \
            central_window_border_fallback != central_window_border:
        print(f"  No detections at border={central_window_border}; "
              f"retrying with fallback border={central_window_border_fallback}")
        detections = _run_detect(int(central_window_border_fallback))

    if not detections:
        print(f"  No valid detections for {bag_name}")
        return []

    if target_depth_idx is not None:
        detections = [det for det in detections if int(det[1]) == int(target_depth_idx)]
        if not detections:
            print(f"  No valid detections for {bag_name} at depth frame {int(target_depth_idx)}")
            return []

    # Limit frames
    if len(detections) > max_frames:
        step = len(detections) / max_frames
        detections = [detections[int(i * step)] for i in range(max_frames)]

    candidate_entries = []

    with h5py.File(hdf5_path, 'r') as f:
        depth_w = int(f.attrs.get('depth_width', 640))
        depth_h = int(f.attrs.get('depth_height', 576))
        center_x = depth_w / 2
        center_y = depth_h / 2

        for ci, di, depth_mask_raw, upper_body_mask_raw, bbox_xywhn in detections:
            depth_img = f['depth/images'][di]

            # Optional mask dilation before BEV (compensates for tight GT
            # polygons that miss the pig's flanks/legs; needed for v2
            # which precisely matches the tight supervision).
            if mask_dilate_px and mask_dilate_px > 0:
                k = max(1, 2 * int(mask_dilate_px) + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                depth_mask_raw = cv2.dilate(depth_mask_raw.astype(np.uint8), kernel, iterations=1).astype(bool)
                if upper_body_mask_raw is not None:
                    upper_body_mask_raw = cv2.dilate(upper_body_mask_raw.astype(np.uint8), kernel, iterations=1).astype(bool)

            # Step 2: Generate height map using depth mask. When MaskDINO
            # ran on undistorted input (encoding 'raw_meters_1ch') or the
            # UNet path produced the mask, masks are already in undistorted
            # coords; tell generate_height_map not to undistort them again.
            masks_undistorted = (maskdino_encoding == 'raw_meters_1ch') or (unet_model is not None)
            hmap, depth_bbox, ground_bev_box, depth_ground_box, undist_depth, depth_mask, yolo_depth_poly, upper_body_mask, pig_bev_mask, upper_body_bev_mask = generate_height_map(
                depth_img, None, camera_params,
                a, b, c, rot_mat_al, bbox_xywhn,
                depth_mask_raw=depth_mask_raw,
                upper_body_mask_raw=upper_body_mask_raw,
                masks_already_undistorted=masks_undistorted)

            if hmap is None:
                continue

            _, ag_deg = rotate_height_map(hmap, pig_bev_mask=pig_bev_mask, upper_body_bev_mask=upper_body_bev_mask)
            if ag_deg is None:
                continue
            rotated_hmap = render_height_map_rotation_float(hmap, ag_deg)
            crop_result = crop_and_center_float(rotated_hmap, return_centered=need_vis)
            if need_vis:
                cropped_hmap, centered_hmap = crop_result
            else:
                cropped_hmap = crop_result
                centered_hmap = None
            if cropped_hmap is None:
                continue

            hmap_final = np.asarray(cropped_hmap, dtype=np.float32)
            depth_mask_area = int(np.count_nonzero(depth_mask))
            height_map_area = int(np.count_nonzero(cropped_hmap > 0.0))

            entry = {
                'hmap_final': hmap_final,
                'bag_name': bag_name,
                'ci': ci,
                'depth_mask_area': depth_mask_area,
                'height_map_area': height_map_area,
            }
            if need_vis:
                bx, by, _, _ = bbox_xywhn
                pig_cx = bx * depth_w
                pig_cy = by * depth_h
                dist = np.sqrt((pig_cx - center_x)**2 + (pig_cy - center_y)**2)
                cropped = np.round(normalize_linear_for_display(cropped_hmap, H_MAX) * 255.0).astype(np.uint8)
                mask_vis = depth_mask.astype(bool, copy=False)
                upper_body_vis = upper_body_mask.astype(bool, copy=False)
                raw_mask_obb = obb_from_mask(mask_vis)
                _, raw_front_edge_xy, _ = front_edge_from_masks(mask_vis, upper_body_vis)
                masked_depth = undist_depth.astype(np.float32).copy()
                masked_depth[~depth_mask] = 0
                entry.update({
                    'dist': float(dist),
                    'cropped': cropped,
                    'vis_data': (
                        undist_depth,
                        mask_vis,
                        upper_body_vis,
                        raw_mask_obb,
                        raw_front_edge_xy,
                        masked_depth,
                        hmap,
                        centered_hmap,
                        cropped_hmap,
                        ag_deg,
                    ),
                })
            candidate_entries.append(entry)

    candidate_entries = filter_area_outlier_entries(candidate_entries, bag_name)
    if not candidate_entries:
        print(f"  No usable samples after area-outlier filtering for {bag_name}")
        return []

    samples = [(entry['hmap_final'], entry['bag_name'], entry['ci']) for entry in candidate_entries]

    if need_vis:
        best_entry = min(candidate_entries, key=lambda entry: entry['dist'])
        last_entry = candidate_entries[-1]
        final_crops = [(entry['ci'], entry['cropped']) for entry in candidate_entries]
        save_single_vis(last_entry['vis_data'], last_entry['ci'], save_vis_dir)
        print(f"  Saved last frame visualization: {last_entry['ci']}")
        save_single_vis(best_entry['vis_data'], best_entry['ci'], save_vis_dir)

        # Final crops grid
        if final_crops:
            n_crops = len(final_crops)
            rows, cols = 4, 5
            fig, axes = plt.subplots(rows, cols, figsize=(15, 12))
            axes = axes.flatten()
            for i in range(rows * cols):
                ax = axes[i]
                if i < n_crops:
                    ci_val, crop_img = final_crops[i]
                    ax.imshow(crop_img.astype(np.float32) / 255.0 * H_MAX, cmap='jet', vmin=0, vmax=VIS_H_MAX)
                    ax.set_title(f"Frame {ci_val}")
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(join(save_vis_dir, 'final_crops_grid.png'), dpi=150)
            plt.close()
            save_random_triplet_visualization(final_crops, save_vis_dir)

    return samples


# ===========================================================================
# Save / Report / Sample Vis (same as preprocess_new.py)
# ===========================================================================

def save_dataset(all_samples, output_path):
    os.makedirs(dirname(output_path), exist_ok=True)
    n = len(all_samples)
    print(f"\nSaving {n} samples to {output_path}")

    height_maps = np.zeros((n, FINAL_CROP_HEIGHT, FINAL_CROP_WIDTH), dtype=np.float32)
    bag_names = []
    source_names = []

    for i, sample in enumerate(all_samples):
        hmap, bn, ci = sample[0], sample[1], sample[2]
        height_maps[i] = hmap
        bag_names.append(bn)
        source_names.append(f"{bn}_frame{ci}")

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('height_maps', data=height_maps, compression='gzip', compression_opts=4)
        dt_str = h5py.string_dtype()
        f.create_dataset('bag_names', data=[bn.encode('utf-8') for bn in bag_names])
        f.create_dataset('source_names', data=[sn.encode('utf-8') for sn in source_names], dtype=dt_str)

    print(f"  Dataset saved: {output_path}")
    print(f"  Shape: height_maps={height_maps.shape}")


def select_shard_records(processing_records, shard_index=0, num_shards=1):
    if int(num_shards) <= 1:
        return list(processing_records)
    shard_index = int(shard_index)
    num_shards = int(num_shards)
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Invalid shard selection: shard_index={shard_index}, num_shards={num_shards}")
    return [record for idx, record in enumerate(processing_records) if (idx % num_shards) == shard_index]


def print_report(all_samples):
    n = len(all_samples)
    print(f"\n{'='*60}")
    print(f"PREPROCESSING REPORT (MASKDINO DEPTH)")
    print(f"{'='*60}")
    print(f"Total samples: {n}")

    from collections import Counter
    src_counts = Counter(s[1] for s in all_samples) # bag_name is index 1
    print(f"\nBy source:")
    for src, cnt in sorted(src_counts.items()):
        print(f"  {src}: {cnt}")
    print(f"{'='*60}")


def save_sample_visualizations(all_samples, vis_dir, n_samples=16):
    os.makedirs(vis_dir, exist_ok=True)
    n = min(n_samples, len(all_samples))
    if n == 0:
        return
    indices = np.linspace(0, len(all_samples) - 1, n, dtype=int)

    fig, axes = plt.subplots(4, 4, figsize=(16, 8))
    for ax_idx in range(4 * 4):
        ax = axes[ax_idx // 4, ax_idx % 4]
        if ax_idx < n:
            sample = all_samples[indices[ax_idx]]
            hmap, bn, ci = sample[0], sample[1], sample[2]
            ax.imshow(hmap, cmap='jet', vmin=0, vmax=VIS_H_MAX)
            ax.set_title(f"{bn}\nframe={ci}", fontsize=7)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(join(vis_dir, 'sample_height_maps.png'), dpi=150)
    plt.close()
    print(f"  Saved visualization: {join(vis_dir, 'sample_height_maps.png')}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Depth-MaskDINO-based preprocessing')
    parser.add_argument('--max_frames_per_bag', type=int, default=20)
    parser.add_argument('--target_bag', type=str, default=None,
                        help='Process only this specific bag')
    parser.add_argument('--target_unique_id', type=str, default=None,
                        help='Process only this specific UniqueID')
    parser.add_argument('--target_depth_idx', type=int, default=None,
                        help='Process only this depth frame index within the selected bag/segment')
    parser.add_argument('--csv_path', type=str, default=CSV_PATH)
    parser.add_argument('--bagfiles_dir', type=str, default=BAGFILES_DIR)
    parser.add_argument('--ground_plane_json', type=str, default=DEFAULT_DATE_GROUND_PLANE_JSON)
    parser.add_argument('--maskdino_config', type=str, default=DEFAULT_MASKDINO_CONFIG)
    parser.add_argument('--maskdino_weights', type=str, default=DEFAULT_MASKDINO_WEIGHTS)
    parser.add_argument('--maskdino_batch_size', type=int, default=8)
    parser.add_argument('--maskdino_score_threshold', type=float, default=0.2)
    parser.add_argument('--maskdino_pig_class_index', type=int, default=0)
    parser.add_argument('--maskdino_upper_body_class_index', type=int, default=1)
    parser.add_argument('--maskdino_encoding', type=str, default='depth_valid_gradient',
                        choices=['depth_valid_gradient', 'depth_repeat', 'raw_meters_1ch'])
    parser.add_argument('--target_unique_ids_file', type=str, default=None,
                        help='Path to a file with one UniqueID per line; processes all listed.')
    parser.add_argument('--mask_dilate_px', type=int, default=0,
                        help='Morphologically dilate pig + upper_body masks by N pixels before BEV. '
                             'Compensates for GT polygons that miss the pig flanks/legs (set ~5-8 '
                             'for v2 which matches the tight supervision).')
    parser.add_argument('--torch_compile', action='store_true',
                        help='Wrap MaskDINO with torch.compile(reduce-overhead). '
                             'Benchmarked ~2x faster forward on H200; first batch pays '
                             'a ~30 s compile cost.')
    parser.add_argument('--site', type=str, default='msu', choices=['msu', 'unl'],
                        help="Site convention for output bag_name. 'msu' uses unique_id "
                             "(release default). 'unl' uses bag basename to match v1 UNL naming.")
    parser.add_argument('--output_path', type=str, default=DATASET_H5_PATH)
    parser.add_argument('--vis_dir', type=str, default=VIS_DIR)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard_index', type=int, default=0)
    parser.add_argument('--save_vis', '--save-vis', dest='save_vis',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--depth_undistort_fx_scale', type=float, default=None,
                        help='Override DEPTH_UNDISTORT_FX_SCALE for the depth/mask undistortion. '
                             'Use 1.0 for v2 MaskDINO (raw_meters_1ch) so the mask and depth '
                             'share the same K-undistorted grid; the legacy v1 build used 0.85.')
    parser.add_argument('--mask_center_border_px', type=int, default=None,
                        help='Reject frames whose pig mask comes within this many pixels of '
                             'any image edge. Defaults to the module constant '
                             f'({MASK_CENTER_BORDER_PX}). For v2 use 15 to filter out partial '
                             'pigs (matches the original v1 setting before edge-only camera '
                             'setups forced it to 0).')
    parser.add_argument('--mask_center_border_px_fallback', type=int, default=0,
                        help='If a bag has zero detections at --mask_center_border_px, retry '
                             'with this looser border. 0 admits any non-empty mask. Set to the '
                             'same value as --mask_center_border_px to disable the fallback.')
    parser.add_argument('--unet_weights', type=str, default=None,
                        help='Path to a trained DepthUNet checkpoint (model_best.pt). When set, '
                             'Stage 1 uses the UNet instead of MaskDINO. The UNet was trained '
                             'on the same on-the-fly K-undistortion that this script uses for '
                             'depth, so masks come out already aligned.')
    parser.add_argument('--unet_score_threshold', type=float, default=0.5,
                        help='Sigmoid threshold for UNet pig/upper_body masks. Default 0.5.')
    parser.add_argument('--unet_min_pig_px', type=int, default=1000,
                        help='Reject frames where the UNet pig mask has fewer than N pixels '
                             'above threshold. Default 1000.')
    parser.add_argument('--maskdino_min_pig_px', type=int, default=0,
                        help='Reject frames where the MaskDINO pig mask has fewer than N pixels. '
                             'Default 0 (disabled). Set to e.g. 20000 to filter out small false '
                             'positives like handlers/humans (real pigs span 30k+ pixels at fx=0.85).')
    args = parser.parse_args()

    if args.depth_undistort_fx_scale is not None:
        global DEPTH_UNDISTORT_FX_SCALE
        DEPTH_UNDISTORT_FX_SCALE = float(args.depth_undistort_fx_scale)
        _DEPTH_UNDISTORT_CACHE.clear()
        _XY_TABLE_CACHE.clear()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Device: {device}")
    processing_records = parse_csv_ground_truth(
        args.csv_path,
        args.bagfiles_dir,
        target_bag=args.target_bag,
        target_unique_id=args.target_unique_id,
        site=args.site,
    )
    if args.target_unique_ids_file:
        with open(args.target_unique_ids_file) as _f:
            wanted = {line.strip() for line in _f if line.strip()}
        processing_records = [r for r in processing_records if r['unique_id'] in wanted]
        print(f"  filtered to {len(processing_records)} UniqueIDs from {args.target_unique_ids_file}")
    processing_records = select_shard_records(
        processing_records,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    if args.target_bag:
        print(f"  Target bag: {args.target_bag}")
    if args.target_unique_id:
        print(f"  Target UniqueID: {args.target_unique_id}")
    if args.target_depth_idx is not None:
        print(f"  Target depth frame: {args.target_depth_idx}")
    if int(args.num_shards) > 1:
        print(f"  Shard: {args.shard_index + 1}/{args.num_shards}")
    print(f"  {len(processing_records)} UniqueIDs to process")
    print(f"  {len({r['bag_name'] for r in processing_records})} output bag IDs selected")
    print(f"  Ground plane cache: {args.ground_plane_json}")
    print(f"  Depth undistort fx/fy scale: {DEPTH_UNDISTORT_FX_SCALE:.2f}")
    print(f"  Output path: {args.output_path}")
    if args.save_vis:
        print(f"  Visualization dir: {args.vis_dir}")
    else:
        print("  Visualization: disabled")

    # Process
    all_samples = []
    date_ground_planes = load_date_ground_planes(args.ground_plane_json)

    unet_model = None
    if args.unet_weights is not None:
        if not isfile(args.unet_weights):
            raise FileNotFoundError(f"UNet weights not found: {args.unet_weights}")
        print("\n=== Initializing UNet (Depth) ===")
        from unet_depth import load_unet
        unet_model, unet_ckpt = load_unet(args.unet_weights, device=str(device))
        print(f"  UNet loaded on {device}  ({sum(p.numel() for p in unet_model.parameters())/1e6:.2f}M params)")
        print(f"  encoder={unet_ckpt.get('encoder')}  epoch={unet_ckpt.get('epoch')}  "
              f"val_loss={unet_ckpt.get('val_loss')}")
        # Skip MaskDINO setup entirely when running the UNet stage.
        maskdino_model = None
        maskdino_augment = None
        model_device = device
        maskdino_encoding = args.maskdino_encoding
    else:
        if not isfile(args.maskdino_config):
            raise FileNotFoundError(f"MaskDINO config not found: {args.maskdino_config}")
        if not isfile(args.maskdino_weights):
            raise FileNotFoundError(f"MaskDINO weights not found: {args.maskdino_weights}")
        print("\n=== Initializing MaskDINO (Depth) ===")
        maskdino_model, maskdino_augment, model_device, maskdino_encoding = init_maskdino_depth_model(
            config_file=args.maskdino_config,
            weights=args.maskdino_weights,
            score_threshold=args.maskdino_score_threshold,
            encoding=args.maskdino_encoding,
            device_override=str(device),
            torch_compile=args.torch_compile,
        )
        print(f"  MaskDINO loaded on {model_device}.")

    failed_bags = []
    total_new = 0
    processed_bag_names = set()

    import time
    for record in tqdm(processing_records, desc="Processing UniqueIDs"):
        unique_id = record['unique_id']
        bag_name = record['bag_name']
        unique_samples = []
        try:
            for seg_idx, segment in enumerate(record['segments']):
                resolved_bag_name = segment['resolved_bag_name']
                hdf5_path = segment['hdf5_path']
                camera_params = get_camera_params_for_bag(resolved_bag_name,
                                                           hdf5_path=hdf5_path)
                if camera_params is None:
                    raise RuntimeError(f"Camera parameters not found for {resolved_bag_name}")

                ground_plane_params, date_ground_planes, ground_plane_record = resolve_ground_plane_for_bag(
                    resolved_bag_name,
                    bagfiles_dir=args.bagfiles_dir,
                    ground_plane_json=args.ground_plane_json,
                    date_ground_planes=date_ground_planes,
                )
                rot_mat_al = compute_rotation_matrix(*ground_plane_params)
                print(
                    f"  Ground plane for {resolved_bag_name}: "
                    f"a={ground_plane_record['a']:.6f}, "
                    f"b={ground_plane_record['b']:.6f}, "
                    f"c={ground_plane_record['c']:.6f}"
                )

                samples = process_bag(
                    hdf5_path, camera_params, ground_plane_params, rot_mat_al,
                    model_device, maskdino_model, maskdino_augment,
                    max_frames=args.max_frames_per_bag,
                    save_vis_dir=(
                        join(args.vis_dir, f"{unique_id}_segment{seg_idx:02d}_{resolved_bag_name}_pipeline")
                        if args.save_vis else None
                    ),
                    start_time=segment['start_time'],
                    end_time=segment['end_time'],
                    output_bag_name=bag_name,
                    target_depth_idx=args.target_depth_idx,
                    maskdino_batch_size=args.maskdino_batch_size,
                    maskdino_score_threshold=args.maskdino_score_threshold,
                    maskdino_pig_class_index=args.maskdino_pig_class_index,
                    maskdino_upper_body_class_index=args.maskdino_upper_body_class_index,
                    maskdino_encoding=maskdino_encoding,
                    mask_dilate_px=args.mask_dilate_px,
                    central_window_border=args.mask_center_border_px,
                    central_window_border_fallback=args.mask_center_border_px_fallback,
                    unet_model=unet_model,
                    unet_score_threshold=args.unet_score_threshold,
                    unet_min_pig_px=args.unet_min_pig_px,
                    maskdino_min_pig_px=args.maskdino_min_pig_px)
                unique_samples.extend(samples)

            for hmap, bn, ci in unique_samples:
                all_samples.append((hmap, bn, ci))
                total_new += 1
                processed_bag_names.add(bn)
            if not unique_samples:
                failed_bags.append((unique_id, f"No samples produced for {bag_name}"))

        except Exception as e:
            import traceback
            traceback.print_exc()
            failed_bags.append((unique_id, str(e)))

    print(f"\n  Total samples: {total_new}")
    if failed_bags:
        print(f"  Failed bags ({len(failed_bags)}):")
        for bn, reason in failed_bags[:10]:
            print(f"    {bn}: {reason}")

    expected_unique_ids = len(processing_records)
    expected_bags = len({r['bag_name'] for r in processing_records})
    if expected_unique_ids != expected_bags or len(processed_bag_names) != expected_bags:
        print(
            "  Warning: count mismatch after preprocessing: "
            f"unique_ids={expected_unique_ids}, "
            f"selected_bags={expected_bags}, "
            f"processed_bags={len(processed_bag_names)}"
        )

    save_dataset(all_samples, args.output_path)
    print_report(all_samples)
    if args.save_vis:
        save_sample_visualizations(all_samples, args.vis_dir)

    print("\nPreprocessing (MaskDINO depth) complete!")

if __name__ == '__main__':
    main()
