#!/usr/bin/env python3

"""Date-level MSU ground-plane estimation and caching."""

import configparser
import copy
import json
import os
import re
from datetime import datetime, timezone
from os.path import abspath, basename, dirname, isdir, isfile, join

import cv2
import h5py
import numpy as np

SCRIPT_DIR = dirname(abspath(__file__))
DATA_ROOT = dirname(SCRIPT_DIR)
CAMERA_PARAMS_DIR = join(SCRIPT_DIR, 'camera_params')
DEFAULT_CAMERA_INI = join(CAMERA_PARAMS_DIR, 'rosbags_20250212_CameraParams.ini')
DEFAULT_BAGFILES_DIR = join(DATA_ROOT, 'bagfiles')
DEFAULT_BAGFILES_DIR_FALLBACK = '/mnt/scratch/basharmk/data/body_condition/msu/bagfiles'
DEFAULT_DATE_GROUND_PLANE_JSON = join(DATA_ROOT, 'verification_fix', 'msu_date_ground_planes.json')

SESSION_CAMERA_PARAM_FALLBACKS = {
    '20250213': '20250212',
}

GROUND_A = 0.024
GROUND_B = 0.088
GROUND_C = 2.353
GROUND_MAX_DEPTH_M = 3.5
GROUND_BORDER_X_PX = 80
GROUND_BORDER_Y_PX = 60
GROUND_FAR_PERCENTILE = 60.0
GROUND_MAX_BAGS_PER_DATE = 4
GROUND_FRAMES_PER_BAG = 6
GROUND_FRAME_CANDIDATES_PER_BAG = 18
GROUND_EARLY_FRAMES_PER_BAG = 4
GROUND_MAX_POINTS_PER_FRAME = 4000
GROUND_MAX_TOTAL_POINTS = 120000
GROUND_RANSAC_ITERS = 400
GROUND_RANSAC_THRESH_M = 0.015
GROUND_TOP_K_CANDIDATES = 8
GROUND_EDGE_REJECT_THRESH_M = 0.06
GROUND_CENTER_BAND_HALF_WIDTH_PX = 8


BAG_DATE_RE = re.compile(r'(20\d{6})')


def resolve_bagfiles_dir(bagfiles_dir=None):
    for candidate in [bagfiles_dir, DEFAULT_BAGFILES_DIR, DEFAULT_BAGFILES_DIR_FALLBACK]:
        if candidate and isdir(candidate):
            return candidate
    return bagfiles_dir or DEFAULT_BAGFILES_DIR


def extract_bag_date(bag_name_or_path):
    name = basename(str(bag_name_or_path))
    if name.endswith('.h5'):
        name = name[:-3]
    match = BAG_DATE_RE.search(name)
    return match.group(1) if match else None


def load_camera_parameters(ini_file):
    if not isfile(ini_file):
        raise FileNotFoundError(f'Camera .ini not found: {ini_file}')

    config = configparser.ConfigParser()
    config.read(ini_file)
    params = {}
    for prefix in ['Color', 'Depth']:
        params.update({
            f'{prefix}Intrinsic': {
                'fx': config.getfloat(f'{prefix}Intrinsic', 'fx'),
                'fy': config.getfloat(f'{prefix}Intrinsic', 'fy'),
                'cx': config.getfloat(f'{prefix}Intrinsic', 'cx'),
                'cy': config.getfloat(f'{prefix}Intrinsic', 'cy'),
                'width': config.getint(f'{prefix}Intrinsic', 'width'),
                'height': config.getint(f'{prefix}Intrinsic', 'height'),
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


def get_camera_params_for_bag(bag_name, camera_params_dir=CAMERA_PARAMS_DIR):
    name = re.sub(r'^rosbag2_', '', str(bag_name))
    flat = name.replace('_', '').replace('-', '')
    match = BAG_DATE_RE.search(flat)
    date_str = match.group(1) if match else extract_bag_date(bag_name)

    candidate_dates = []
    if date_str:
        candidate_dates.append(date_str)
        fallback_date = SESSION_CAMERA_PARAM_FALLBACKS.get(date_str)
        if fallback_date and fallback_date not in candidate_dates:
            candidate_dates.append(fallback_date)

    for candidate_date in candidate_dates:
        session_ini = join(camera_params_dir, f'rosbags_{candidate_date}_CameraParams.ini')
        if isfile(session_ini):
            return load_camera_parameters(session_ini)

    if isfile(DEFAULT_CAMERA_INI):
        return load_camera_parameters(DEFAULT_CAMERA_INI)
    return None


def distortion_array_to_dict(distortion_arr):
    vals = list(distortion_arr) + [0.0] * 8
    return {
        'k1': float(vals[0]),
        'k2': float(vals[1]),
        'p1': float(vals[2]),
        'p2': float(vals[3]),
        'k3': float(vals[4]),
        'k4': float(vals[5]),
        'k5': float(vals[6]),
        'k6': float(vals[7]),
    }


def load_depth_camera_info_from_hdf5(hdf5_path):
    with h5py.File(hdf5_path, 'r') as handle:
        if 'depth/camera_info' not in handle:
            return None, None
        grp = handle['depth/camera_info']
        K = grp['K'][:]
        D = grp['D'][:]
        height = int(grp.attrs.get('height', handle['depth/images'].shape[1] if 'depth/images' in handle else 0))
        width = int(grp.attrs.get('width', handle['depth/images'].shape[2] if 'depth/images' in handle else 0))
    intrinsic = {
        'fx': float(K[0, 0]),
        'fy': float(K[1, 1]),
        'cx': float(K[0, 2]),
        'cy': float(K[1, 2]),
        'width': width,
        'height': height,
    }
    return intrinsic, distortion_array_to_dict(D)


def proj2plane(x_pts, y_pts, z_pts, a, b, c):
    k = (-c + z_pts - a * x_pts - b * y_pts) / (a ** 2 + b ** 2 + 1)
    x_pts1 = x_pts + k * a
    y_pts1 = y_pts + k * b
    z_pts1 = z_pts - k
    return x_pts1, y_pts1, z_pts1


def compute_rotation_matrix(a, b, c):
    x_pts = np.array([0, 1], dtype=np.float64)
    y_pts = np.array([0, 0], dtype=np.float64)
    z_pts = np.array([0, 0], dtype=np.float64)
    xs_proj, ys_proj, zs_proj = proj2plane(x_pts, y_pts, z_pts, a, b, c)
    vect_x_bev = np.array([
        xs_proj[1] - xs_proj[0],
        ys_proj[1] - ys_proj[0],
        zs_proj[1] - zs_proj[0],
    ], dtype=np.float64)
    vect_x_bev = vect_x_bev / np.linalg.norm(vect_x_bev)
    vect_z_bev = np.array([a, b, -1.0], dtype=np.float64)
    vect_z_bev = vect_z_bev / np.linalg.norm(vect_z_bev)
    vect_y_bev = np.cross(vect_x_bev, vect_z_bev)
    return np.stack([vect_x_bev, vect_y_bev, vect_z_bev], axis=1).T.astype(np.float32)


def undistort_depth_image2(image, intrinsic, distortion):
    camera_matrix = np.array([
        [intrinsic['fx'], 0, intrinsic['cx']],
        [0, intrinsic['fy'], intrinsic['cy']],
        [0, 0, 1],
    ], dtype=np.float32)
    dist_coeffs = np.array([
        distortion['k1'], distortion['k2'], distortion['p1'], distortion['p2'],
        distortion['k3'], distortion.get('k4', 0.0), distortion.get('k5', 0.0), distortion.get('k6', 0.0),
    ], dtype=np.float32)
    height, width = image.shape[:2]
    height2, width2 = height + 50, width + 50
    camera_matrix2, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width2, height2))
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, camera_matrix2, (width2, height2), cv2.CV_32FC1)
    undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_NEAREST)
    intrinsic2 = copy.deepcopy(intrinsic)
    intrinsic2['fx'] = float(camera_matrix2[0, 0])
    intrinsic2['fy'] = float(camera_matrix2[1, 1])
    intrinsic2['cx'] = float(camera_matrix2[0, 2])
    intrinsic2['cy'] = float(camera_matrix2[1, 2])
    intrinsic2['width'] = int(width2)
    intrinsic2['height'] = int(height2)
    return undistorted_image, intrinsic2


def create_xy_table(depth_intrinsic):
    width = int(depth_intrinsic['width'])
    height = int(depth_intrinsic['height'])
    fx = float(depth_intrinsic['fx'])
    fy = float(depth_intrinsic['fy'])
    cx = float(depth_intrinsic['cx'])
    cy = float(depth_intrinsic['cy'])
    x = np.arange(width, dtype=np.float32) - cx
    y = np.arange(height, dtype=np.float32) - cy
    xx, yy = np.meshgrid(x / fx, y / fy)
    return np.dstack((xx, yy)).astype(np.float32)


def solve_plane_least_squares(points_xyz):
    xyz = np.asarray(points_xyz, dtype=np.float64)
    if len(xyz) < 3:
        raise ValueError('Need at least 3 points to solve plane')
    A = np.c_[xyz[:, 0], xyz[:, 1], np.ones(len(xyz))]
    coef, _, _, _ = np.linalg.lstsq(A, xyz[:, 2], rcond=None)
    return coef.astype(np.float64, copy=False)


def plane_residuals(points_xyz, coef):
    xyz = np.asarray(points_xyz, dtype=np.float64)
    coef = np.asarray(coef, dtype=np.float64)
    return xyz[:, 2] - (xyz[:, 0] * coef[0] + xyz[:, 1] * coef[1] + coef[2])


def normalize_plane_normal(coef):
    a, b, _ = [float(v) for v in coef]
    normal = np.array([a, b, -1.0], dtype=np.float64)
    norm = np.linalg.norm(normal)
    return normal if norm < 1e-12 else (normal / norm)


def plane_candidates_similar(coef_a, coef_b, max_angle_deg=1.5, max_c_diff=0.05):
    n_a = normalize_plane_normal(coef_a)
    n_b = normalize_plane_normal(coef_b)
    dot = float(np.clip(np.abs(np.dot(n_a, n_b)), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(dot)))
    c_diff = abs(float(coef_a[2]) - float(coef_b[2]))
    return angle_deg <= float(max_angle_deg) and c_diff <= float(max_c_diff)


def load_date_ground_planes(json_path=DEFAULT_DATE_GROUND_PLANE_JSON):
    if not json_path or not isfile(json_path):
        return {}
    with open(json_path, 'r', encoding='utf-8') as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def save_date_ground_planes(date_ground_planes, json_path=DEFAULT_DATE_GROUND_PLANE_JSON):
    if not json_path:
        return
    parent = dirname(json_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(date_ground_planes, handle, indent=2, sort_keys=True)


def plane_tuple_from_record(record):
    if isinstance(record, dict):
        return float(record['a']), float(record['b']), float(record['c'])
    if isinstance(record, (list, tuple)) and len(record) >= 3:
        return float(record[0]), float(record[1]), float(record[2])
    raise ValueError(f'Invalid ground-plane record: {record!r}')


def default_ground_plane_record():
    return {
        'a': float(GROUND_A),
        'b': float(GROUND_B),
        'c': float(GROUND_C),
        'source': 'legacy_msu_fallback',
    }


def compute_depth_smooth_mask(depth_m, edge_thresh_m=GROUND_EDGE_REJECT_THRESH_M):
    valid = depth_m > 0
    smooth = valid.copy()
    if not np.any(valid):
        return smooth

    dx = np.abs(depth_m[:, 1:] - depth_m[:, :-1])
    dy = np.abs(depth_m[1:, :] - depth_m[:-1, :])
    valid_dx = valid[:, 1:] & valid[:, :-1]
    valid_dy = valid[1:, :] & valid[:-1, :]

    edge = np.zeros(depth_m.shape, dtype=bool)
    bad_dx = valid_dx & (dx > float(edge_thresh_m))
    bad_dy = valid_dy & (dy > float(edge_thresh_m))
    edge[:, 1:] |= bad_dx
    edge[:, :-1] |= bad_dx
    edge[1:, :] |= bad_dy
    edge[:-1, :] |= bad_dy
    return smooth & (~edge)


def choose_ground_frame_indices(num_frames, desired_count):
    if num_frames <= 0 or desired_count <= 0:
        return np.empty((0,), dtype=int)

    desired_count = int(min(desired_count, num_frames))
    early_count = int(min(GROUND_EARLY_FRAMES_PER_BAG, num_frames, desired_count))
    early = np.arange(early_count, dtype=int)
    if desired_count == early_count:
        return early

    remainder = desired_count - early_count
    spread = np.linspace(early_count, num_frames - 1, remainder, dtype=int)
    return np.unique(np.concatenate([early, spread])).astype(int, copy=False)


def sample_ground_plane_points_from_depth(depth_img_raw, depth_intrinsic, depth_distortion,
                                          rng, max_depth_m=GROUND_MAX_DEPTH_M,
                                          border_x_px=GROUND_BORDER_X_PX,
                                          border_y_px=GROUND_BORDER_Y_PX,
                                          far_percentile=GROUND_FAR_PERCENTILE,
                                          max_points=GROUND_MAX_POINTS_PER_FRAME):
    undist_depth, intrinsic2 = undistort_depth_image2(
        depth_img_raw.copy(), depth_intrinsic, depth_distortion)
    depth_m = undist_depth.astype(np.float32) / 1000.0
    depth_m[(depth_m <= 0) | (depth_m > max_depth_m)] = 0

    h, w = depth_m.shape
    if h == 0 or w == 0:
        return np.empty((0, 3), dtype=np.float32)

    bx = max(1, min(int(border_x_px), max(1, w // 4)))
    by = max(1, min(int(border_y_px), max(1, h // 4)))
    ys, xs = np.mgrid[:h, :w]
    border_masks = {
        'bottom': ys >= h - by,
        'left': xs < bx,
        'right': xs >= w - bx,
        'top': ys < by,
    }
    border_union = np.logical_or.reduce(list(border_masks.values()))
    valid_border = border_union & (depth_m > 0)
    vals = depth_m[valid_border]
    if vals.size < 128:
        return np.empty((0, 3), dtype=np.float32)

    smooth_mask = compute_depth_smooth_mask(depth_m)
    side_weights = {'bottom': 0.40, 'left': 0.25, 'right': 0.25, 'top': 0.10}
    flat_indices = []
    min_points = 32
    for side_name, side_mask in border_masks.items():
        valid_side = side_mask & (depth_m > 0)
        if int(valid_side.sum()) < min_points:
            continue

        quota = max(min_points, int(round(float(max_points) * side_weights[side_name])))
        side_vals = depth_m[valid_side]
        depth_cut = float(np.percentile(side_vals, far_percentile))

        candidate = valid_side & smooth_mask & (depth_m >= depth_cut)
        if int(candidate.sum()) < min_points:
            candidate = valid_side & (depth_m >= depth_cut)
        if int(candidate.sum()) < min_points:
            candidate = valid_side & smooth_mask
        if int(candidate.sum()) < min_points:
            candidate = valid_side
        if int(candidate.sum()) == 0:
            continue

        side_flat = np.flatnonzero(candidate)
        if len(side_flat) > quota:
            side_flat = rng.choice(side_flat, size=int(quota), replace=False)
        flat_indices.append(side_flat)

    if not flat_indices:
        return np.empty((0, 3), dtype=np.float32)

    selected_flat = np.unique(np.concatenate(flat_indices))
    if len(selected_flat) > int(max_points):
        selected_flat = rng.choice(selected_flat, size=int(max_points), replace=False)

    valid = np.zeros(depth_m.shape, dtype=bool)
    valid.flat[selected_flat] = True
    xy_table = create_xy_table(intrinsic2)
    pts = np.stack([
        xy_table[..., 0][valid] * depth_m[valid],
        xy_table[..., 1][valid] * depth_m[valid],
        depth_m[valid],
    ], axis=1).astype(np.float32, copy=False)
    if len(pts) > max_points:
        idx = rng.choice(len(pts), size=int(max_points), replace=False)
        pts = pts[idx]
    return pts


def compute_ground_support_mask_from_depth(depth_img_raw, depth_intrinsic, depth_distortion, coef,
                                           residual_thresh_m=GROUND_RANSAC_THRESH_M,
                                           max_depth_m=GROUND_MAX_DEPTH_M):
    undist_depth, intrinsic2 = undistort_depth_image2(
        depth_img_raw.copy(), depth_intrinsic, depth_distortion)
    depth_m = undist_depth.astype(np.float32) / 1000.0
    depth_m[(depth_m <= 0) | (depth_m > max_depth_m)] = 0
    valid = depth_m > 0
    if not np.any(valid):
        return undist_depth, intrinsic2, np.zeros_like(depth_m, dtype=bool)

    xy_table = create_xy_table(intrinsic2)
    x = xy_table[..., 0] * depth_m
    y = xy_table[..., 1] * depth_m
    residual = np.abs(depth_m - (float(coef[0]) * x + float(coef[1]) * y + float(coef[2])))
    support_thresh = max(float(residual_thresh_m) * 3.0, 0.04)
    support_mask = valid & (residual <= support_thresh)
    smooth_mask = compute_depth_smooth_mask(depth_m)
    support_mask &= smooth_mask
    if not np.any(support_mask):
        support_mask = valid & (residual <= support_thresh)
    return undist_depth, intrinsic2, support_mask


def sample_ground_support_points_from_depth(depth_img_raw, depth_intrinsic, depth_distortion,
                                            coef, rng,
                                            residual_thresh_m=GROUND_RANSAC_THRESH_M,
                                            max_points=GROUND_MAX_POINTS_PER_FRAME):
    undist_depth, intrinsic2, support_mask = compute_ground_support_mask_from_depth(
        depth_img_raw,
        depth_intrinsic,
        depth_distortion,
        coef,
        residual_thresh_m=residual_thresh_m,
    )
    if not np.any(support_mask):
        return np.empty((0, 3), dtype=np.float32), support_mask

    depth_m = undist_depth.astype(np.float32) / 1000.0
    xy_table = create_xy_table(intrinsic2)
    pts = np.stack([
        xy_table[..., 0][support_mask] * depth_m[support_mask],
        xy_table[..., 1][support_mask] * depth_m[support_mask],
        depth_m[support_mask],
    ], axis=1).astype(np.float32, copy=False)
    if len(pts) > int(max_points):
        idx = rng.choice(len(pts), size=int(max_points), replace=False)
        pts = pts[idx]
    return pts, support_mask


def score_plane_on_depth_frame(depth_img_raw, depth_intrinsic, depth_distortion, coef,
                               residual_thresh_m=GROUND_RANSAC_THRESH_M):
    _, _, support_mask = compute_ground_support_mask_from_depth(
        depth_img_raw,
        depth_intrinsic,
        depth_distortion,
        coef,
        residual_thresh_m=residual_thresh_m,
    )
    if support_mask.size == 0 or not np.any(support_mask):
        return {
            'score': 0.0,
            'coverage': 0.0,
            'bottom_coverage': 0.0,
            'center_h_coverage': 0.0,
            'center_v_coverage': 0.0,
            'crosses_center': False,
        }

    h, w = support_mask.shape
    band = max(1, min(int(GROUND_CENTER_BAND_HALF_WIDTH_PX), h // 8, w // 8))
    center_y = h // 2
    center_x = w // 2
    y0 = max(0, center_y - band)
    y1 = min(h, center_y + band + 1)
    x0 = max(0, center_x - band)
    x1 = min(w, center_x + band + 1)

    center_h_coverage = float(support_mask[y0:y1, :].mean())
    center_v_coverage = float(support_mask[:, x0:x1].mean())
    bottom_rows = max(1, h // 8)
    bottom_coverage = float(support_mask[h - bottom_rows:, :].mean())
    coverage = float(support_mask.mean())
    center_support = max(center_h_coverage, center_v_coverage)
    crosses_center = center_support >= 0.05
    score = 3.0 * bottom_coverage + 2.5 * center_support + 0.75 * coverage + (1.5 if crosses_center else 0.0)
    return {
        'score': float(score),
        'coverage': coverage,
        'bottom_coverage': bottom_coverage,
        'center_h_coverage': center_h_coverage,
        'center_v_coverage': center_v_coverage,
        'crosses_center': bool(crosses_center),
    }


def select_best_ground_frames_for_bag(depth_imgs, depth_intrinsic, depth_distortion, rng,
                                      frames_per_bag=GROUND_FRAMES_PER_BAG,
                                      candidate_frames=GROUND_FRAME_CANDIDATES_PER_BAG):
    candidate_indices = choose_ground_frame_indices(len(depth_imgs), candidate_frames)
    frame_records = []
    for fi in candidate_indices:
        depth_img_raw = depth_imgs[int(fi)]
        pts = sample_ground_plane_points_from_depth(
            depth_img_raw,
            depth_intrinsic,
            depth_distortion,
            rng,
        )
        if len(pts) < 128:
            continue

        try:
            provisional = solve_plane_least_squares(pts)
            provisional_metrics = score_plane_on_depth_frame(
                depth_img_raw,
                depth_intrinsic,
                depth_distortion,
                provisional,
            )
        except Exception:
            provisional_metrics = {
                'score': 0.0,
                'coverage': 0.0,
                'bottom_coverage': 0.0,
                'center_h_coverage': 0.0,
                'center_v_coverage': 0.0,
                'crosses_center': False,
            }

        early_bonus = max(0.0, (GROUND_EARLY_FRAMES_PER_BAG - int(fi)) * 0.15)
        total_score = float(provisional_metrics['score']) + early_bonus
        frame_records.append({
            'frame_index': int(fi),
            'points': pts,
            'depth_img_raw': depth_img_raw.copy(),
            'score': total_score,
            'early_bonus': float(early_bonus),
            'crosses_center': bool(provisional_metrics['crosses_center']),
            'coverage': float(provisional_metrics['coverage']),
            'bottom_coverage': float(provisional_metrics['bottom_coverage']),
            'center_h_coverage': float(provisional_metrics['center_h_coverage']),
            'center_v_coverage': float(provisional_metrics['center_v_coverage']),
        })

    if not frame_records:
        return []

    frame_records.sort(
        key=lambda rec: (
            int(rec['crosses_center']),
            rec['score'],
            rec['bottom_coverage'],
            rec['coverage'],
            -rec['frame_index'],
        ),
        reverse=True,
    )

    selected = []
    seen = set()
    earliest_valid = min(frame_records, key=lambda rec: rec['frame_index'])
    selected.append(earliest_valid)
    seen.add(int(earliest_valid['frame_index']))
    for rec in frame_records:
        fi = int(rec['frame_index'])
        if fi in seen:
            continue
        selected.append(rec)
        seen.add(fi)
        if len(selected) >= int(frames_per_bag):
            break
    selected.sort(key=lambda rec: rec['frame_index'])
    return selected


def robust_refine_ground_plane(points_xyz, initial_coef,
                               inlier_thresh_m=GROUND_RANSAC_THRESH_M,
                               max_iters=4):
    xyz = np.asarray(points_xyz, dtype=np.float64)
    coef = np.asarray(initial_coef, dtype=np.float64).copy()
    A = np.c_[xyz[:, 0], xyz[:, 1], np.ones(len(xyz))]
    delta = max(float(inlier_thresh_m) * 1.5, 1e-3)

    for _ in range(int(max_iters)):
        residual = xyz[:, 2] - A @ coef
        abs_residual = np.abs(residual)
        weights = np.ones(len(xyz), dtype=np.float64)
        bad = abs_residual > delta
        weights[bad] = delta / np.maximum(abs_residual[bad], 1e-9)
        strong_outlier = abs_residual > (delta * 4.0)
        weights[strong_outlier] = 0.0
        if np.count_nonzero(weights > 0) < 3:
            break

        sqrt_w = np.sqrt(weights)
        Aw = A * sqrt_w[:, None]
        zw = xyz[:, 2] * sqrt_w
        coef_next, _, _, _ = np.linalg.lstsq(Aw, zw, rcond=None)
        if np.allclose(coef, coef_next, atol=1e-7, rtol=0):
            coef = coef_next
            break
        coef = coef_next

    residual = plane_residuals(xyz, coef)
    inliers = np.abs(residual) <= float(inlier_thresh_m)
    if int(inliers.sum()) >= 3:
        coef = solve_plane_least_squares(xyz[inliers])
        residual = plane_residuals(xyz, coef)
        inliers = np.abs(residual) <= float(inlier_thresh_m)
    return coef.astype(np.float64, copy=False), residual, inliers


def fit_ground_plane_ransac(points_xyz, rng, iterations=GROUND_RANSAC_ITERS,
                            inlier_thresh_m=GROUND_RANSAC_THRESH_M,
                            support_frames=None,
                            top_k=GROUND_TOP_K_CANDIDATES):
    if len(points_xyz) < 3:
        raise ValueError('Need at least 3 points to fit a ground plane')

    xyz = points_xyz.astype(np.float64, copy=False)
    candidate_records = []
    for _ in range(int(iterations)):
        ids = rng.choice(len(xyz), size=3, replace=False)
        sample = xyz[ids]
        A = np.c_[sample[:, 0], sample[:, 1], np.ones(3)]
        if abs(np.linalg.det(A)) < 1e-8:
            continue
        try:
            coef = np.linalg.solve(A, sample[:, 2])
        except np.linalg.LinAlgError:
            continue

        err = np.abs(xyz[:, 2] - (xyz[:, 0] * coef[0] + xyz[:, 1] * coef[1] + coef[2]))
        inliers = err <= float(inlier_thresh_m)
        n_inliers = int(inliers.sum())
        if n_inliers < 3:
            continue
        med_err = float(np.median(err[inliers]))
        candidate = {
            'coef': coef.astype(np.float64, copy=False),
            'inliers': inliers,
            'n_inliers': n_inliers,
            'median_abs_residual_m': med_err,
        }
        duplicate = False
        for prev in candidate_records:
            if plane_candidates_similar(prev['coef'], candidate['coef']):
                duplicate = True
                if candidate['n_inliers'] > prev['n_inliers'] or (
                    candidate['n_inliers'] == prev['n_inliers'] and
                    candidate['median_abs_residual_m'] < prev['median_abs_residual_m']
                ):
                    prev.update(candidate)
                break
        if not duplicate:
            candidate_records.append(candidate)

    if not candidate_records:
        raise RuntimeError('RANSAC failed to find a valid ground plane')

    candidate_records.sort(
        key=lambda rec: (rec['n_inliers'], -rec['median_abs_residual_m']),
        reverse=True,
    )
    candidate_records = candidate_records[:max(1, int(top_k))]

    best_record = None
    for rec in candidate_records:
        coef_refined, residual, inliers = robust_refine_ground_plane(
            xyz,
            rec['coef'],
            inlier_thresh_m=inlier_thresh_m,
        )
        support_metrics = []
        if support_frames:
            for frame in support_frames:
                support_metrics.append(score_plane_on_depth_frame(
                    frame['depth_img_raw'],
                    frame['depth_intrinsic'],
                    frame['depth_distortion'],
                    coef_refined,
                    residual_thresh_m=inlier_thresh_m,
                ))

        cross_frames = int(sum(1 for metric in support_metrics if metric['crosses_center']))
        support_score = float(sum(metric['score'] for metric in support_metrics))
        avg_support_score = support_score / max(1, len(support_metrics))
        avg_bottom = float(np.mean([metric['bottom_coverage'] for metric in support_metrics])) if support_metrics else 0.0
        avg_center = float(np.mean([
            max(metric['center_h_coverage'], metric['center_v_coverage'])
            for metric in support_metrics
        ])) if support_metrics else 0.0
        candidate_summary = {
            'a': float(coef_refined[0]),
            'b': float(coef_refined[1]),
            'c': float(coef_refined[2]),
            'median_abs_residual_m': float(np.median(np.abs(residual[inliers]))) if int(inliers.sum()) >= 1 else float(np.median(np.abs(residual))),
            'ransac_inliers': int(inliers.sum()),
            'support_score': float(support_score),
            'avg_support_score': float(avg_support_score),
            'support_cross_frames': cross_frames,
            'avg_bottom_coverage': float(avg_bottom),
            'avg_center_coverage': float(avg_center),
        }
        ranking_key = (
            cross_frames,
            avg_support_score,
            avg_bottom,
            avg_center,
            int(inliers.sum()),
            -candidate_summary['median_abs_residual_m'],
        )
        if best_record is None or ranking_key > best_record['ranking_key']:
            best_record = {
                'ranking_key': ranking_key,
                'coef': coef_refined,
                'summary': candidate_summary,
            }

    if best_record is None:
        raise RuntimeError('RANSAC failed to select a valid ground plane candidate')

    coef = best_record['coef']
    support_refit_points = []
    support_refit_frames = 0
    if support_frames:
        for frame in support_frames:
            pts_support, _ = sample_ground_support_points_from_depth(
                frame['depth_img_raw'],
                frame['depth_intrinsic'],
                frame['depth_distortion'],
                coef,
                rng,
                residual_thresh_m=inlier_thresh_m,
            )
            if len(pts_support) < 128:
                continue
            support_refit_points.append(pts_support)
            support_refit_frames += 1

    if support_refit_points:
        support_xyz = np.concatenate(support_refit_points, axis=0)
        if len(support_xyz) > GROUND_MAX_TOTAL_POINTS:
            idx = rng.choice(len(support_xyz), size=int(GROUND_MAX_TOTAL_POINTS), replace=False)
            support_xyz = support_xyz[idx]
        coef, _, _ = robust_refine_ground_plane(
            support_xyz,
            coef,
            inlier_thresh_m=inlier_thresh_m,
        )

    err = np.abs(plane_residuals(xyz, coef))
    inliers = err <= float(inlier_thresh_m)
    return {
        'a': float(coef[0]),
        'b': float(coef[1]),
        'c': float(coef[2]),
        'median_abs_residual_m': float(np.median(err[inliers])) if int(inliers.sum()) >= 1 else float(np.median(err)),
        'ransac_inliers': int(inliers.sum()),
        'points_used': int(len(xyz)),
        'source_candidates_considered': int(len(candidate_records)),
        'support_cross_frames': int(best_record['summary']['support_cross_frames']),
        'support_score': float(best_record['summary']['support_score']),
        'avg_support_score': float(best_record['summary']['avg_support_score']),
        'avg_bottom_coverage': float(best_record['summary']['avg_bottom_coverage']),
        'avg_center_coverage': float(best_record['summary']['avg_center_coverage']),
        'support_refit_frames': int(support_refit_frames),
        'support_refit_points_used': int(sum(len(x) for x in support_refit_points)) if support_refit_points else 0,
    }


def estimate_ground_plane_for_date(date_str, bagfiles_dir=None,
                                   max_bags_per_date=GROUND_MAX_BAGS_PER_DATE,
                                   frames_per_bag=GROUND_FRAMES_PER_BAG,
                                   seed=0):
    resolved_bagfiles_dir = resolve_bagfiles_dir(bagfiles_dir)
    bag_names = sorted([
        filename[:-3]
        for filename in os.listdir(resolved_bagfiles_dir)
        if filename.endswith('.h5') and extract_bag_date(filename[:-3]) == date_str
    ])
    if max_bags_per_date:
        bag_names = bag_names[:int(max_bags_per_date)]
    if not bag_names:
        raise FileNotFoundError(f'No HDF5 bags found for date {date_str} in {resolved_bagfiles_dir}')

    rng = np.random.default_rng(seed)
    all_points = []
    support_frames = []
    bags_used = []
    frames_used = 0

    for bag_name in bag_names:
        hdf5_path = join(resolved_bagfiles_dir, f'{bag_name}.h5')
        depth_intrinsic, depth_distortion = load_depth_camera_info_from_hdf5(hdf5_path)
        if depth_intrinsic is None:
            camera_params = get_camera_params_for_bag(bag_name)
            if camera_params is None:
                continue
            depth_intrinsic = copy.deepcopy(camera_params['DepthIntrinsic'])
            depth_distortion = copy.deepcopy(camera_params['DepthDistortion'])

        bag_frame_count = 0
        with h5py.File(hdf5_path, 'r') as handle:
            depth_imgs = handle.get('depth/images')
            if depth_imgs is None or len(depth_imgs) == 0:
                continue
            selected_frames = select_best_ground_frames_for_bag(
                depth_imgs,
                depth_intrinsic,
                depth_distortion,
                rng,
                frames_per_bag=frames_per_bag,
            )
            for frame_rec in selected_frames:
                pts = frame_rec['points']
                if len(pts) < 128:
                    continue
                all_points.append(pts)
                support_frames.append({
                    'depth_img_raw': frame_rec['depth_img_raw'],
                    'depth_intrinsic': copy.deepcopy(depth_intrinsic),
                    'depth_distortion': copy.deepcopy(depth_distortion),
                    'bag_name': bag_name,
                    'frame_index': int(frame_rec['frame_index']),
                    'frame_score': float(frame_rec['score']),
                })
                frames_used += 1
                bag_frame_count += 1
        if bag_frame_count:
            bags_used.append(bag_name)

    if not all_points:
        raise RuntimeError(f'Could not extract ground-plane points for date {date_str}')

    points_xyz = np.concatenate(all_points, axis=0)
    if len(points_xyz) > GROUND_MAX_TOTAL_POINTS:
        idx = rng.choice(len(points_xyz), size=int(GROUND_MAX_TOTAL_POINTS), replace=False)
        points_xyz = points_xyz[idx]

    record = fit_ground_plane_ransac(points_xyz, rng, support_frames=support_frames)
    record.update({
        'source': 'estimated_from_depth_support_mask',
        'date': date_str,
        'bags_used': bags_used,
        'frames_used': int(frames_used),
        'support_frames': [
            f"{frame['bag_name']}:{frame['frame_index']:05d}"
            for frame in support_frames
        ],
        'bagfiles_dir': resolved_bagfiles_dir,
        'estimated_utc': datetime.now(timezone.utc).isoformat(),
    })
    return record


def resolve_ground_plane_for_bag(bag_name_or_path, bagfiles_dir=None,
                                 ground_plane_json=DEFAULT_DATE_GROUND_PLANE_JSON,
                                 date_ground_planes=None):
    bag_name = basename(str(bag_name_or_path))
    if bag_name.endswith('.h5'):
        bag_name = bag_name[:-3]
    date_str = extract_bag_date(bag_name)

    cache = date_ground_planes if date_ground_planes is not None else load_date_ground_planes(ground_plane_json)
    if date_str and date_str in cache:
        record = cache[date_str]
        return plane_tuple_from_record(record), cache, record

    if not date_str:
        record = default_ground_plane_record()
        return plane_tuple_from_record(record), cache, record

    record = estimate_ground_plane_for_date(date_str, bagfiles_dir=bagfiles_dir)
    cache[date_str] = record
    save_date_ground_planes(cache, ground_plane_json)
    return plane_tuple_from_record(record), cache, record
