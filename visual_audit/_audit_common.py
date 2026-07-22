#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visual_audit/_audit_common.py
=============================
Shared helpers for the MORAI 3D-detection visual audit.

Reuses the EXACT camera-projection convention from visualize_camera_proj.py /
camera_configs.py and the BEV rotated-box convention from visualize_camera_proj.py.
Reuses eval_relative_speed.load_gt_with_relspeed for rel_speed and ego/obj velocity.

Read-only against the dataset.  Writes nothing here.
"""
import os
import sys
import csv
import json
import math
from collections import defaultdict

import numpy as np

PROJECT_ROOT = "/home/autonav/projects/morai-3d-detection"
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from camera_configs import EXTRINSICS as _EXTRINSICS, INTRINSICS as _INTRINSICS, CAM_ORDER  # noqa: E402
from eval_relative_speed import (  # noqa: E402
    load_gt_with_relspeed,
    local_linear_velocity,
    _read_epoch_ids,
)

CAM_W, CAM_H = 1600, 900
CLASS_NAME = {0: "vehicle", 1: "pedestrian"}

# cuboid 12 edges — identical to visualize_camera_proj.BOX_EDGES
BOX_EDGES = [
    (0, 2), (2, 6), (6, 4), (4, 0),
    (1, 3), (3, 7), (7, 5), (5, 1),
    (0, 1), (2, 3), (4, 5), (6, 7),
]

# v3 label column order (schema from task spec / confirmed against CSV header)
V3_COLS = ["frame_id", "timestamp", "track_id", "class_id", "x", "y", "z_center",
           "w", "l", "h", "yaw_ego", "vx_ego", "vy_ego", "vz", "yaw_global",
           "gx", "gy", "sin_yaw_ego", "cos_yaw_ego", "vel_source",
           "corr_dx", "corr_dy", "corr_dz", "corr_dist", "correction_valid"]


def _rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


# ----------------------------------------------------------------------------
# 3D box -> 8 ego corners  (raw sizes w,l,h ; z_center already the center)
# corner order + local frame identical to visualize_camera_proj.box_corners_ego
# local X = length(l), Y = width(w), Z = height(h)
# ----------------------------------------------------------------------------
def box_corners_ego(x, y, z_center, w, l, h, sin_y, cos_y):
    l = float(l); w = float(w); h = float(h)
    corners_local = np.array([
        [ l / 2,  w / 2,  h / 2], [ l / 2,  w / 2, -h / 2],
        [ l / 2, -w / 2,  h / 2], [ l / 2, -w / 2, -h / 2],
        [-l / 2,  w / 2,  h / 2], [-l / 2,  w / 2, -h / 2],
        [-l / 2, -w / 2,  h / 2], [-l / 2, -w / 2, -h / 2],
    ], dtype=np.float64)
    Rz = np.array([[cos_y, -sin_y, 0.0],
                   [sin_y,  cos_y, 0.0],
                   [0.0,    0.0,   1.0]], dtype=np.float64)
    return (Rz @ corners_local.T).T + np.array([x, y, z_center], dtype=np.float64)


def project_ego_points(pts_ego, cam):
    """pts_ego [N,3] ego-frame -> (us[N], vs[N], valid[N], depth[N]).
    Convention: depth=cam_x ; u=fx*(-cam_y)/depth+cx ; v=fy*(-cam_z)/depth+cy."""
    pts_ego = np.asarray(pts_ego, dtype=np.float64).reshape(-1, 3)
    n = pts_ego.shape[0]
    h = np.hstack([pts_ego, np.ones((n, 1))])
    E = _EXTRINSICS[cam]
    K = _INTRINSICS[cam]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    cam_pts = (E @ h.T).T
    depth = cam_pts[:, 0]
    valid = depth > 0.1
    us = np.full(n, np.nan)
    vs = np.full(n, np.nan)
    if valid.any():
        d = depth[valid]
        us[valid] = fx * (-cam_pts[valid, 1]) / d + cx
        vs[valid] = fy * (-cam_pts[valid, 2]) / d + cy
    return us, vs, valid, depth


def box_center_in_camera(box, cam):
    """True if box (dict with x,y,z_center) center projects inside `cam` frame (depth>0)."""
    us, vs, valid, _ = project_ego_points([[box["x"], box["y"], box["z_center"]]], cam)
    if not valid[0]:
        return False
    return (0 <= us[0] <= CAM_W) and (0 <= vs[0] <= CAM_H)


def best_camera_for_box(box):
    """Camera in CAM_ORDER whose frame the box center lands in; else None."""
    for cam in CAM_ORDER:
        if box_center_in_camera(box, cam):
            return cam
    return None


# ----------------------------------------------------------------------------
# label loading
# ----------------------------------------------------------------------------
def load_boxes(scen, stem, version):
    """Read one label CSV -> {track_id: {col:val,...}} with numeric fields cast.
    version in {'v2','v3'}."""
    sub = "labels_3d_v2" if version == "v2" else "labels_3d_v3"
    path = os.path.join(DATASET_ROOT, scen, sub, stem + ".csv")
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            b = {}
            for k, v in row.items():
                if k in ("vel_source",):
                    b[k] = v
                else:
                    try:
                        b[k] = float(v)
                    except (TypeError, ValueError):
                        b[k] = v
            b["track_id"] = int(float(row["track_id"]))
            b["class_id"] = int(float(row["class_id"]))
            b["frame_id"] = int(float(row["frame_id"]))
            out[b["track_id"]] = b
    return out


def scene_timing_map(scen):
    """stem -> timing dict (ego_dt_ms, obj_dt_ms, ref_ts, correction_valid...) from scene_info_v3."""
    path = os.path.join(DATASET_ROOT, scen, "scene_info_v3.json")
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    out = {}
    for fr in info["frames"]:
        out[fr["stem"]] = {
            "frame_id": fr["frame_id"],
            "timing": fr.get("timing", {}),
            "ego": fr.get("ego", {}),
        }
    return out


def frameid_to_stem(scen, frame_id, version="v3"):
    name = "scene_info_v3.json" if version == "v3" else "scene_info.json"
    path = os.path.join(DATASET_ROOT, scen, name)
    with open(path, encoding="utf-8") as f:
        info = json.load(f)
    for fr in info["frames"]:
        if int(fr["frame_id"]) == int(frame_id):
            return fr["stem"]
    return None


# ----------------------------------------------------------------------------
# ego world velocity vector per stem (local linear regression, same as eval_relative_speed)
# ----------------------------------------------------------------------------
def ego_world_velocity_map(scen):
    """stem -> {'vx_world','vy_world','yaw','ex','ey','n_used'} using local_linear_velocity."""
    scen_dir = os.path.join(DATASET_ROOT, scen)
    with open(os.path.join(scen_dir, "scene_info_v3.json"), encoding="utf-8") as f:
        info = json.load(f)
    frames = sorted(info["frames"], key=lambda fr: fr["stem"])
    stems = [fr["stem"] for fr in frames]
    ts = [float(fr["timestamp"]) for fr in frames]
    ex = [float(fr["ego"]["x"]) for fr in frames]
    ey = [float(fr["ego"]["y"]) for fr in frames]
    eyaw = [float(fr["ego"]["yaw"]) for fr in frames]
    epoch_map = _read_epoch_ids(scen_dir)
    epoch_ids = [epoch_map.get(s, "") for s in stems]
    out = {}
    for i, s in enumerate(stems):
        vx, vy, a = local_linear_velocity(ts, ex, ey, epoch_ids, i)
        out[s] = {"vx_world": vx, "vy_world": vy, "yaw": eyaw[i],
                  "ex": ex[i], "ey": ey[i], "n_used": a["n_used"]}
    return out


def relspeed_lookup(scen, gt_version="v3"):
    """(scen) -> {stem: {track_id: box_with_rel_speed}} using eval_relative_speed."""
    frames, audit = load_gt_with_relspeed(DATASET_ROOT, scen, gt_version)
    out = {}
    for fr in frames:
        out[fr["stem"]] = {b["track_id"]: b for b in fr["boxes"]}
    return out, audit


# ----------------------------------------------------------------------------
# drawing helpers
# ----------------------------------------------------------------------------
def draw_cuboid(ax, box, cam, color, lw=2.0):
    """draw projected 3D cuboid on an image axis. returns (drew, center_uv or None)."""
    corners = box_corners_ego(box["x"], box["y"], box["z_center"],
                              box["w"], box["l"], box["h"],
                              box["sin_yaw_ego"], box["cos_yaw_ego"])
    us, vs, valid, _ = project_ego_points(corners, cam)
    drew = False
    for a, b in BOX_EDGES:
        if not (valid[a] and valid[b]):
            continue
        ax.plot([us[a], us[b]], [vs[a], vs[b]], color=color, lw=lw, solid_capstyle="round")
        drew = True
    cu, cv, cvalid, _ = project_ego_points([[box["x"], box["y"], box["z_center"]]], cam)
    center = (float(cu[0]), float(cv[0])) if cvalid[0] else None
    return drew, center


def draw_bev_box(ax, box, color, lw=2.0, alpha=1.0, ls="-"):
    """rotated BEV rect (same convention as visualize_camera_proj.draw_rotated_box_bev)."""
    import matplotlib.patches as patches
    import matplotlib.transforms as mtransforms
    x, y = float(box["x"]), float(box["y"])
    w, l = float(box["w"]), float(box["l"])
    angle = math.degrees(math.atan2(float(box["sin_yaw_ego"]), float(box["cos_yaw_ego"])))
    rect = patches.Rectangle((x - l / 2, y - w / 2), l, w, linewidth=lw,
                             edgecolor=color, facecolor="none", alpha=alpha, linestyle=ls)
    rect.set_transform(mtransforms.Affine2D().rotate_deg_around(x, y, angle) + ax.transData)
    ax.add_patch(rect)


def scen_dir(scen):
    return os.path.join(DATASET_ROOT, scen)


def image_path(scen, stem, cam):
    return os.path.join(DATASET_ROOT, scen, "images", cam, stem + ".jpg")


def depth_path(scen, stem, cam):
    return os.path.join(DATASET_ROOT, scen, "depth_gt", cam, stem + ".npy")
