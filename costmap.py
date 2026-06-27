import os

import cv2
import numpy as np


DEFAULT_X_RANGE = (0.0, 60.0)
DEFAULT_Y_RANGE = (-30.0, 30.0)
DEFAULT_RESOLUTION = 0.25


def box_bev_corners(box):
    x, y = float(box[0]), float(box[1])
    w = float(np.exp(box[3]))
    l = float(np.exp(box[4]))
    sin_yaw = float(box[6])
    cos_yaw = float(box[7])

    local = np.array([
        [ l * 0.5,  w * 0.5],
        [ l * 0.5, -w * 0.5],
        [-l * 0.5, -w * 0.5],
        [-l * 0.5,  w * 0.5],
    ], dtype=np.float32)
    rot = np.array([
        [cos_yaw, -sin_yaw],
        [sin_yaw,  cos_yaw],
    ], dtype=np.float32)
    return (rot @ local.T).T + np.array([x, y], dtype=np.float32)


def ego_xy_to_grid(points_xy, x_range, y_range, resolution):
    x_min, x_max = x_range
    y_min, _ = y_range
    cols = (points_xy[:, 1] - y_min) / resolution
    rows = (x_max - points_xy[:, 0]) / resolution
    return np.stack([cols, rows], axis=1).astype(np.int32)


def rasterize_detections(
    boxes,
    labels,
    scores=None,
    x_range=DEFAULT_X_RANGE,
    y_range=DEFAULT_Y_RANGE,
    resolution=DEFAULT_RESOLUTION,
):
    """
    Returns [4, H, W]:
      0 occupancy score
      1 vehicle risk
      2 pedestrian risk
      3 speed-weighted dynamic risk
    """
    boxes = np.asarray(boxes, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if scores is None:
        scores = np.ones((len(boxes),), dtype=np.float32)
    else:
        scores = np.asarray(scores, dtype=np.float32)

    h = int(np.ceil((x_range[1] - x_range[0]) / resolution))
    w = int(np.ceil((y_range[1] - y_range[0]) / resolution))
    costmap = np.zeros((4, h, w), dtype=np.float32)

    for box, cls_id, score in zip(boxes, labels, scores):
        corners = box_bev_corners(box)
        poly = ego_xy_to_grid(corners, x_range, y_range, resolution)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [poly.reshape(-1, 1, 2)], 1)
        mask_bool = mask.astype(bool)
        if not mask_bool.any():
            continue

        dist_decay = np.exp(-max(float(box[0]), 0.0) / 35.0)
        speed = float(np.linalg.norm(box[8:10])) if len(box) >= 10 else 0.0
        dynamic_gain = 1.0 + min(speed / 10.0, 2.0)
        base_risk = float(score) * (0.5 + 0.5 * dist_decay)

        costmap[0, mask_bool] = np.maximum(costmap[0, mask_bool], float(score))
        if int(cls_id) == 0:
            costmap[1, mask_bool] = np.maximum(costmap[1, mask_bool], base_risk)
        elif int(cls_id) == 1:
            costmap[2, mask_bool] = np.maximum(costmap[2, mask_bool], min(base_risk * 1.3, 1.0))
        costmap[3, mask_bool] = np.maximum(costmap[3, mask_bool], min(base_risk * dynamic_gain, 1.0))

    return np.clip(costmap, 0.0, 1.0)


def save_costmap(costmap, out_prefix):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    np.save(out_prefix + ".npy", costmap.astype(np.float32))

    occ = np.clip(costmap[0], 0.0, 1.0)
    vehicle = np.clip(costmap[1], 0.0, 1.0)
    ped = np.clip(costmap[2], 0.0, 1.0)
    dyn = np.clip(costmap[3], 0.0, 1.0)
    preview = np.stack([ped, vehicle, np.maximum(occ, dyn)], axis=-1)
    preview = (preview * 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(cv2.cvtColor(preview, cv2.COLOR_RGB2GRAY), cv2.COLORMAP_TURBO)
    cv2.imwrite(out_prefix + ".png", preview)
