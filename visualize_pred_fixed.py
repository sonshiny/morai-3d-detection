import os
import cv2
import json
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms

from train import AutoNavModel
from morai_dataset import MoraiDataset
from camera_configs import CAM_ORDER

IMG_W = 640
IMG_H = 480

DET_COLORS = {
    0: 'red',      # car
    1: 'orange',   # truck
    2: 'magenta',  # bus
}
DET_NAMES = {
    0: 'car',
    1: 'truck',
    2: 'bus',
}
MAP_COLORS = {
    0: 'yellow',   # lane
    1: 'cyan',     # crosswalk
    2: 'white',    # road_boundary
}
MAP_NAMES = {
    0: 'lane',
    1: 'crosswalk',
    2: 'road_boundary',
}

# 너무 많이 나오면 조절
DET_THRESH = 0.20
MAP_THRESH = 0.99


def draw_rotated_box_bev(ax, x, y, w, l, sin_yaw, cos_yaw, color, lw=2.0, label=None):
    angle = np.degrees(np.arctan2(sin_yaw, cos_yaw))
    rect = patches.Rectangle(
        (x - l / 2, y - w / 2),
        l, w,
        angle=0,
        linewidth=lw,
        edgecolor=color,
        facecolor='none'
    )
    t = transforms.Affine2D().rotate_deg_around(x, y, angle) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    if label is not None:
        ax.text(x, y, label, color=color, fontsize=8)


def box3d_corners_from_param(box):
    # box = [x, y, z, ln_w, ln_l, ln_h, sin_yaw, cos_yaw, vx, vy, vz]
    x, y, z = box[0], box[1], box[2]
    w = float(np.exp(box[3]))
    l = float(np.exp(box[4]))
    h = float(np.exp(box[5]))
    sin_yaw, cos_yaw = box[6], box[7]

    yaw = math.atan2(sin_yaw, cos_yaw)

    # 차량 local 좌표: x=길이방향, y=폭방향, z=높이
    x_c = np.array([ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2], dtype=np.float32)
    y_c = np.array([ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2], dtype=np.float32)
    z_c = np.array([ h/2,  h/2,  h/2,  h/2, -h/2, -h/2, -h/2, -h/2], dtype=np.float32)

    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    pts = np.stack([x_c, y_c, z_c], axis=0)
    pts = rot @ pts
    pts[0, :] += x
    pts[1, :] += y
    pts[2, :] += z

    corners = np.concatenate([pts.T, np.ones((8, 1), dtype=np.float32)], axis=1)  # [8,4]
    return corners


def project_box_to_image(box, K, E):
    corners = box3d_corners_from_param(box)   # [8,4]
    cam_pts = (E @ corners.T).T               # [8,4]

    depth = cam_pts[:, 0]
    valid = depth > 0.1
    if valid.sum() < 8:
        return None

    u = K[0, 0] * (-cam_pts[:, 1]) / depth + K[0, 2]
    v = K[1, 1] * (-cam_pts[:, 2]) / depth + K[1, 2]

    x1, y1 = np.min(u), np.min(v)
    x2, y2 = np.max(u), np.max(v)

    # 화면 밖이면 어느 정도는 허용하되 완전히 밖이면 스킵
    if x2 < 0 or y2 < 0 or x1 > IMG_W or y1 > IMG_H:
        return None

    x1 = max(0, min(IMG_W - 1, x1))
    y1 = max(0, min(IMG_H - 1, y1))
    x2 = max(0, min(IMG_W - 1, x2))
    y2 = max(0, min(IMG_H - 1, y2))

    if x2 - x1 < 2 or y2 - y1 < 2:
        return None

    return int(x1), int(y1), int(x2), int(y2)


def find_group_by_stem(dataset, stem):
    for i, g in enumerate(dataset.groups):
        if g['label_stem'] == stem:
            return i, g
    raise ValueError(f"stem not found in train split: {stem}")


def visualize_one(model, dataset, stem, out_dir="pred_vis"):
    os.makedirs(out_dir, exist_ok=True)

    idx, group = find_group_by_stem(dataset, stem)
    sample = dataset[idx]

    device = next(model.parameters()).device

    images = sample['images'].unsqueeze(0).to(device)
    intrinsics = sample['intrinsics'].unsqueeze(0).to(device)
    extrinsics = sample['extrinsics'].unsqueeze(0).to(device)

    with torch.no_grad():
        det_logits, det_boxes, map_logits, map_lines = model(images, intrinsics, extrinsics)

    det_probs = torch.softmax(det_logits, dim=-1).cpu().numpy()   # [900,4]
    det_boxes = det_boxes.cpu().numpy()                           # [900,11]
    map_probs = torch.softmax(map_logits, dim=-1).cpu().numpy()   # [100,3]
    map_lines = map_lines.cpu().numpy()                           # [100,20,2]

    gt_boxes = sample['dynamic_gt_boxes'].cpu().numpy()
    gt_labels = sample['dynamic_gt_labels'].cpu().numpy()
    gt_lines = sample['static_gt_polylines'].cpu().numpy()
    gt_line_labels = sample['static_gt_labels'].cpu().numpy()

    # -----------------------------
    # 1) BEV 시각화 (GT + Pred)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('black')
    ax.plot(0, 0, marker='>', color='white', markersize=16)

    # GT static
    for i in range(len(gt_lines)):
        cls_id = int(gt_line_labels[i])
        color = 'lime'
        line = gt_lines[i]
        ax.plot(line[:, 0], line[:, 1], color=color, linewidth=1.5, alpha=0.6)


    # Pred static
    pred_map_count = 0
    for i in range(map_lines.shape[0]):
        bg_prob = float(map_probs[i, 3])
        if bg_prob > 0.5:
            continue
        cls_id = int(np.argmax(map_probs[i, :3]))
        conf = float(np.max(map_probs[i, :3]))
        if conf < MAP_THRESH:
            continue
        line = map_lines[i]
        color = MAP_COLORS.get(cls_id, 'yellow')
        ax.plot(line[:, 0], line[:, 1], color=color,
                linestyle='--', linewidth=2.0, alpha=0.9)
        pred_map_count += 1


    # GT dynamic
    for i in range(len(gt_boxes)):
        box = gt_boxes[i]
        cls_id = int(gt_labels[i])
        w = float(np.exp(box[3]))
        l = float(np.exp(box[4]))
        draw_rotated_box_bev(
            ax, box[0], box[1], w, l, box[6], box[7],
            color='cyan', lw=1.5, label=DET_NAMES.get(cls_id, 'obj')
        )

    # Pred dynamic
    pred_count = 0
    for i in range(len(det_boxes)):
        cls_id = int(np.argmax(det_probs[i]))
        conf = float(det_probs[i, cls_id])

        # background = 3
        if cls_id == 3 or conf < DET_THRESH:
            continue

        box = det_boxes[i]
        w = float(np.exp(box[3]))
        l = float(np.exp(box[4]))

        draw_rotated_box_bev(
            ax, box[0], box[1], w, l, box[6], box[7],
            color=DET_COLORS.get(cls_id, 'red'),
            lw=2.0,
            label=f"{DET_NAMES.get(cls_id, 'obj')}:{conf:.2f}"
        )
        pred_count += 1

    ax.set_xlim(-60, 60)
    ax.set_ylim(-60, 60)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"Pred BEV — {stem} | pred_boxes={pred_count}", color='white', fontsize=14)
    ax.set_xlabel("X (forward, m)", color='white')
    ax.set_ylabel("Y (left, m)", color='white')
    ax.tick_params(colors='white')
    ax.grid(True, color='gray', linestyle=':', alpha=0.4)
    plt.tight_layout()
    bev_path = os.path.join(out_dir, f"{stem}_pred_bev.png")
    plt.savefig(bev_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # -----------------------------
    # 2) Camera 시각화 (Pred only)
    # -----------------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    axes = axes.reshape(2, 3)

    for cam_i, cam_name in enumerate(CAM_ORDER):
        r = cam_i // 3
        c = cam_i % 3
        ax = axes[r, c]

        if cam_name not in group['cams']:
            ax.set_title(cam_name)
            ax.axis('off')
            continue

        img_stem = group['cams'][cam_name]
        img_path = os.path.join(dataset.img_dir, f"{img_stem}.jpg")
        img = cv2.imread(img_path)

        if img is None:
            ax.set_title(cam_name)
            ax.axis('off')
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        K = sample['intrinsics'][cam_i].cpu().numpy()
        E = sample['extrinsics'][cam_i].cpu().numpy()

        for i in range(len(det_boxes)):
            cls_id = int(np.argmax(det_probs[i]))
            conf = float(det_probs[i, cls_id])

            if cls_id == 3 or conf < DET_THRESH:
                continue

            rect = project_box_to_image(det_boxes[i], K, E)
            if rect is None:
                continue

            x1, y1, x2, y2 = rect
            color = DET_COLORS.get(cls_id, 'red')
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(
                img,
                f"{DET_NAMES.get(cls_id, 'obj')}:{conf:.2f}",
                (x1, max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
                cv2.LINE_AA
            )

        ax.imshow(img)
        ax.set_title(cam_name)
        ax.axis('off')

    plt.suptitle(f"Pred Camera Projection — {stem}", fontsize=18)
    plt.tight_layout()
    cam_path = os.path.join(out_dir, f"{stem}_pred_cam.png")
    plt.savefig(cam_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[saved] {bev_path}")
    print(f"[saved] {cam_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = "checkpoint_epoch80.pth"
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError("best_model.pth not found")

    model = AutoNavModel().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dataset = MoraiDataset(dataset_dir='./dataset', split='train')

    stems = [
        "cam_front_00065",
        "cam_front_00108",
        "cam_front_00169",
    ]

    for stem in stems:
        visualize_one(model, dataset, stem, out_dir="pred_vis")


if __name__ == "__main__":
    main()