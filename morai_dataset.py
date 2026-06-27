import os
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from camera_configs import INTRINSICS as _INTRINSICS, EXTRINSICS as _EXTRINSICS, CAM_ORDER

IMG_WIDTH = 704
IMG_HEIGHT = 256
IMG_SIZE = IMG_HEIGHT  # legacy alias; do not use for new resize code.
ORIG_IMG_WIDTH = 1600
ORIG_IMG_HEIGHT = 900
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def scale_intrinsic_for_input(K, input_w=IMG_WIDTH, input_h=IMG_HEIGHT):
    """
    Scale the original 1600x900 camera matrix into the resized model input
    coordinate system. This keeps 3D projection and grid_sample aligned.
    """
    K_scaled = K.copy()
    sx = float(input_w) / float(ORIG_IMG_WIDTH)
    sy = float(input_h) / float(ORIG_IMG_HEIGHT)
    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy
    return K_scaled


def _box_corners_ego(box):
    x, y, z_bottom = box[0], box[1], box[2]
    w = float(np.exp(box[3]))
    l = float(np.exp(box[4]))
    h = float(np.exp(box[5]))
    sin_y, cos_y = box[6], box[7]
    z_center = z_bottom + h * 0.5

    corners_local = np.array([
        [ l * 0.5,  w * 0.5,  h * 0.5],
        [ l * 0.5,  w * 0.5, -h * 0.5],
        [ l * 0.5, -w * 0.5,  h * 0.5],
        [ l * 0.5, -w * 0.5, -h * 0.5],
        [-l * 0.5,  w * 0.5,  h * 0.5],
        [-l * 0.5,  w * 0.5, -h * 0.5],
        [-l * 0.5, -w * 0.5,  h * 0.5],
        [-l * 0.5, -w * 0.5, -h * 0.5],
        [0.0, 0.0, 0.0],
    ], dtype=np.float32)

    rot = np.array([
        [cos_y, -sin_y, 0.0],
        [sin_y,  cos_y, 0.0],
        [0.0,    0.0,   1.0],
    ], dtype=np.float32)
    return (rot @ corners_local.T).T + np.array([x, y, z_center], dtype=np.float32)


def box_visible_in_any_camera(box, min_depth=0.1, min_visible_points=1):
    corners = _box_corners_ego(box)
    corners_h = np.concatenate(
        [corners, np.ones((corners.shape[0], 1), dtype=np.float32)],
        axis=1,
    )

    for cam_name in CAM_ORDER:
        K = scale_intrinsic_for_input(_INTRINSICS[cam_name])
        E = _EXTRINSICS[cam_name]
        pts = (E @ corners_h.T).T
        depth = pts[:, 0]
        valid_depth = depth > min_depth
        if int(valid_depth.sum()) < min_visible_points:
            continue

        d = depth[valid_depth]
        u = K[0, 0] * (-pts[valid_depth, 1]) / (d + 1e-6) + K[0, 2]
        v = K[1, 1] * (-pts[valid_depth, 2]) / (d + 1e-6) + K[1, 2]

        inside = (
            (u >= 0.0) & (u < float(IMG_WIDTH)) &
            (v >= 0.0) & (v < float(IMG_HEIGHT))
        )
        if inside.any():
            return True

        # Keep partially visible boxes whose projected valid-depth bbox intersects the image.
        if (
            float(u.max()) >= 0.0 and float(u.min()) < float(IMG_WIDTH) and
            float(v.max()) >= 0.0 and float(v.min()) < float(IMG_HEIGHT)
        ):
            return True

    return False


class MoraiDataset(Dataset):
    """
    dataset_root/
      scen01/
        images/cam_front/live_000000.jpg
        images/cam_front_left/live_000000.jpg
        images/cam_front_right/live_000000.jpg
        labels_3d/live_000000.csv
      scen02/...

    Split 방식 (시나리오 단위 — data leakage 방지):
      - val_scenarios에 명시된 시나리오만 val, 나머지 전부 train
      - val_scenarios=None → 알파벳 정렬 마지막 5개를 자동 val

    __getitem__ 반환:
        images              : [3, 3, 256, 704]
        intrinsics          : [3, 3, 3]
        extrinsics          : [3, 4, 4]
        dynamic_gt_boxes    : [N, 11]
        dynamic_gt_labels   : [N]
        ego_pose            : [6] = timestamp, ego_x, ego_y, ego_z, ego_yaw_rad, valid
        stem                : str
    """

    def __init__(
        self,
        dataset_root='/data/dataset',
        split='train',
        val_scenarios=None,
        filter_visible=True,
    ):
        if split not in ('train', 'val'):
            raise ValueError(f"split는 'train' 또는 'val'이어야 합니다: {split}")
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"[ERROR] dataset_root 없음: {dataset_root}")

        scen_dirs = sorted([
            os.path.join(dataset_root, d)
            for d in os.listdir(dataset_root)
            if os.path.isdir(os.path.join(dataset_root, d))
            and os.path.isdir(os.path.join(dataset_root, d, 'labels_3d'))
        ])
        if not scen_dirs:
            raise FileNotFoundError(
                f"[ERROR] {dataset_root} 아래에 labels_3d 폴더를 가진 시나리오가 없습니다."
            )

        scen_names = [os.path.basename(d) for d in scen_dirs]

        if val_scenarios is None:
            n_val = min(5, len(scen_names))
            val_scenarios = scen_names[-n_val:]
        else:
            val_scenarios = list(val_scenarios)
            unknown = [n for n in val_scenarios if n not in scen_names]
            if unknown:
                raise ValueError(
                    f"[ERROR] val_scenarios에 존재하지 않는 시나리오: {unknown}\n"
                    f"  사용 가능한 시나리오: {scen_names}"
                )

        if split == 'train':
            selected = [d for d, n in zip(scen_dirs, scen_names) if n not in val_scenarios]
        else:
            selected = [d for d, n in zip(scen_dirs, scen_names) if n in val_scenarios]

        if not selected:
            raise RuntimeError(
                f"[ERROR] split='{split}'에 해당하는 시나리오가 없습니다.\n"
                f"  val_scenarios={val_scenarios}, 전체={scen_names}"
            )

        self.items = []
        self.filter_visible = filter_visible
        for scen_dir in selected:
            lbl_dir = os.path.join(scen_dir, 'labels_3d')
            stems = sorted([
                os.path.splitext(f)[0]
                for f in os.listdir(lbl_dir)
                if f.endswith('.csv')
            ])
            for stem in stems:
                self.items.append((scen_dir, stem))

        if not self.items:
            raise FileNotFoundError(f"[ERROR] {split} split에 CSV 파일이 없습니다.")

        selected_names = [os.path.basename(d) for d in selected]
        print(f"[MoraiDataset:{split}] 시나리오 {selected_names} | {len(self.items):,} 프레임")

    def __len__(self):
        return len(self.items)

    def _load_image(self, scen_dir, stem, cam_name):
        path    = os.path.join(scen_dir, 'images', cam_name, f"{stem}.jpg")
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return torch.zeros(3, IMG_HEIGHT, IMG_WIDTH)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rs  = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
        img_t = torch.from_numpy(img_rs).permute(2, 0, 1).float() / 255.0
        return (img_t - IMG_MEAN) / IMG_STD

    def _load_labels(self, scen_dir, stem):
        csv_path = os.path.join(scen_dir, 'labels_3d', f"{stem}.csv")
        boxes, labels = [], []

        if os.path.isfile(csv_path):
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cls_id = int(float(row['class_id']))
                    vals = [
                        float(row['x']),       float(row['y']),       float(row['z']),
                        float(row['ln_w']),    float(row['ln_l']),    float(row['ln_h']),
                        float(row['sin_yaw']), float(row['cos_yaw']),
                        float(row['vx']),      float(row['vy']),      float(row['vz']),
                    ]
                    if self.filter_visible and not box_visible_in_any_camera(vals):
                        continue
                    boxes.append(vals)
                    labels.append(cls_id)

        if boxes:
            return (
                torch.tensor(boxes,  dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
            )
        return (
            torch.zeros((0, 11), dtype=torch.float32),
            torch.zeros((0,),    dtype=torch.long),
        )

    def _load_ego_pose(self, scen_dir, stem):
        pose_path = os.path.join(scen_dir, 'ego_pose', f"{stem}.csv")
        if os.path.isfile(pose_path):
            with open(pose_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
            if row is not None:
                yaw = row.get('ego_yaw_rad')
                if yaw is None or yaw == "":
                    yaw = np.radians(float(row.get('ego_heading_deg', 0.0)))
                return torch.tensor([
                    float(row.get('timestamp', 0.0)),
                    float(row.get('ego_x', 0.0)),
                    float(row.get('ego_y', 0.0)),
                    float(row.get('ego_z', 0.0)),
                    float(yaw),
                    1.0,
                ], dtype=torch.float32)

        # Older datasets do not contain ego pose metadata. valid=0 keeps
        # temporal alignment disabled rather than using an inaccurate identity.
        return torch.zeros(6, dtype=torch.float32)

    def __getitem__(self, idx):
        scen_dir, stem = self.items[idx]
        n_cams = len(CAM_ORDER)

        images = torch.zeros(n_cams, 3, IMG_HEIGHT, IMG_WIDTH)
        for ci, cam_name in enumerate(CAM_ORDER):
            images[ci] = self._load_image(scen_dir, stem, cam_name)

        intrinsics = torch.zeros(n_cams, 3, 3)
        extrinsics = torch.zeros(n_cams, 4, 4)
        for ci, cam_name in enumerate(CAM_ORDER):
            intrinsics[ci] = torch.from_numpy(scale_intrinsic_for_input(_INTRINSICS[cam_name]))
            extrinsics[ci] = torch.from_numpy(_EXTRINSICS[cam_name])

        gt_boxes, gt_labels = self._load_labels(scen_dir, stem)
        ego_pose = self._load_ego_pose(scen_dir, stem)

        return {
            'images':            images,
            'intrinsics':        intrinsics,
            'extrinsics':        extrinsics,
            'dynamic_gt_boxes':  gt_boxes,
            'dynamic_gt_labels': gt_labels,
            'ego_pose':          ego_pose,
            'stem':              f"{os.path.basename(scen_dir)}/{stem}",
        }


def morai_collate_fn(batch):
    return {
        'images':            torch.stack([b['images']     for b in batch]),
        'intrinsics':        torch.stack([b['intrinsics'] for b in batch]),
        'extrinsics':        torch.stack([b['extrinsics'] for b in batch]),
        'dynamic_gt_boxes':  [b['dynamic_gt_boxes']  for b in batch],
        'dynamic_gt_labels': [b['dynamic_gt_labels'] for b in batch],
        'ego_pose':          torch.stack([b['ego_pose'] for b in batch]),
        'stem':              [b['stem'] for b in batch],
    }


if __name__ == "__main__":
    ds_tr = MoraiDataset(dataset_root='/data/dataset', split='train')
    ds_va = MoraiDataset(dataset_root='/data/dataset', split='val')

    loader = DataLoader(ds_tr, batch_size=2, shuffle=True,
                        collate_fn=morai_collate_fn, num_workers=0)
    batch  = next(iter(loader))
    print(f"images     : {batch['images'].shape}")
    print(f"intrinsics : {batch['intrinsics'].shape}")
    print(f"extrinsics : {batch['extrinsics'].shape}")
    print(f"GT boxes   : {batch['dynamic_gt_boxes'][0].shape}")
    print(f"stems      : {batch['stem']}")
    print("✅ 데이터셋 정상!")
