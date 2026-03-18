import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ===========================================================
# 카메라 설정
# ===========================================================
CAMERA_CONFIGS = {
    'cam_back':         {'offset_xyz': [-0.35,  0.00, 1.23], 'rpy_deg': [0.0,  5.0, 180.0], 'fov_h': 90, 'w': 640, 'h': 480},
    'cam_back_left':    {'offset_xyz': [ 0.57,  0.85, 1.30], 'rpy_deg': [0.0,  5.0, 125.0], 'fov_h': 90, 'w': 640, 'h': 480},
    'cam_back_right':   {'offset_xyz': [ 0.57, -0.85, 1.30], 'rpy_deg': [0.0,  5.0, 235.0], 'fov_h': 90, 'w': 640, 'h': 480},
    'cam_front':        {'offset_xyz': [ 1.88,  0.00, 1.35], 'rpy_deg': [0.0, 15.0,   0.0], 'fov_h': 90, 'w': 640, 'h': 480},
    'cam_front_left':   {'offset_xyz': [ 1.40,  0.85, 1.35], 'rpy_deg': [0.0,  5.0,  55.0], 'fov_h': 90, 'w': 640, 'h': 480},
    'cam_front_right':  {'offset_xyz': [ 1.40, -0.85, 1.35], 'rpy_deg': [0.0,  5.0, 305.0], 'fov_h': 90, 'w': 640, 'h': 480},
}

CAM_ORDER = ['cam_front', 'cam_front_left', 'cam_front_right',
             'cam_back',  'cam_back_left',  'cam_back_right']

IMG_SIZE = 224


# ===========================================================
# 카메라 행렬 계산
# ===========================================================
def _compute_intrinsic(w, h, fov_h_deg):
    fov_rad = np.radians(fov_h_deg)
    fx = (w / 2.0) / np.tan(fov_rad / 2.0)
    return np.array([[fx,  0, w/2],
                     [ 0, fx, h/2],
                     [ 0,  0,   1]], dtype=np.float32)

def _compute_extrinsic(offset_xyz, rpy_deg):
    from scipy.spatial.transform import Rotation
    roll, pitch, yaw = rpy_deg
    R = Rotation.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix().T
    t = -R @ np.array(offset_xyz, dtype=np.float32)
    E = np.eye(4, dtype=np.float32)
    E[:3, :3] = R
    E[:3,  3] = t
    return E

_INTRINSICS = {k: _compute_intrinsic(v['w'], v['h'], v['fov_h'])    for k, v in CAMERA_CONFIGS.items()}
_EXTRINSICS = {k: _compute_extrinsic(v['offset_xyz'], v['rpy_deg']) for k, v in CAMERA_CONFIGS.items()}


# ===========================================================
# MoraiDataset — 6대 카메라 1묶음 버전
# ===========================================================
class MoraiDataset(Dataset):
    """
    frame_groups.json 기반으로 같은 타임스탬프의
    6대 카메라를 한 번에 로드합니다.

    __getitem__ 반환:
        images     : [6, 3, 224, 224]  (없는 카메라 슬롯은 0)
        intrinsics : [6, 3, 3]
        extrinsics : [6, 4, 4]
        dynamic_gt_boxes  : [N, 11]
        dynamic_gt_labels : [N]
        stem       : str (cam_front 기준 파일명)
    """

    def __init__(self, dataset_dir='./dataset', split='train', val_ratio=0.1):
        self.img_dir = os.path.join(dataset_dir, 'images')
        self.lbl_dir = os.path.join(dataset_dir, 'labels_3d')
        groups_path  = os.path.join(dataset_dir, 'frame_groups.json')

        # ── frame_groups.json 존재 여부 확인 ─────────────────
        if not os.path.isfile(groups_path):
            raise FileNotFoundError(
                f"\n[ERROR] {groups_path} 파일이 없습니다!\n"
                f"먼저 실행하세요:\n"
                f"  python build_frame_groups.py your.bag --dataset_dir {dataset_dir}"
            )

        with open(groups_path) as f:
            all_groups = json.load(f)

        # ── train / val 분리 ─────────────────────────────────
        n_val   = max(1, int(len(all_groups) * val_ratio))
        n_train = len(all_groups) - n_val
        self.groups = all_groups[:n_train] if split == 'train' else all_groups[n_train:]

        print(f"[MoraiDataset] {split} : {len(self.groups):,} 그룹 "
              f"(각 그룹 = 6대 카메라 1묶음) 로드 완료")

    def __len__(self):
        return len(self.groups)

    def _load_image(self, stem):
        """이미지 로드 → [3, 224, 224] float tensor"""
        path    = os.path.join(self.img_dir, f"{stem}.jpg")
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rs  = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        return torch.from_numpy(img_rs).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx):
        group = self.groups[idx]
        cams  = group['cams']           # {cam_name: stem}
        label_stem = group['label_stem']

        # ── 6대 카메라 이미지 로드 ────────────────────────────
        images = torch.zeros(6, 3, IMG_SIZE, IMG_SIZE)
        for ci, cam_name in enumerate(CAM_ORDER):
            if cam_name in cams:
                images[ci] = self._load_image(cams[cam_name])

        # ── 카메라 행렬 ──────────────────────────────────────
        intrinsics = torch.zeros(6, 3, 3)
        extrinsics = torch.zeros(6, 4, 4)
        for ci, cam_name in enumerate(CAM_ORDER):
            intrinsics[ci] = torch.from_numpy(_INTRINSICS[cam_name])
            extrinsics[ci] = torch.from_numpy(_EXTRINSICS[cam_name])

        # ── 라벨 로드 (cam_front 기준, Ego 좌표계) ───────────
        lbl_path = os.path.join(self.lbl_dir, f"{label_stem}.txt")
        boxes, labels = [], []

        if os.path.isfile(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 12:
                        continue
                    cls_id = int(float(parts[0]))
                    vals   = list(map(float, parts[1:]))
                    boxes.append(vals)
                    labels.append(cls_id)

        if boxes:
            gt_boxes  = torch.tensor(boxes,  dtype=torch.float32)
            gt_labels = torch.tensor(labels, dtype=torch.long)
        else:
            gt_boxes  = torch.zeros((0, 11), dtype=torch.float32)
            gt_labels = torch.zeros((0,),    dtype=torch.long)

        return {
            'images':            images,
            'intrinsics':        intrinsics,
            'extrinsics':        extrinsics,
            'dynamic_gt_boxes':  gt_boxes,
            'dynamic_gt_labels': gt_labels,
            'stem':              label_stem,
        }


# ===========================================================
# collate_fn
# ===========================================================
def morai_collate_fn(batch):
    return {
        'images':            torch.stack([b['images']     for b in batch]),
        'intrinsics':        torch.stack([b['intrinsics'] for b in batch]),
        'extrinsics':        torch.stack([b['extrinsics'] for b in batch]),
        'dynamic_gt_boxes':  [b['dynamic_gt_boxes']  for b in batch],
        'dynamic_gt_labels': [b['dynamic_gt_labels'] for b in batch],
        'stem':              [b['stem'] for b in batch],
    }


# ===========================================================
# 테스트 실행
# ===========================================================
if __name__ == "__main__":
    print("🚀 MoraiDataset (6대 카메라 묶음) 테스트!\n")

    train_ds = MoraiDataset(dataset_dir='./dataset', split='train')
    val_ds   = MoraiDataset(dataset_dir='./dataset', split='val')

    loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                        collate_fn=morai_collate_fn)

    for batch in loader:
        print(f"이미지 텐서 : {batch['images'].shape}")
        print(f"Intrinsic  : {batch['intrinsics'].shape}")
        print(f"Extrinsic  : {batch['extrinsics'].shape}")
        print(f"GT 박스    : {batch['dynamic_gt_boxes'][0].shape}")
        print(f"GT 라벨    : {batch['dynamic_gt_labels'][0]}")
        print(f"파일명     : {batch['stem'][0]}")

        # 카메라별 유효 슬롯 확인
        imgs = batch['images'][0]  # [6, 3, 224, 224]
        for ci, cam in enumerate(['cam_front','cam_front_left','cam_front_right',
                                   'cam_back','cam_back_left','cam_back_right']):
            filled = imgs[ci].abs().sum().item() > 0
            print(f"  {cam:20s}: {'✅ 이미지 있음' if filled else '⬜ 빈 슬롯'}")
        break

    print("\n✅ 6대 카메라 묶음 데이터 로더 정상 동작!")
