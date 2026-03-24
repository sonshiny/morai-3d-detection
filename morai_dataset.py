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
# MoraiDataset — Detection + Static Mapping 지원
# ===========================================================
class MoraiDataset(Dataset):
    """
    수정사항: 정적 맵 GT 필드 추가
    
    __getitem__ 반환:
        images              : [6, 3, 224, 224]
        intrinsics          : [6, 3, 3]
        extrinsics          : [6, 4, 4]
        dynamic_gt_boxes    : [N, 11]        동적 객체 3D 박스
        dynamic_gt_labels   : [N]            동적 객체 클래스
        static_gt_polylines : [M, 20, 3]     정적 맵 폴리라인 (없으면 [0, 20, 3])
        static_gt_labels    : [M]            정적 맵 클래스 (없으면 [0])
        stem                : str
    """

    def __init__(self, dataset_dir='./dataset', split='train', val_ratio=0.1):
        self.img_dir    = os.path.join(dataset_dir, 'images')
        self.lbl_dir    = os.path.join(dataset_dir, 'labels_3d')
        self.static_dir = os.path.join(dataset_dir, 'labels_static')  # 정적 맵 라벨 폴더
        groups_path     = os.path.join(dataset_dir, 'frame_groups.json')

        if not os.path.isfile(groups_path):
            raise FileNotFoundError(
                f"\n[ERROR] {groups_path} 파일이 없습니다!\n"
                f"먼저 실행하세요:\n"
                f"  python build_frame_groups.py your.bag --dataset_dir {dataset_dir}"
            )

        with open(groups_path) as f:
            all_groups = json.load(f)

        n_val   = max(1, int(len(all_groups) * val_ratio))
        n_train = len(all_groups) - n_val
        self.groups = all_groups[:n_train] if split == 'train' else all_groups[n_train:]

        # 정적 맵 라벨 존재 여부 확인
        self.has_static = os.path.isdir(self.static_dir)
        if self.has_static:
            print(f"[MoraiDataset] ✅ 정적 맵 라벨 폴더 발견: {self.static_dir}")
        else:
            print(f"[MoraiDataset] ⚠️ 정적 맵 라벨 없음 (Mapping Loss = 0으로 처리)")

        print(f"[MoraiDataset] {split} : {len(self.groups):,} 그룹 로드 완료")

    def __len__(self):
        return len(self.groups)

    def _load_image(self, stem):
        path    = os.path.join(self.img_dir, f"{stem}.jpg")
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rs  = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        return torch.from_numpy(img_rs).permute(2, 0, 1).float() / 255.0

    def _load_static_labels(self, stem):
        """
        정적 맵 라벨 로드.
        라벨 형식 (labels_static/*.txt):
            class_id x0 y0 z0 x1 y1 z1 ... x19 y19 z19
            (1 + 60 = 61개 값, 공백 구분)
        
        아직 라벨 파일이 없으면 빈 텐서 반환.
        """
        POINTS_PER_LINE = 20
        
        if not self.has_static:
            return (torch.zeros((0, POINTS_PER_LINE, 2), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.long))
        
        lbl_path = os.path.join(self.static_dir, f"{stem}.txt")
        if not os.path.isfile(lbl_path):
            return (torch.zeros((0, POINTS_PER_LINE, 2), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.long))
        
        polylines, labels = [], []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 41:  # 1(class) + 20*2(x,y) = 41
                    continue
                cls_id = int(float(parts[0]))
                coords = list(map(float, parts[1:]))
                # 40개 값을 [20, 2]으로 reshape
                polyline = np.array(coords, dtype=np.float32).reshape(POINTS_PER_LINE, 2)
                polylines.append(polyline)
                labels.append(cls_id)
        
        if polylines:
            return (torch.tensor(np.array(polylines), dtype=torch.float32),
                    torch.tensor(labels, dtype=torch.long))
        else:
            return (torch.zeros((0, POINTS_PER_LINE, 2), dtype=torch.float32),
                    torch.zeros((0,), dtype=torch.long))

    def __getitem__(self, idx):
        group = self.groups[idx]
        cams  = group['cams']
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

        # ── 동적 객체 라벨 로드 ──────────────────────────────
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

        # ── 정적 맵 라벨 로드 ────────────────────────────────
        static_polylines, static_labels = self._load_static_labels(label_stem)

        return {
            'images':              images,
            'intrinsics':          intrinsics,
            'extrinsics':          extrinsics,
            'dynamic_gt_boxes':    gt_boxes,
            'dynamic_gt_labels':   gt_labels,
            'static_gt_polylines': static_polylines,
            'static_gt_labels':    static_labels,
            'stem':                label_stem,
        }


# ===========================================================
# collate_fn
# ===========================================================
def morai_collate_fn(batch):
    return {
        'images':              torch.stack([b['images']     for b in batch]),
        'intrinsics':          torch.stack([b['intrinsics'] for b in batch]),
        'extrinsics':          torch.stack([b['extrinsics'] for b in batch]),
        'dynamic_gt_boxes':    [b['dynamic_gt_boxes']    for b in batch],
        'dynamic_gt_labels':   [b['dynamic_gt_labels']   for b in batch],
        'static_gt_polylines': [b['static_gt_polylines'] for b in batch],
        'static_gt_labels':    [b['static_gt_labels']    for b in batch],
        'stem':                [b['stem'] for b in batch],
    }


# ===========================================================
# 테스트
# ===========================================================
if __name__ == "__main__":
    print("🚀 MoraiDataset (Detection + Static Mapping) 테스트!\n")

    train_ds = MoraiDataset(dataset_dir='./dataset', split='train')
    val_ds   = MoraiDataset(dataset_dir='./dataset', split='val')

    loader = DataLoader(train_ds, batch_size=2, shuffle=True,
                        collate_fn=morai_collate_fn)

    for batch in loader:
        print(f"이미지 텐서      : {batch['images'].shape}")
        print(f"Intrinsic        : {batch['intrinsics'].shape}")
        print(f"Extrinsic        : {batch['extrinsics'].shape}")
        print(f"동적 GT 박스     : {batch['dynamic_gt_boxes'][0].shape}")
        print(f"동적 GT 라벨     : {batch['dynamic_gt_labels'][0]}")
        print(f"정적 GT 폴리라인 : {batch['static_gt_polylines'][0].shape}")
        print(f"정적 GT 라벨     : {batch['static_gt_labels'][0]}")
        print(f"파일명           : {batch['stem'][0]}")
        break

    print("\n✅ Detection + Static Mapping 데이터 로더 정상 동작!")