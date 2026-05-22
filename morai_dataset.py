import os
import cv2
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from camera_configs import INTRINSICS as _INTRINSICS, EXTRINSICS as _EXTRINSICS, CAM_ORDER

IMG_SIZE = 224


class MoraiDataset(Dataset):
    """
    __getitem__ 반환:
        images              : [3, 3, 224, 224]
        intrinsics          : [3, 3, 3]
        extrinsics          : [3, 4, 4]
        dynamic_gt_boxes    : [N, 11]
        dynamic_gt_labels   : [N]
        static_gt_polylines : [M, 20, 2]
        static_gt_labels    : [M]
        stem                : str
    """

    def __init__(self, dataset_dir='./dataset', split='train', val_ratio=0.1):
        self.img_dir    = os.path.join(dataset_dir, 'images')
        self.lbl_dir    = os.path.join(dataset_dir, 'labels_3d')
        self.static_dir = os.path.join(dataset_dir, 'labels_static')
        groups_path     = os.path.join(dataset_dir, 'frame_groups.json')

        if not os.path.isfile(groups_path):
            raise FileNotFoundError(
                f"\n[ERROR] {groups_path} 없음!\n"
                f"먼저 실행: python build_frame_groups.py your.bag"
            )

        with open(groups_path) as f:
            all_groups = json.load(f)

        # scenario4 제외 (빈 도로, vehicle GT 거의 없음 → detection collapse 원인)
        before = len(all_groups)
        all_groups = [g for g in all_groups if g.get('bag_key') != 'scenario4']
        print(f"[MoraiDataset] scenario4 제외: {before} → {len(all_groups)} 그룹")

        n_val   = max(1, int(len(all_groups) * val_ratio))
        n_train = len(all_groups) - n_val
        self.groups = all_groups[:n_train] if split == 'train' \
                      else all_groups[n_train:]

        self.has_static = os.path.isdir(self.static_dir)
        if self.has_static:
            print(f"[MoraiDataset] ✅ 정적 맵 라벨 폴더: {self.static_dir}")
        else:
            print(f"[MoraiDataset] ⚠️  정적 맵 라벨 없음 → Map Loss=0")

        print(f"[MoraiDataset] {split}: {len(self.groups):,} 그룹")

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
        POINTS_PER_LINE = 20
        empty = (
            torch.zeros((0, POINTS_PER_LINE, 2), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )

        if not self.has_static:
            return empty

        lbl_path = os.path.join(self.static_dir, f"{stem}.txt")
        if not os.path.isfile(lbl_path):
            return empty

        polylines, labels = [], []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 41:   # 1(class) + 20*2(x,y)
                    continue
                cls_id  = int(float(parts[0]))
                coords  = list(map(float, parts[1:]))
                polyline = np.array(coords, dtype=np.float32).reshape(
                    POINTS_PER_LINE, 2
                )
                polylines.append(polyline)
                labels.append(cls_id)

        if polylines:
            return (
                torch.tensor(np.array(polylines), dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
            )
        return empty

    def __getitem__(self, idx):
        group      = self.groups[idx]
        cams       = group['cams']
        label_stem = group['label_stem']

        # 이미지
        n_cams = len(CAM_ORDER)
        images = torch.zeros(n_cams, 3, IMG_SIZE, IMG_SIZE)
        for ci, cam_name in enumerate(CAM_ORDER):
            if cam_name in cams:
                images[ci] = self._load_image(cams[cam_name])

        # 카메라 행렬 (camera_configs.py 기반)
        intrinsics = torch.zeros(n_cams, 3, 3)
        extrinsics = torch.zeros(n_cams, 4, 4)
        for ci, cam_name in enumerate(CAM_ORDER):
            intrinsics[ci] = torch.from_numpy(_INTRINSICS[cam_name])
            extrinsics[ci] = torch.from_numpy(_EXTRINSICS[cam_name])

        # 동적 GT
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

        # 정적 GT
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


if __name__ == "__main__":
    ds     = MoraiDataset(dataset_dir='./dataset', split='train')
    loader = DataLoader(ds, batch_size=2, shuffle=True,
                        collate_fn=morai_collate_fn)
    batch  = next(iter(loader))
    print(f"images     : {batch['images'].shape}")
    print(f"intrinsics : {batch['intrinsics'].shape}")
    print(f"extrinsics : {batch['extrinsics'].shape}")
    print(f"GT boxes   : {batch['dynamic_gt_boxes'][0].shape}")
    print(f"GT static  : {batch['static_gt_polylines'][0].shape}")
    print("✅ 데이터셋 정상!")