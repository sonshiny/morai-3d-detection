import os
import csv
import cv2
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
        dynamic_gt_boxes    : [N, 11]  (x,y,z,ln_w,ln_l,ln_h,sin_yaw,cos_yaw,vx,vy,vz)
        dynamic_gt_labels   : [N]
        stem                : str
    """

    def __init__(self, dataset_dir='./dataset', split='train', val_ratio=0.1):
        self.img_root = os.path.join(dataset_dir, 'images')
        self.lbl_dir  = os.path.join(dataset_dir, 'labels_3d')

        if not os.path.isdir(self.lbl_dir):
            raise FileNotFoundError(
                f"\n[ERROR] {self.lbl_dir} 없음!\n"
                f"먼저 morai_3d_live.py를 실행해 데이터를 수집하세요."
            )

        all_stems = sorted([
            os.path.splitext(f)[0]
            for f in os.listdir(self.lbl_dir)
            if f.endswith('.csv')
        ])

        if len(all_stems) == 0:
            raise FileNotFoundError(f"[ERROR] {self.lbl_dir} 에 CSV 파일이 없습니다.")

        n_val   = max(1, int(len(all_stems) * val_ratio))
        n_train = len(all_stems) - n_val
        self.stems = all_stems[:n_train] if split == 'train' else all_stems[n_train:]

        print(f"[MoraiDataset] 전체: {len(all_stems):,} | {split}: {len(self.stems):,} 프레임")

    def __len__(self):
        return len(self.stems)

    def _load_image(self, stem, cam_name):
        path    = os.path.join(self.img_root, cam_name, f"{stem}.jpg")
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rs  = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        return torch.from_numpy(img_rs).permute(2, 0, 1).float() / 255.0

    def _load_labels(self, stem):
        csv_path = os.path.join(self.lbl_dir, f"{stem}.csv")
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

    def __getitem__(self, idx):
        stem   = self.stems[idx]
        n_cams = len(CAM_ORDER)

        images = torch.zeros(n_cams, 3, IMG_SIZE, IMG_SIZE)
        for ci, cam_name in enumerate(CAM_ORDER):
            images[ci] = self._load_image(stem, cam_name)

        intrinsics = torch.zeros(n_cams, 3, 3)
        extrinsics = torch.zeros(n_cams, 4, 4)
        for ci, cam_name in enumerate(CAM_ORDER):
            intrinsics[ci] = torch.from_numpy(_INTRINSICS[cam_name])
            extrinsics[ci] = torch.from_numpy(_EXTRINSICS[cam_name])

        gt_boxes, gt_labels = self._load_labels(stem)

        return {
            'images':           images,
            'intrinsics':       intrinsics,
            'extrinsics':       extrinsics,
            'dynamic_gt_boxes':  gt_boxes,
            'dynamic_gt_labels': gt_labels,
            'stem':             stem,
        }


def morai_collate_fn(batch):
    return {
        'images':           torch.stack([b['images']     for b in batch]),
        'intrinsics':       torch.stack([b['intrinsics'] for b in batch]),
        'extrinsics':       torch.stack([b['extrinsics'] for b in batch]),
        'dynamic_gt_boxes':  [b['dynamic_gt_boxes']  for b in batch],
        'dynamic_gt_labels': [b['dynamic_gt_labels'] for b in batch],
        'stem':             [b['stem'] for b in batch],
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
    print("✅ 데이터셋 정상!")
