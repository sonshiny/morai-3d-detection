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
    dataset_root/
      scen01/
        images/cam_front/live_000000.jpg
        images/cam_front_left/live_000000.jpg
        images/cam_front_right/live_000000.jpg
        labels_3d/live_000000.csv
      scen02/
        ...

    __getitem__ 반환:
        images              : [3, 3, 224, 224]
        intrinsics          : [3, 3, 3]
        extrinsics          : [3, 4, 4]
        dynamic_gt_boxes    : [N, 11]  (x,y,z,ln_w,ln_l,ln_h,sin_yaw,cos_yaw,vx,vy,vz)
        dynamic_gt_labels   : [N]
        stem                : str
    """

    def __init__(self, dataset_root='./dataset', split='train', val_ratio=0.1):
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"[ERROR] dataset_root 없음: {dataset_root}")

        # scen01, scen02, ... 형태 폴더 모두 수집
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

        # (scen_dir, stem) 쌍 전체 수집
        all_items = []
        for scen_dir in scen_dirs:
            lbl_dir = os.path.join(scen_dir, 'labels_3d')
            stems = sorted([
                os.path.splitext(f)[0]
                for f in os.listdir(lbl_dir)
                if f.endswith('.csv')
            ])
            for stem in stems:
                all_items.append((scen_dir, stem))

        if not all_items:
            raise FileNotFoundError(f"[ERROR] CSV 파일이 하나도 없습니다.")

        n_val   = max(1, int(len(all_items) * val_ratio))
        n_train = len(all_items) - n_val
        self.items = all_items[:n_train] if split == 'train' else all_items[n_train:]

        scen_names = [os.path.basename(d) for d in scen_dirs]
        print(f"[MoraiDataset] 시나리오: {scen_names}")
        print(f"[MoraiDataset] 전체: {len(all_items):,} | {split}: {len(self.items):,} 프레임")

    def __len__(self):
        return len(self.items)

    def _load_image(self, scen_dir, stem, cam_name):
        path    = os.path.join(scen_dir, 'images', cam_name, f"{stem}.jpg")
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rs  = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
        return torch.from_numpy(img_rs).permute(2, 0, 1).float() / 255.0

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
        scen_dir, stem = self.items[idx]
        n_cams = len(CAM_ORDER)

        images = torch.zeros(n_cams, 3, IMG_SIZE, IMG_SIZE)
        for ci, cam_name in enumerate(CAM_ORDER):
            images[ci] = self._load_image(scen_dir, stem, cam_name)

        intrinsics = torch.zeros(n_cams, 3, 3)
        extrinsics = torch.zeros(n_cams, 4, 4)
        for ci, cam_name in enumerate(CAM_ORDER):
            intrinsics[ci] = torch.from_numpy(_INTRINSICS[cam_name])
            extrinsics[ci] = torch.from_numpy(_EXTRINSICS[cam_name])

        gt_boxes, gt_labels = self._load_labels(scen_dir, stem)

        return {
            'images':            images,
            'intrinsics':        intrinsics,
            'extrinsics':        extrinsics,
            'dynamic_gt_boxes':  gt_boxes,
            'dynamic_gt_labels': gt_labels,
            'stem':              f"{os.path.basename(scen_dir)}/{stem}",
        }


def morai_collate_fn(batch):
    return {
        'images':            torch.stack([b['images']     for b in batch]),
        'intrinsics':        torch.stack([b['intrinsics'] for b in batch]),
        'extrinsics':        torch.stack([b['extrinsics'] for b in batch]),
        'dynamic_gt_boxes':  [b['dynamic_gt_boxes']  for b in batch],
        'dynamic_gt_labels': [b['dynamic_gt_labels'] for b in batch],
        'stem':              [b['stem'] for b in batch],
    }


if __name__ == "__main__":
    ds     = MoraiDataset(dataset_root='./dataset', split='train')
    loader = DataLoader(ds, batch_size=2, shuffle=True,
                        collate_fn=morai_collate_fn, num_workers=0)
    batch  = next(iter(loader))
    print(f"images     : {batch['images'].shape}")
    print(f"intrinsics : {batch['intrinsics'].shape}")
    print(f"extrinsics : {batch['extrinsics'].shape}")
    print(f"GT boxes   : {batch['dynamic_gt_boxes'][0].shape}")
    print(f"stems      : {batch['stem']}")
    print("✅ 데이터셋 정상!")
