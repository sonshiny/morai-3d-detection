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
      scen02/...

    Split 방식 (시나리오 단위 — data leakage 방지):
      - val_scenarios에 명시된 시나리오만 val, 나머지 전부 train
      - val_scenarios=None → 알파벳 정렬 마지막 1개를 자동 val

    __getitem__ 반환:
        images              : [3, 3, 224, 224]
        intrinsics          : [3, 3, 3]
        extrinsics          : [3, 4, 4]
        dynamic_gt_boxes    : [N, 11]
        dynamic_gt_labels   : [N]
        stem                : str
    """

    def __init__(self, dataset_root='./dataset', split='train', val_scenarios=None):
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
            val_scenarios = [scen_names[-1]]
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
    ds_tr = MoraiDataset(dataset_root='./dataset', split='train')
    ds_va = MoraiDataset(dataset_root='./dataset', split='val')

    loader = DataLoader(ds_tr, batch_size=2, shuffle=True,
                        collate_fn=morai_collate_fn, num_workers=0)
    batch  = next(iter(loader))
    print(f"images     : {batch['images'].shape}")
    print(f"intrinsics : {batch['intrinsics'].shape}")
    print(f"extrinsics : {batch['extrinsics'].shape}")
    print(f"GT boxes   : {batch['dynamic_gt_boxes'][0].shape}")
    print(f"stems      : {batch['stem']}")
    print("✅ 데이터셋 정상!")
