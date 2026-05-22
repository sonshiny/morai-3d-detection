"""
morai 데이터셋 train split의 모든 GT (x, y) 좌표를 모아
K-means K=900 클러스터링 → anchor_kmeans_xy.npy 갱신.

⚠️ leak 방지: train.py에서 val로 쓸 시나리오와 동일한 분할을 넘겨야
   val 시나리오의 GT가 anchor prior에 새지 않음.

사용 예:
    # 기본값(val=알파벳 마지막 1개)으로 실행 — train.py 기본과 일치
    python make_kmeans.py

    # val 시나리오를 명시
    python make_kmeans.py --val-scenarios scen11

    # K 값 변경
    python make_kmeans.py --k 900
"""
import argparse
import csv
import os
import numpy as np
from sklearn.cluster import KMeans

from morai_dataset import MoraiDataset


def collect_xy(dataset_root, val_scenarios):
    ds = MoraiDataset(
        dataset_root=dataset_root,
        split='train',
        val_scenarios=val_scenarios,
    )
    xs = []
    for scen_dir, stem in ds.items:
        path = os.path.join(scen_dir, 'labels_3d', f"{stem}.csv")
        if not os.path.isfile(path):
            continue
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                xs.append([float(row['x']), float(row['y'])])
    return np.asarray(xs, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default='./dataset')
    parser.add_argument(
        '--val-scenarios', nargs='*', default=None,
        help='val로 뺄 시나리오 이름. 비우면 알파벳 마지막 1개. '
             'train.py의 VAL_SCENARIOS와 반드시 일치시킬 것.',
    )
    parser.add_argument('--k', type=int, default=900)
    parser.add_argument('--out', default='anchor_kmeans_xy.npy')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    xs = collect_xy(args.dataset_root, args.val_scenarios)

    if len(xs) == 0:
        raise RuntimeError("GT 박스가 0개입니다. dataset_root와 split을 확인하세요.")
    if len(xs) < args.k:
        raise RuntimeError(
            f"GT 박스 수({len(xs):,}) < K({args.k}). K를 낮추거나 데이터를 늘리세요."
        )
    if len(xs) < args.k * 5:
        print(f"[make_kmeans] ⚠️  GT 수({len(xs):,})가 K×5({args.k*5:,})보다 적습니다. "
              f"클러스터 일부가 노이즈에 끌릴 수 있어요.")

    print(f"[make_kmeans] GT 총 {len(xs):,}개 | K={args.k}")
    print(f"  x range: {xs[:, 0].min():.2f} ~ {xs[:, 0].max():.2f}")
    print(f"  y range: {xs[:, 1].min():.2f} ~ {xs[:, 1].max():.2f}")

    km = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed)
    km.fit(xs)
    centers = km.cluster_centers_.astype(np.float32)

    np.save(args.out, centers)
    print(f"[make_kmeans] 저장 완료: {args.out}  shape={centers.shape}")
    print(f"  center x: {centers[:, 0].min():.2f} ~ {centers[:, 0].max():.2f}")
    print(f"  center y: {centers[:, 1].min():.2f} ~ {centers[:, 1].max():.2f}")


if __name__ == "__main__":
    main()
