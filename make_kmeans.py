"""
Build SparseDrive-style detection anchors from MORAI train split labels.

This script deliberately reads only train scenarios. Validation/test labels must
not influence anchor priors.
"""
import argparse
import csv
import json
import os

import numpy as np

from morai_dataset import box_visible_in_any_camera


DEFAULT_K = 900
DEFAULT_XY_OUT = 'anchor_kmeans_xy.npy'
DEFAULT_FULL_OUT = 'anchor_kmeans_full.npy'
DEFAULT_META_OUT = 'anchor_kmeans_meta.json'


def list_scenarios(dataset_root):
    scen_names = sorted([
        d for d in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, d, 'labels_3d'))
    ])
    if not scen_names:
        raise FileNotFoundError(
            f"[ERROR] {dataset_root} 아래에 labels_3d 폴더를 가진 시나리오가 없습니다."
        )
    return scen_names


def resolve_val_scenarios(dataset_root, val_scenarios):
    scen_names = list_scenarios(dataset_root)
    if val_scenarios is None:
        n_val = min(5, len(scen_names))
        return scen_names[-n_val:]

    val_scenarios = list(val_scenarios)
    unknown = [name for name in val_scenarios if name not in scen_names]
    if unknown:
        raise ValueError(
            f"[ERROR] val_scenarios에 존재하지 않는 시나리오: {unknown}\n"
            f"  사용 가능한 시나리오: {scen_names}"
        )
    return val_scenarios


def iter_split_label_files(dataset_root, val_scenarios, split='train'):
    val_scenarios = resolve_val_scenarios(dataset_root, val_scenarios)
    scen_names = list_scenarios(dataset_root)

    if split == 'train':
        selected = [name for name in scen_names if name not in val_scenarios]
    elif split == 'val':
        selected = [name for name in scen_names if name in val_scenarios]
    else:
        raise ValueError(f"split는 train 또는 val이어야 합니다: {split}")

    if not selected:
        raise RuntimeError(
            f"[ERROR] split='{split}'에 해당하는 시나리오가 없습니다. "
            f"val_scenarios={val_scenarios}, 전체={scen_names}"
        )

    for scen_name in selected:
        label_dir = os.path.join(dataset_root, scen_name, 'labels_3d')
        for file_name in sorted(os.listdir(label_dir)):
            if file_name.endswith('.csv'):
                yield os.path.join(label_dir, file_name)


def collect_boxes(dataset_root, val_scenarios):
    boxes = []
    for path in iter_split_label_files(dataset_root, val_scenarios, split='train'):
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                boxes.append([
                    float(row['x']),       float(row['y']),       float(row['z']),
                    float(row['ln_w']),    float(row['ln_l']),    float(row['ln_h']),
                    float(row['sin_yaw']), float(row['cos_yaw']),
                    float(row['vx']),      float(row['vy']),      float(row['vz']),
                ])
                if not box_visible_in_any_camera(boxes[-1]):
                    boxes.pop()
    return np.asarray(boxes, dtype=np.float32)


def numpy_kmeans(xy, k, seed=42, max_iter=80, tol=1e-4):
    """
    Small dependency-free K-means fallback.
    sklearn이 없는 학습 컨테이너에서도 anchor 생성을 계속 진행하기 위함.
    """
    rng = np.random.default_rng(seed)
    xy = np.asarray(xy, dtype=np.float32)
    n = xy.shape[0]

    if n < k:
        raise RuntimeError(f"sample 수({n:,}) < K({k})")

    centers = np.empty((k, xy.shape[1]), dtype=np.float32)
    first = int(rng.integers(n))
    centers[0] = xy[first]
    closest_dist_sq = np.sum((xy - centers[0]) ** 2, axis=1)

    for center_idx in range(1, k):
        total = float(closest_dist_sq.sum())
        if total <= 1e-12:
            pick = int(rng.integers(n))
        else:
            pick = int(rng.choice(n, p=closest_dist_sq / total))
        centers[center_idx] = xy[pick]
        dist_sq = np.sum((xy - centers[center_idx]) ** 2, axis=1)
        closest_dist_sq = np.minimum(closest_dist_sq, dist_sq)

    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dist_sq = np.sum((xy[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dist_sq, axis=1)

        counts = np.bincount(new_labels, minlength=k).astype(np.float32)
        new_centers = np.zeros_like(centers)
        np.add.at(new_centers, new_labels, xy)

        nonempty = counts > 0
        new_centers[nonempty] /= counts[nonempty, None]

        empty = np.where(~nonempty)[0]
        if len(empty) > 0:
            farthest = np.argsort(dist_sq[np.arange(n), new_labels])[-len(empty):]
            new_centers[empty] = xy[farthest]

        shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
        centers = new_centers.astype(np.float32)
        labels = new_labels
        if shift < tol:
            break

    dist_sq = np.sum((xy[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    labels = np.argmin(dist_sq, axis=1)
    return labels, centers.astype(np.float32)


def run_kmeans(xy, k, seed=42):
    try:
        from sklearn.cluster import KMeans
    except ModuleNotFoundError:
        print("[make_kmeans] sklearn 없음 → NumPy fallback K-means 사용")
        return numpy_kmeans(xy, k=k, seed=seed)

    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(xy)
    return labels.astype(np.int64), km.cluster_centers_.astype(np.float32)


def build_kmeans_anchors(boxes, k=DEFAULT_K, seed=42):

    if len(boxes) == 0:
        raise RuntimeError("GT 박스가 0개입니다. dataset_root와 split을 확인하세요.")
    if len(boxes) < k:
        raise RuntimeError(
            f"GT 박스 수({len(boxes):,}) < K({k}). K를 낮추거나 데이터를 늘리세요."
        )
    if len(boxes) < k * 5:
        print(
            f"[make_kmeans] WARNING: GT 수({len(boxes):,})가 K*5({k*5:,})보다 적습니다. "
            "클러스터 일부가 노이즈에 끌릴 수 있습니다."
        )

    xy = boxes[:, :2]
    print(f"[make_kmeans] GT 총 {len(boxes):,}개 | K={k}")
    print(f"  x range: {xy[:, 0].min():.2f} ~ {xy[:, 0].max():.2f}")
    print(f"  y range: {xy[:, 1].min():.2f} ~ {xy[:, 1].max():.2f}")

    labels, centers_xy = run_kmeans(xy, k=k, seed=seed)

    global_mean = boxes.mean(axis=0).astype(np.float32)
    anchors_full = np.zeros((k, 11), dtype=np.float32)
    anchors_full[:, :2] = centers_xy

    for cluster_idx in range(k):
        cluster_boxes = boxes[labels == cluster_idx]
        if len(cluster_boxes) == 0:
            mean_box = global_mean.copy()
        else:
            mean_box = cluster_boxes.mean(axis=0).astype(np.float32)

        anchors_full[cluster_idx] = mean_box
        anchors_full[cluster_idx, :2] = centers_xy[cluster_idx]

        yaw = anchors_full[cluster_idx, 6:8]
        yaw_norm = np.linalg.norm(yaw)
        if yaw_norm < 1e-6:
            anchors_full[cluster_idx, 6] = -1.0
            anchors_full[cluster_idx, 7] = 0.0
        else:
            anchors_full[cluster_idx, 6:8] = yaw / yaw_norm

        # Velocity priors are not stable enough for static anchors.
        anchors_full[cluster_idx, 8:11] = 0.0

    return centers_xy, anchors_full


def anchors_file_is_valid(path, shape):
    if not os.path.isfile(path):
        return False
    try:
        arr = np.load(path)
    except Exception:
        return False
    return arr.shape == shape


def metadata_is_valid(path, dataset_root, val_scenarios, k):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception:
        return False

    resolved_val = resolve_val_scenarios(dataset_root, val_scenarios)
    return (
        meta.get('k') == k and
        meta.get('val_scenarios') == resolved_val
    )


def ensure_kmeans_files(
    dataset_root='/data/dataset',
    val_scenarios=None,
    k=DEFAULT_K,
    xy_out=DEFAULT_XY_OUT,
    full_out=DEFAULT_FULL_OUT,
    meta_out=DEFAULT_META_OUT,
    seed=42,
    force=False,
):
    xy_ok = anchors_file_is_valid(xy_out, (k, 2))
    full_ok = anchors_file_is_valid(full_out, (k, 11))
    meta_ok = metadata_is_valid(meta_out, dataset_root, val_scenarios, k)
    if xy_ok and full_ok and meta_ok and not force:
        print(f"[make_kmeans] 기존 anchor 사용: {xy_out}, {full_out}")
        return xy_out, full_out

    resolved_val = resolve_val_scenarios(dataset_root, val_scenarios)
    scen_names = list_scenarios(dataset_root)
    train_scenarios = [name for name in scen_names if name not in resolved_val]
    boxes = collect_boxes(dataset_root, val_scenarios)
    centers_xy, anchors_full = build_kmeans_anchors(boxes, k=k, seed=seed)

    np.save(xy_out, centers_xy.astype(np.float32))
    np.save(full_out, anchors_full.astype(np.float32))
    with open(meta_out, 'w', encoding='utf-8') as f:
        json.dump({
            'k': k,
            'seed': seed,
            'val_scenarios': resolved_val,
            'train_scenarios': train_scenarios,
            'num_train_boxes': int(len(boxes)),
        }, f, indent=2, ensure_ascii=False)
    print(f"[make_kmeans] 저장 완료: {xy_out} shape={centers_xy.shape}")
    print(f"[make_kmeans] 저장 완료: {full_out} shape={anchors_full.shape}")
    print(f"[make_kmeans] 저장 완료: {meta_out}")
    print(f"  center x: {centers_xy[:, 0].min():.2f} ~ {centers_xy[:, 0].max():.2f}")
    print(f"  center y: {centers_xy[:, 1].min():.2f} ~ {centers_xy[:, 1].max():.2f}")
    return xy_out, full_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default='/data/dataset')
    parser.add_argument(
        '--val-scenarios', nargs='*', default=None,
        help='val로 뺄 시나리오 이름. 비우면 알파벳 마지막 5개. '
             'train.py의 VAL_SCENARIOS와 반드시 일치시킬 것.',
    )
    parser.add_argument('--k', type=int, default=DEFAULT_K)
    parser.add_argument('--out', default=DEFAULT_XY_OUT)
    parser.add_argument('--full-out', default=DEFAULT_FULL_OUT)
    parser.add_argument('--meta-out', default=DEFAULT_META_OUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    ensure_kmeans_files(
        dataset_root=args.dataset_root,
        val_scenarios=args.val_scenarios,
        k=args.k,
        xy_out=args.out,
        full_out=args.full_out,
        meta_out=args.meta_out,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
