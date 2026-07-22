"""
Build SparseDrive-style detection anchors from MORAI train split labels.

This script deliberately reads only train scenarios. Validation/test labels must
not influence anchor priors.
"""
import argparse
import csv
import hashlib
import json
import os

import numpy as np

from morai_dataset import box_visible_in_any_camera


DEFAULT_K = 900
DEFAULT_XY_OUT = 'anchor_kmeans_xy.npy'
DEFAULT_FULL_OUT = 'anchor_kmeans_full.npy'
DEFAULT_META_OUT = 'anchor_kmeans_meta.json'
DEFAULT_SPLIT_OUT = 'dataset_split_v11.json'
DEFAULT_SCENARIOS = [f'scen{i:02d}' for i in range(1, 61)]
DEFAULT_VAL_SCENARIOS = [
    'scen08', 'scen18', 'scen23', 'scen30',
    'scen32', 'scen33', 'scen41', 'scen54',
]
# 기본 label 디렉터리(하위호환). GT 버전별 anchor 를 만들 땐 label_dir_name 인자 또는
# ANCHOR_GT_VERSION / --gt-version 으로 명시한다(task D: anchor 는 어떤 GT 로 만들었는지
# 기록·격리돼야 한다).
LABEL_DIR_NAME = 'labels_3d_v2'


def label_dir_for_gt(gt_version):
    """GT 버전 문자열 -> label 디렉터리 이름."""
    gv = str(gt_version).lower()
    if gv in ('v2', 'labels_3d_v2'):
        return 'labels_3d_v2'
    if gv in ('v3', 'labels_3d_v3'):
        return 'labels_3d_v3'
    raise ValueError(f"알 수 없는 ANCHOR_GT_VERSION: {gt_version} (v2 또는 v3)")


def resolve_label_dir_name(label_dir_name=None, gt_version=None):
    """label_dir(직접) > gt_version > env ANCHOR_GT_VERSION > 기본 labels_3d_v2 순으로 결정."""
    if label_dir_name is not None:
        return label_dir_name
    if gt_version is not None:
        return label_dir_for_gt(gt_version)
    env_gv = os.environ.get('ANCHOR_GT_VERSION')
    if env_gv:
        return label_dir_for_gt(env_gv)
    return LABEL_DIR_NAME


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def hash_train_label_files(dataset_root, val_scenarios, label_dir_name):
    """anchor 입력이 된 train-split label 파일 집합의 순서-불변 aggregate SHA-256.
    파일명이 바뀌거나 내용이 바뀌면 hash 가 바뀐다 → 학습 시작 시 anchor 정합 검증에 사용."""
    agg = hashlib.sha256()
    n_files = 0
    for path in sorted(iter_split_label_files(
            dataset_root, val_scenarios, split='train', label_dir_name=label_dir_name)):
        rel = os.path.relpath(path, dataset_root)
        agg.update((rel + ':' + sha256_file(path) + '\n').encode())
        n_files += 1
    return agg.hexdigest(), n_files


def scenario_sort_key(name):
    if name.startswith('scen') and name[4:].isdigit():
        return int(name[4:])
    return 10**9


def list_scenarios(dataset_root, label_dir_name=LABEL_DIR_NAME):
    scen_names = [
        name for name in os.listdir(dataset_root)
        if name.startswith('scen') and name[4:].isdigit()
        if os.path.isdir(os.path.join(dataset_root, name, label_dir_name))
    ]
    scen_names = sorted(scen_names, key=scenario_sort_key)
    if not scen_names:
        raise FileNotFoundError(
            f"[ERROR] {dataset_root} 아래에 {label_dir_name} 폴더를 가진 scen*이 없습니다."
        )
    return scen_names


def resolve_val_scenarios(dataset_root, val_scenarios, label_dir_name=LABEL_DIR_NAME):
    scen_names = list_scenarios(dataset_root, label_dir_name=label_dir_name)
    if val_scenarios is None:
        return list(DEFAULT_VAL_SCENARIOS)

    val_scenarios = list(val_scenarios)
    unknown = [name for name in val_scenarios if name not in scen_names]
    if unknown:
        raise ValueError(
            f"[ERROR] val_scenarios에 존재하지 않는 시나리오: {unknown}\n"
            f"  사용 가능한 시나리오: {scen_names}"
        )
    return val_scenarios


def iter_split_label_files(dataset_root, val_scenarios, split='train',
                           label_dir_name=LABEL_DIR_NAME):
    val_scenarios = resolve_val_scenarios(dataset_root, val_scenarios,
                                          label_dir_name=label_dir_name)
    scen_names = list_scenarios(dataset_root, label_dir_name=label_dir_name)

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
        label_dir = os.path.join(dataset_root, scen_name, label_dir_name)
        for file_name in sorted(os.listdir(label_dir)):
            if file_name.endswith('.csv'):
                yield os.path.join(label_dir, file_name)


def row_to_box_v2(row, z_mode='center'):
    w = float(row['w'])
    l = float(row['l'])
    h = float(row['h'])
    yaw = float(row['yaw_ego'])
    z_center = float(row['z_center'])
    if z_mode == 'center':
        z = z_center
    elif z_mode == 'bottom':
        z = z_center - h * 0.5
    else:
        raise ValueError(f"z_mode는 'center' 또는 'bottom'이어야 합니다: {z_mode}")
    return [
        float(row['x']), float(row['y']), z,
        np.log(max(w, 1e-6)), np.log(max(l, 1e-6)), np.log(max(h, 1e-6)),
        np.sin(yaw), np.cos(yaw),
        float(row['vx_ego']), float(row['vy_ego']), float(row['vz']),
    ]


def collect_boxes(dataset_root, val_scenarios, label_dir_name=LABEL_DIR_NAME):
    boxes = []
    for path in iter_split_label_files(dataset_root, val_scenarios, split='train',
                                       label_dir_name=label_dir_name):
        with open(path, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                # v3 는 시간보정이 유효한 박스만 anchor 입력으로 쓴다(correction_valid=0 제외).
                if 'correction_valid' in row and int(float(row['correction_valid'])) != 1:
                    continue
                box_center = row_to_box_v2(row, z_mode='center')
                box_bottom = row_to_box_v2(row, z_mode='bottom')
                if box_visible_in_any_camera(box_bottom):
                    boxes.append(box_center)
    return np.asarray(boxes, dtype=np.float32)


def collect_split_stats(dataset_root, scenarios, label_dir_name=LABEL_DIR_NAME):
    stats = {
        'num_scenarios': len(scenarios),
        'num_frames': 0,
        'num_nonempty_frames': 0,
        'num_boxes': 0,
        'num_vehicle_boxes': 0,
        'num_pedestrian_boxes': 0,
    }
    per_scenario = {}
    for scen_name in scenarios:
        label_dir = os.path.join(dataset_root, scen_name, label_dir_name)
        scen_stats = {
            'num_frames': 0,
            'num_nonempty_frames': 0,
            'num_boxes': 0,
            'num_vehicle_boxes': 0,
            'num_pedestrian_boxes': 0,
        }
        for file_name in sorted(os.listdir(label_dir)):
            if not file_name.endswith('.csv'):
                continue
            scen_stats['num_frames'] += 1
            has_box = False
            with open(os.path.join(label_dir, file_name), newline='', encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    has_box = True
                    cls_id = int(float(row['class_id']))
                    scen_stats['num_boxes'] += 1
                    if cls_id == 0:
                        scen_stats['num_vehicle_boxes'] += 1
                    elif cls_id == 1:
                        scen_stats['num_pedestrian_boxes'] += 1
            if has_box:
                scen_stats['num_nonempty_frames'] += 1
        per_scenario[scen_name] = scen_stats
        for key in stats:
            if key != 'num_scenarios':
                stats[key] += scen_stats[key]
    return stats, per_scenario


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


def anchor_meta_matches_run(meta, k, gt_version, train_scenarios, val_scenarios,
                            seed=None, input_label_sha256=None):
    """anchor meta 가 현재 run 의 해석된 설정과 일치하는지 검사(순수 함수, 테스트 가능).
    반환: (ok, mismatches[list[str]]). 학습 시작 fail-fast 근거.
    3-scene anchor 를 150-scene split 에 지정하면 train_scenarios/input hash 에서 걸린다."""
    m = []
    if meta.get('k') != k:
        m.append(f"k: meta={meta.get('k')} run={k}")
    if str(meta.get('gt_version')) != str(gt_version):
        m.append(f"gt_version: meta={meta.get('gt_version')} run={gt_version}")
    if list(meta.get('train_scenarios') or []) != list(train_scenarios):
        m.append(f"train_scenarios: meta={meta.get('train_scenarios')} run={list(train_scenarios)}")
    if list(meta.get('val_scenarios') or []) != list(val_scenarios):
        m.append(f"val_scenarios: meta={meta.get('val_scenarios')} run={list(val_scenarios)}")
    if seed is not None and meta.get('seed') != seed:
        m.append(f"seed: meta={meta.get('seed')} run={seed}")
    if input_label_sha256 is not None and meta.get('input_label_sha256') != input_label_sha256:
        m.append(f"input_label_sha256: meta={str(meta.get('input_label_sha256'))[:12]} "
                 f"run={str(input_label_sha256)[:12]}")
    return (len(m) == 0), m


def metadata_is_valid(path, dataset_root, val_scenarios, k, label_dir_name=LABEL_DIR_NAME,
                      seed=None):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
    except Exception:
        return False

    resolved_val = resolve_val_scenarios(dataset_root, val_scenarios,
                                         label_dir_name=label_dir_name)
    scen_names = list_scenarios(dataset_root, label_dir_name=label_dir_name)
    train_scenarios = [name for name in scen_names if name not in resolved_val]
    base_ok = (
        meta.get('k') == k and
        meta.get('label_dir') == label_dir_name and
        meta.get('scenarios') == scen_names and
        meta.get('val_scenarios') == resolved_val and
        meta.get('train_scenarios') == train_scenarios
    )
    if not base_ok:
        return False
    # stale 방지: seed 또는 입력 train-label aggregate hash 가 바뀌면 재사용하지 않는다.
    if seed is not None and meta.get('seed') != seed:
        return False
    cur_hash, _ = hash_train_label_files(dataset_root, val_scenarios, label_dir_name)
    if meta.get('input_label_sha256') != cur_hash:
        return False
    return True


def ensure_kmeans_files(
    dataset_root='/data/dataset',
    val_scenarios=None,
    k=DEFAULT_K,
    xy_out=DEFAULT_XY_OUT,
    full_out=DEFAULT_FULL_OUT,
    meta_out=DEFAULT_META_OUT,
    split_out=DEFAULT_SPLIT_OUT,
    seed=42,
    force=False,
    label_dir_name=None,
    gt_version=None,
):
    label_dir_name = resolve_label_dir_name(label_dir_name, gt_version)
    xy_ok = anchors_file_is_valid(xy_out, (k, 2))
    full_ok = anchors_file_is_valid(full_out, (k, 11))
    meta_ok = metadata_is_valid(meta_out, dataset_root, val_scenarios, k,
                                label_dir_name=label_dir_name, seed=seed)
    if xy_ok and full_ok and meta_ok and not force:
        print(f"[make_kmeans] 기존 anchor 사용: {xy_out}, {full_out}")
        return xy_out, full_out

    resolved_val = resolve_val_scenarios(dataset_root, val_scenarios,
                                         label_dir_name=label_dir_name)
    scen_names = list_scenarios(dataset_root, label_dir_name=label_dir_name)
    train_scenarios = [name for name in scen_names if name not in resolved_val]
    input_label_sha256, n_input_files = hash_train_label_files(
        dataset_root, val_scenarios, label_dir_name)
    boxes = collect_boxes(dataset_root, val_scenarios, label_dir_name=label_dir_name)
    centers_xy, anchors_full = build_kmeans_anchors(boxes, k=k, seed=seed)
    train_stats, train_per_scenario = collect_split_stats(
        dataset_root, train_scenarios, label_dir_name=label_dir_name)
    val_stats, val_per_scenario = collect_split_stats(
        dataset_root, resolved_val, label_dir_name=label_dir_name)
    all_stats, all_per_scenario = collect_split_stats(
        dataset_root, scen_names, label_dir_name=label_dir_name)

    # 원자적 저장 후 산출물 SHA-256 기록(학습 시작 시 fail-fast 검증용).
    np.save(xy_out, centers_xy.astype(np.float32))
    np.save(full_out, anchors_full.astype(np.float32))
    anchor_full_sha256 = sha256_file(full_out)
    anchor_xy_sha256 = sha256_file(xy_out)
    meta = {
        'k': k,
        'seed': seed,
        'label_dir': label_dir_name,
        'gt_version': ('v3' if label_dir_name == 'labels_3d_v3'
                       else 'v2' if label_dir_name == 'labels_3d_v2' else label_dir_name),
        'dataset_root': os.path.abspath(dataset_root),
        'scenarios': scen_names,
        'train_scenarios': train_scenarios,
        'val_scenarios': resolved_val,
        'input_label_sha256': input_label_sha256,
        'n_input_label_files': n_input_files,
        'anchor_full_sha256': anchor_full_sha256,
        'anchor_xy_sha256': anchor_xy_sha256,
        'anchor_full_file': os.path.basename(full_out),
        'anchor_xy_file': os.path.basename(xy_out),
        'stats': {
            'all': all_stats,
            'train': train_stats,
            'val': val_stats,
        },
        'num_train_boxes_visible': int(len(boxes)),
    }
    with open(meta_out, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    with open(split_out, 'w', encoding='utf-8') as f:
        json.dump({
            'dataset_root': dataset_root,
            'label_dir': label_dir_name,
            'scenarios': scen_names,
            'train_scenarios': train_scenarios,
            'val_scenarios': resolved_val,
            'input_label_sha256': input_label_sha256,
            'stats': {
                'all': all_stats,
                'train': train_stats,
                'val': val_stats,
            },
            'per_scenario': all_per_scenario,
            'train_per_scenario': train_per_scenario,
            'val_per_scenario': val_per_scenario,
        }, f, indent=2, ensure_ascii=False)
    print(f"[make_kmeans] 저장 완료: {xy_out} shape={centers_xy.shape}")
    print(f"[make_kmeans] 저장 완료: {full_out} shape={anchors_full.shape} sha={anchor_full_sha256[:16]}")
    print(f"[make_kmeans] 저장 완료: {meta_out}")
    print(f"[make_kmeans] 저장 완료: {split_out}")
    print(f"  label_dir: {label_dir_name} | input_label_sha={input_label_sha256[:16]} ({n_input_files} files)")
    print(f"  split: train={len(train_scenarios)} scen({train_scenarios}) / val={len(resolved_val)} scen({resolved_val})")
    print(f"  train boxes(raw/visible): {train_stats['num_boxes']:,} / {len(boxes):,}")
    print(f"  val boxes(raw): {val_stats['num_boxes']:,}")
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
    parser.add_argument('--split-out', default=DEFAULT_SPLIT_OUT)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true')
    parser.add_argument(
        '--label-dir', default=None,
        help="anchor 입력 label 디렉터리(예: labels_3d_v2, labels_3d_v3). "
             "--gt-version 보다 우선. 미지정 시 --gt-version/ANCHOR_GT_VERSION/기본 v2.")
    parser.add_argument(
        '--gt-version', default=None, choices=['v2', 'v3'],
        help="anchor 를 어떤 GT 버전에서 만들지 명시(v2->labels_3d_v2, v3->labels_3d_v3).")
    args = parser.parse_args()

    ensure_kmeans_files(
        dataset_root=args.dataset_root,
        val_scenarios=args.val_scenarios,
        k=args.k,
        xy_out=args.out,
        full_out=args.full_out,
        meta_out=args.meta_out,
        split_out=args.split_out,
        seed=args.seed,
        force=args.force,
        label_dir_name=args.label_dir,
        gt_version=args.gt_version,
    )


if __name__ == "__main__":
    main()
