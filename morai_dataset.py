import os
import csv
import json
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler

from camera_configs import INTRINSICS as _INTRINSICS, EXTRINSICS as _EXTRINSICS, CAM_ORDER

IMG_WIDTH = 704
IMG_HEIGHT = 256
IMG_SIZE = IMG_HEIGHT  # legacy alias; do not use for new resize code.
ORIG_IMG_WIDTH = 1600
ORIG_IMG_HEIGHT = 900
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
DEFAULT_SPLIT_FILE = "dataset_split_v11.json"
DEFAULT_SCENARIOS = [f"scen{i:02d}" for i in range(1, 61)]
DEFAULT_VAL_SCENARIOS = [
    "scen08", "scen18", "scen23", "scen30",
    "scen32", "scen33", "scen41", "scen54",
]
DEPTH_STRIDES = (4, 8, 16)


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


def _scenario_sort_key(name):
    if name.startswith("scen") and name[4:].isdigit():
        return int(name[4:])
    return 10**9


def _discover_scenarios(dataset_root, label_dir_name):
    if not os.path.isdir(dataset_root):
        return []
    names = []
    for name in os.listdir(dataset_root):
        if not (name.startswith("scen") and name[4:].isdigit()):
            continue
        if os.path.isdir(os.path.join(dataset_root, name, label_dir_name)):
            names.append(name)
    return sorted(names, key=_scenario_sort_key)


def _load_split_file(dataset_root, split_file=DEFAULT_SPLIT_FILE):
    path = split_file
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if not os.path.isfile(path):
        path = os.path.join(dataset_root, split_file)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_split_names(dataset_root, split, val_scenarios=None, label_dir_name="labels_3d"):
    if split not in ("train", "val"):
        raise ValueError(f"split는 'train' 또는 'val'이어야 합니다: {split}")

    split_cfg = _load_split_file(dataset_root)
    if split_cfg is not None:
        default_val = list(split_cfg.get("val_scenarios", DEFAULT_VAL_SCENARIOS))
    else:
        default_val = list(DEFAULT_VAL_SCENARIOS)

    available = _discover_scenarios(dataset_root, label_dir_name)
    scenarios = available
    missing = [name for name in scenarios if name not in available]
    if missing:
        raise FileNotFoundError(
            f"[ERROR] 확정 시나리오 중 {label_dir_name} 누락: {missing}"
        )
    if not available:
        raise FileNotFoundError(
            f"[ERROR] {dataset_root} 아래에 {label_dir_name} 폴더를 가진 확정 시나리오가 없습니다."
        )

    val_names = default_val if val_scenarios is None else list(val_scenarios)
    unknown = [name for name in val_names if name not in available]
    if unknown:
        raise ValueError(
            f"[ERROR] val_scenarios에 존재하지 않는 시나리오: {unknown}\n"
            f"  사용 가능한 시나리오: {available}"
        )

    if split == "train":
        selected = [name for name in available if name not in val_names]
    else:
        selected = [name for name in available if name in val_names]
    if not selected:
        raise RuntimeError(
            f"[ERROR] split='{split}'에 해당하는 시나리오가 없습니다. val_scenarios={val_names}"
        )
    return selected, val_names


def _build_projection_mat(K, E):
    """Current MORAI camera convention projection matrix, returned as 4x4."""
    proj_cam = np.zeros((4, 4), dtype=np.float32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    proj_cam[0, 0] = cx
    proj_cam[0, 1] = -fx
    proj_cam[1, 0] = cy
    proj_cam[1, 2] = -fy
    proj_cam[2, 0] = 1.0
    proj_cam[3, 3] = 1.0
    return proj_cam @ E.astype(np.float32)


def _label_v2_to_box(row, z_mode="center"):
    w = float(row["w"])
    l = float(row["l"])
    h = float(row["h"])
    z_center = float(row["z_center"])
    if z_mode == "center":
        z = z_center
    elif z_mode == "bottom":
        z = z_center - h * 0.5
    else:
        raise ValueError(f"z_mode는 center 또는 bottom이어야 합니다: {z_mode}")
    yaw = float(row["yaw_ego"])
    return [
        float(row["x"]), float(row["y"]), z,
        float(np.log(max(w, 1e-6))), float(np.log(max(l, 1e-6))), float(np.log(max(h, 1e-6))),
        float(np.sin(yaw)), float(np.cos(yaw)),
        float(row["vx_ego"]), float(row["vy_ego"]), float(row["vz"]),
    ]


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
        dataset_root='/home/jang/dataset',
        split='train',
        val_scenarios=None,
        filter_visible=True,
    ):
        if split not in ('train', 'val'):
            raise ValueError(f"split는 'train' 또는 'val'이어야 합니다: {split}")
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"[ERROR] dataset_root 없음: {dataset_root}")

        selected_names, val_scenarios = _resolve_split_names(
            dataset_root, split, val_scenarios, label_dir_name="labels_3d"
        )
        selected = [os.path.join(dataset_root, name) for name in selected_names]

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


def _apply_photometric(img_rgb_uint8, p):
    """
    mmdet PhotoMetricDistortion 축약(순서 고정: brightness → contrast → HSV sat/hue).
    img_rgb_uint8: HxWx3 uint8 RGB. 반환 동일 형식.
    """
    img = img_rgb_uint8.astype(np.float32)
    if p["brightness"] != 0.0:
        img += p["brightness"]
    if p["contrast"] != 1.0:
        img *= p["contrast"]
    img = np.clip(img, 0, 255).astype(np.uint8)
    if p["saturation"] != 1.0 or p["hue"] != 0.0:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * p["saturation"], 0, 255)
        hsv[..., 0] = (hsv[..., 0] + p["hue"]) % 180.0
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img


class MoraiTemporalDataset(Dataset):
    """
    SparseDrive temporal migration loader.

    반환 키:
      images            [3,3,256,704]
      intrinsics        [3,3,3]
      extrinsics        [3,4,4]
      projection_mat    [3,4,4]
      T_ego2global      [4,4]
      T_global          [4,4]  alias
      T_global_inv      [4,4]
      gt_boxes          [N,11] = x,y,z_center,ln_w,ln_l,ln_h,sin_yaw,cos_yaw,vx,vy,vz
      gt_labels         [N]
      gt_track_ids      [N]
      gt_depth          list([3,H/4,W/4], [3,H/8,W/8], [3,H/16,W/16])
      seq_id, is_seq_start, timestamp, frame_id

    기존 train.py 호환 키(dynamic_gt_boxes 등)도 함께 반환한다.
    """

    def __init__(
        self,
        dataset_root="/data/dataset",
        split="train",
        val_scenarios=None,
        sequence_length=150,
        depth_strides=DEPTH_STRIDES,
        filter_visible=True,
        load_depth=True,
        photometric_aug=False,
        occlusion_min_pts=None,
        visibility_min_ratio=None,
        visibility_loss_weighting=None,
    ):
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"[ERROR] dataset_root 없음: {dataset_root}")
        self.dataset_root = dataset_root
        self.split = split
        self.sequence_length = int(sequence_length)
        self.depth_strides = tuple(int(s) for s in depth_strides)
        self.filter_visible = bool(filter_visible)
        self.load_depth = bool(load_depth)
        # [legacy] occlusion(num_lidar_pts) 라이다 단독 필터 임계값.
        # 2026-07-18 GT 제거 규칙(GT_REMOVAL_RULE)으로 교체되어 현재는 사용되지 않는다
        # (인자·env 는 하위호환으로 파싱만 유지; _load_labels_v2 의 주석 백업 참조).
        if occlusion_min_pts is None:
            occlusion_min_pts = int(os.environ.get("OCCLUSION_MIN_PTS", "0"))
        self.occlusion_min_pts = int(occlusion_min_pts)

        # GT 제거 규칙 (scen02~06 검증 완료: 오제거 0, 놓침 1.4%):
        #   제거 <=> (radial_dist > 50m)                          … 인지범위 밖
        #         or (npts == 0 AND YOLO 트랙 다수결 미검출)      … 완전가림
        # sidecar:
        #   dataset/<scen>/occlusion/<stem>.npy          [tid, npts, dist]
        #   dataset/<scen>/camera_occlusion/<stem>.npy   [tid, track_undetected_flag]
        #     (generate_camera_occlusion_sidecar.py — YOLO 트랙 다수결 사전계산)
        # 안전 규약: sidecar 없음/track 미매칭은 '미상' → 절대 제거하지 않는다.
        # env GT_REMOVAL_RULE=0 으로 전체 비활성(기존 무필터 동작).
        self.gt_removal_rule = os.environ.get("GT_REMOVAL_RULE", "1") not in ("0", "", "false", "False")
        self.gt_removal_range_m = float(os.environ.get("GT_REMOVAL_RANGE_M", "50"))

        # camera visibility(z-buffer) sidecar 소비. generate_visibility_gt.py 가 만든
        # dataset/<scen>/visibility/<stem>.npz 를 track_id 로 매칭한다.
        #   - VISIBILITY_MIN_RATIO(기본 0.0=비활성): best_visible_ratio 가 이 값 미만인
        #     object 를 GT 에서 제외(동적 상호가림으로 카메라에 거의 안 보이는 박스 필터).
        #   - VISIBILITY_LOSS_WEIGHTING(기본 off): object 별 best_visible_ratio 기반 가중치를
        #     'visibility_weight' 키로 추가(soft down-weight; 필터 대신 손실 가중용).
        # 둘 다 기본 비활성 → sidecar 를 아예 읽지 않아 기존 학습 동작과 완전히 동일하다.
        # sidecar 없음/track 미매칭(ratio 미상)은 절대 제외/감쇠하지 않는다(weight=1).
        if visibility_min_ratio is None:
            visibility_min_ratio = float(os.environ.get("VISIBILITY_MIN_RATIO", "0"))
        self.visibility_min_ratio = float(visibility_min_ratio)
        if visibility_loss_weighting is None:
            visibility_loss_weighting = os.environ.get("VISIBILITY_LOSS_WEIGHTING", "0") not in ("0", "", "false", "False")
        self.visibility_loss_weighting = bool(visibility_loss_weighting)
        self._visibility_enabled = (self.visibility_min_ratio > 0.0) or self.visibility_loss_weighting
        # weighting 시 완전 가림 박스도 0 이 아니라 이 최소가중치까지만 감쇠(학습 신호 보존).
        self.visibility_weight_floor = float(os.environ.get("VISIBILITY_WEIGHT_FLOOR", "0.1"))
        # PhotoMetric 증강: 시퀀스(seq_id)+epoch 단위로 파라미터 고정 →
        # temporal 학습에서 시퀀스 내 프레임 간 외관 일관성 유지 (SparseDrive
        # keep_consistent_seq_aug 대응). set_epoch()으로 epoch마다 파라미터 변경.
        self.photometric_aug = bool(photometric_aug)
        self._aug_epoch = 0

        selected_names, self.val_scenarios = _resolve_split_names(
            dataset_root, split, val_scenarios, label_dir_name="labels_3d_v2"
        )
        self.scenario_names = selected_names
        self.items = []
        self.scene_infos = {}

        for scen_name in selected_names:
            scen_dir = os.path.join(dataset_root, scen_name)
            info_path = os.path.join(scen_dir, "scene_info.json")
            if not os.path.isfile(info_path):
                raise FileNotFoundError(f"[ERROR] scene_info.json 없음: {info_path}")
            with open(info_path, "r", encoding="utf-8") as f:
                scene_info = json.load(f)
            frames_by_stem = {frame["stem"]: frame for frame in scene_info["frames"]}
            self.scene_infos[scen_name] = frames_by_stem

            label_dir = os.path.join(scen_dir, "labels_3d_v2")
            stems = sorted(
                os.path.splitext(f)[0]
                for f in os.listdir(label_dir)
                if f.endswith(".csv")
            )

            # segment-aware chunking: ego 텔레포트(리포지션) 경계를 넘지 않도록
            # segments.json([start,end] offset 구간, analyze_ego_teleport.py 생성)을
            # 로드해 seq_id를 segment 로컬로 자른다. 없으면 전체 1 segment(기존 동작).
            seg_path = os.path.join(scen_dir, "segments.json")
            if os.path.isfile(seg_path):
                with open(seg_path, "r", encoding="utf-8") as sf:
                    segments = json.load(sf).get("segments", [[0, len(stems) - 1]])
            else:
                segments = [[0, len(stems) - 1]]
            off2seg = {}   # offset -> (seg_idx, seg_start)
            for seg_idx, (s0, s1) in enumerate(segments):
                for off in range(int(s0), int(s1) + 1):
                    off2seg[off] = (seg_idx, int(s0))

            for offset, stem in enumerate(stems):
                if stem not in frames_by_stem:
                    raise KeyError(f"[ERROR] {scen_name}/{stem}가 scene_info.json에 없음")
                seg_idx, seg_start = off2seg.get(offset, (0, 0))
                local = offset - seg_start                       # segment 내부 오프셋
                chunk_idx = local // self.sequence_length
                seq_id = f"{scen_name}_seg{seg_idx:03d}_chunk{chunk_idx:03d}"
                self.items.append({
                    "scen_dir": scen_dir,
                    "scen_name": scen_name,
                    "stem": stem,
                    "frame_offset": offset,
                    "seg_idx": seg_idx,
                    # is_seq_start: segment 시작(local=0) 또는 chunk 시작에서 True
                    "seq_frame_idx": local % self.sequence_length,
                    "seq_id": seq_id,
                    "is_seq_start": (local % self.sequence_length) == 0,
                })

        if not self.items:
            raise FileNotFoundError(f"[ERROR] {split} split에 temporal frame이 없습니다.")
        print(
            f"[MoraiTemporalDataset:{split}] 시나리오 {selected_names} | "
            f"{len(self.items):,} 프레임 | chunk={self.sequence_length}"
        )

    def __len__(self):
        return len(self.items)

    def verify_sequence_integrity(self, speed_max=30.0):
        """같은 seq_id 인접 프레임이 dt>0 이고 ego implied speed<=speed_max 인지 검증.
        segment 배선이 ego 텔레포트를 시퀀스 안에 남기지 않았음을 보장하는 smoke test.
        위반 리스트를 반환(빈 리스트면 PASS)."""
        from collections import defaultdict
        groups = defaultdict(list)
        for it in self.items:
            groups[it["seq_id"]].append(it)
        violations = []
        for seq_id, its in groups.items():
            its = sorted(its, key=lambda x: x["seq_frame_idx"])
            for a, b in zip(its, its[1:]):
                fa = self.scene_infos[a["scen_name"]][a["stem"]]
                fb = self.scene_infos[b["scen_name"]][b["stem"]]
                dt = float(fb["timestamp"]) - float(fa["timestamp"])
                dx = float(fb["ego"]["x"]) - float(fa["ego"]["x"])
                dy = float(fb["ego"]["y"]) - float(fa["ego"]["y"])
                speed = (dx * dx + dy * dy) ** 0.5 / dt if dt > 0 else float("inf")
                if dt <= 0 or speed > speed_max:
                    violations.append((seq_id, a["stem"], b["stem"], round(dt, 4), round(speed, 1)))
        return violations

    def set_epoch(self, epoch):
        """epoch마다 시퀀스-일관 증강 파라미터를 갱신 (train loop에서 호출)."""
        self._aug_epoch = int(epoch)

    def _photometric_params(self, seq_id):
        rng = random.Random(f"{seq_id}|{self._aug_epoch}")
        return {
            "brightness": rng.uniform(-32.0, 32.0) if rng.random() < 0.5 else 0.0,
            "contrast": rng.uniform(0.5, 1.5) if rng.random() < 0.5 else 1.0,
            "saturation": rng.uniform(0.5, 1.5) if rng.random() < 0.5 else 1.0,
            "hue": rng.uniform(-18.0, 18.0) if rng.random() < 0.5 else 0.0,
        }

    def _load_image(self, scen_dir, stem, cam_name, pm=None):
        path = os.path.join(scen_dir, "images", cam_name, f"{stem}.jpg")
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise FileNotFoundError(f"[ERROR] 이미지 로드 실패: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rs = cv2.resize(img_rgb, (IMG_WIDTH, IMG_HEIGHT))
        if pm is not None:
            img_rs = _apply_photometric(img_rs, pm)
        img_t = torch.from_numpy(img_rs).permute(2, 0, 1).float() / 255.0
        return (img_t - IMG_MEAN) / IMG_STD

    def _load_occlusion(self, scen_dir, stem):
        """occlusion sidecar → {track_id: num_lidar_pts}. 없으면 빈 dict."""
        path = os.path.join(scen_dir, "occlusion", f"{stem}.npy")
        if not os.path.isfile(path):
            return {}
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return {}
        return {int(r[0]): int(r[1]) for r in arr}

    def _load_camera_occlusion(self, scen_dir, stem):
        """camera_occlusion sidecar → {track_id: yolo_track_undetected(bool)}. 없으면 빈 dict.
        generate_camera_occlusion_sidecar.py 산출. 트랙 단위라 시나리오 내 동일 flag."""
        path = os.path.join(scen_dir, "camera_occlusion", f"{stem}.npy")
        if not os.path.isfile(path):
            return {}
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return {}
        return {int(r[0]): bool(r[1] >= 0.5) for r in arr}

    def _load_visibility(self, scen_dir, stem):
        """visibility sidecar → {track_id: best_visible_ratio}. 없으면 빈 dict."""
        path = os.path.join(scen_dir, "visibility", f"{stem}.npz")
        if not os.path.isfile(path):
            return {}
        d = np.load(path, allow_pickle=True)
        if "track_id" not in d or d["track_id"].size == 0:
            return {}
        tid = d["track_id"]
        br = d["best_visible_ratio"]
        return {int(tid[i]): float(br[i]) for i in range(len(tid))}

    def _load_labels_v2(self, scen_dir, stem):
        csv_path = os.path.join(scen_dir, "labels_3d_v2", f"{stem}.csv")
        boxes, labels, track_ids, velocity_valid, vel_sources = [], [], [], [], []
        vis_weights = []
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"[ERROR] labels_3d_v2 CSV 없음: {csv_path}")

        occ = self._load_occlusion(scen_dir, stem) if self.gt_removal_rule else None
        cam_occ = self._load_camera_occlusion(scen_dir, stem) if self.gt_removal_rule else None
        vis = self._load_visibility(scen_dir, stem) if self._visibility_enabled else None

        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                box_center = _label_v2_to_box(row, z_mode="center")
                # GT 제거 규칙 (2026-07-18 확정, scen02~06 검증):
                #   1) 인지범위: radial_dist > 50m → 항상 제거
                #   2) 완전가림: (npts == 0) AND (YOLO 트랙 다수결 미검출) → 제거
                # 부분가림은 전부 유지(가중치 없음). sidecar 미상은 제거하지 않음.
                if self.gt_removal_rule:
                    _tid = int(float(row["track_id"]))
                    _rd = float(np.hypot(float(row["x"]), float(row["y"])))
                    if _rd > self.gt_removal_range_m:
                        continue
                    _npts = occ.get(_tid, -1) if occ else -1
                    _undet = cam_occ.get(_tid, False) if cam_occ else False
                    if _npts == 0 and _undet:
                        continue
                # [백업 2026-07-18] 기존 라이다 단독 필터(OCCLUSION_MIN_PTS) — 위 규칙으로 교체.
                # 되돌리려면 GT_REMOVAL_RULE=0 설정 후 아래 주석 해제 + 상단 occ 로드 조건을
                # `if self.occlusion_min_pts > 0` 로 복원.
                # if occ is not None:
                #     npts = occ.get(int(float(row["track_id"])), -1)
                #     if 0 <= npts < self.occlusion_min_pts:
                #         continue
                # camera visibility 필터/가중: ratio 미상(sidecar/track 없음)은 유지·weight=1.
                weight = 1.0
                if vis is not None:
                    ratio = vis.get(int(float(row["track_id"])), None)
                    if ratio is not None:
                        if self.visibility_min_ratio > 0.0 and ratio < self.visibility_min_ratio:
                            continue
                        if self.visibility_loss_weighting:
                            weight = max(float(ratio), self.visibility_weight_floor)
                if self.filter_visible:
                    box_bottom = _label_v2_to_box(row, z_mode="bottom")
                    if not box_visible_in_any_camera(box_bottom):
                        continue
                boxes.append(box_center)
                labels.append(int(float(row["class_id"])))
                track_ids.append(int(float(row["track_id"])))
                vel_sources.append(row.get("vel_source", "unknown"))
                velocity_valid.append(1.0)
                vis_weights.append(weight)

        if not boxes:
            return (
                torch.zeros((0, 11), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0,), dtype=torch.long),
                torch.zeros((0,), dtype=torch.float32),
                [],
                torch.zeros((0,), dtype=torch.float32),
            )
        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(track_ids, dtype=torch.long),
            torch.tensor(velocity_valid, dtype=torch.float32),
            vel_sources,
            torch.tensor(vis_weights, dtype=torch.float32),
        )

    def _load_depth_maps(self, scen_dir, stem):
        depth_maps = [
            torch.full(
                (len(CAM_ORDER), IMG_HEIGHT // stride, IMG_WIDTH // stride),
                -1.0,
                dtype=torch.float32,
            )
            for stride in self.depth_strides
        ]
        if not self.load_depth:
            return depth_maps

        for cam_idx, cam_name in enumerate(CAM_ORDER):
            path = os.path.join(scen_dir, "depth_gt", cam_name, f"{stem}.npy")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"[ERROR] depth_gt 없음: {path}")
            pts = np.load(path).astype(np.float32)
            if pts.size == 0:
                continue
            u = pts[:, 0]
            v = pts[:, 1]
            depth = np.clip(pts[:, 2], 0.1, 60.0)
            order = np.argsort(depth)[::-1]  # 먼 점 먼저, 가까운 점이 덮어씀
            u, v, depth = u[order], v[order], depth[order]
            for level_idx, stride in enumerate(self.depth_strides):
                h = IMG_HEIGHT // stride
                w = IMG_WIDTH // stride
                uu = np.floor(u / stride).astype(np.int64)
                vv = np.floor(v / stride).astype(np.int64)
                mask = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
                if not mask.any():
                    continue
                depth_maps[level_idx][cam_idx, vv[mask], uu[mask]] = torch.from_numpy(depth[mask])
        return depth_maps

    def __getitem__(self, idx):
        item = self.items[idx]
        scen_dir = item["scen_dir"]
        scen_name = item["scen_name"]
        stem = item["stem"]
        n_cams = len(CAM_ORDER)

        pm = self._photometric_params(item["seq_id"]) if self.photometric_aug else None

        images = torch.zeros(n_cams, 3, IMG_HEIGHT, IMG_WIDTH)
        intrinsics = torch.zeros(n_cams, 3, 3)
        extrinsics = torch.zeros(n_cams, 4, 4)
        projection_mat = torch.zeros(n_cams, 4, 4)
        for ci, cam_name in enumerate(CAM_ORDER):
            K = scale_intrinsic_for_input(_INTRINSICS[cam_name].astype(np.float32))
            E = _EXTRINSICS[cam_name].astype(np.float32)
            images[ci] = self._load_image(scen_dir, stem, cam_name, pm=pm)
            intrinsics[ci] = torch.from_numpy(K)
            extrinsics[ci] = torch.from_numpy(E)
            projection_mat[ci] = torch.from_numpy(_build_projection_mat(K, E))

        frame_info = self.scene_infos[scen_name][stem]
        T = torch.tensor(frame_info["T_ego2global"], dtype=torch.float32)
        T_inv = torch.linalg.inv(T)
        ego = frame_info["ego"]
        timestamp = float(frame_info["timestamp"])
        ego_pose = torch.tensor(
            [timestamp, float(ego["x"]), float(ego["y"]), float(ego["z"]), float(ego["yaw"]), 1.0],
            dtype=torch.float32,
        )

        gt_boxes, gt_labels, gt_track_ids, velocity_valid, vel_sources, visibility_weight = \
            self._load_labels_v2(scen_dir, stem)
        gt_depth = self._load_depth_maps(scen_dir, stem)
        focal = intrinsics[:, 0, 0].clone()

        # stem_full의 앞부분(‘/’ 이전)이 temporal instance bank의 context가 된다
        # (train.py _context_from_stem = split('/')[0]). ego 텔레포트 seg 경계에서
        # bank가 리셋되도록 시나리오가 아니라 '시나리오#seg_idx'를 context로 쓴다
        # (같은 시나리오라도 seg가 다르면 context가 달라 전파 차단; seg 내부는 유지).
        # 순수 시나리오명이 필요하면 batch['scenario'] 사용.
        stem_full = f"{scen_name}#{item['seg_idx']}/{stem}"
        return {
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "projection_mat": projection_mat,
            "image_wh": torch.tensor([[IMG_WIDTH, IMG_HEIGHT]] * n_cams, dtype=torch.float32),
            "focal": focal,
            "T_ego2global": T,
            "T_global": T,
            "T_global_inv": T_inv,
            "timestamp": torch.tensor(timestamp, dtype=torch.float64),
            "frame_id": torch.tensor(int(frame_info["frame_id"]), dtype=torch.long),
            "ego_pose": ego_pose,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "gt_track_ids": gt_track_ids,
            "velocity_valid": velocity_valid,
            "vel_source": vel_sources,
            # camera visibility 가중치([N], best_visible_ratio 기반). 기본은 전부 1.0
            # (VISIBILITY_LOSS_WEIGHTING off) → 손실에 영향 없음. train.py 가 opt-in 으로 사용.
            "visibility_weight": visibility_weight,
            "gt_depth": gt_depth,
            "depth_gt": gt_depth,
            "seq_id": item["seq_id"],
            "seq_frame_idx": torch.tensor(item["seq_frame_idx"], dtype=torch.long),
            "is_seq_start": torch.tensor(float(item["is_seq_start"]), dtype=torch.float32),
            "scenario": scen_name,
            "stem": stem_full,
            # 기존 train.py 호환 alias
            "dynamic_gt_boxes": gt_boxes,
            "dynamic_gt_labels": gt_labels,
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


def morai_temporal_collate_fn(batch):
    depth_levels = len(batch[0]["gt_depth"])
    return {
        "images": torch.stack([b["images"] for b in batch]),
        "intrinsics": torch.stack([b["intrinsics"] for b in batch]),
        "extrinsics": torch.stack([b["extrinsics"] for b in batch]),
        "projection_mat": torch.stack([b["projection_mat"] for b in batch]),
        "image_wh": torch.stack([b["image_wh"] for b in batch]),
        "focal": torch.stack([b["focal"] for b in batch]),
        "T_ego2global": torch.stack([b["T_ego2global"] for b in batch]),
        "T_global": torch.stack([b["T_global"] for b in batch]),
        "T_global_inv": torch.stack([b["T_global_inv"] for b in batch]),
        "timestamp": torch.stack([b["timestamp"] for b in batch]),
        "frame_id": torch.stack([b["frame_id"] for b in batch]),
        "ego_pose": torch.stack([b["ego_pose"] for b in batch]),
        "gt_boxes": [b["gt_boxes"] for b in batch],
        "gt_labels": [b["gt_labels"] for b in batch],
        "gt_track_ids": [b["gt_track_ids"] for b in batch],
        "velocity_valid": [b["velocity_valid"] for b in batch],
        "vel_source": [b["vel_source"] for b in batch],
        "visibility_weight": [b["visibility_weight"] for b in batch],
        "gt_depth": [
            torch.stack([b["gt_depth"][level] for b in batch])
            for level in range(depth_levels)
        ],
        "depth_gt": [
            torch.stack([b["depth_gt"][level] for b in batch])
            for level in range(depth_levels)
        ],
        "seq_id": [b["seq_id"] for b in batch],
        "seq_frame_idx": torch.stack([b["seq_frame_idx"] for b in batch]),
        "is_seq_start": torch.stack([b["is_seq_start"] for b in batch]),
        "scenario": [b["scenario"] for b in batch],
        "stem": [b["stem"] for b in batch],
        # 기존 train.py 호환 alias
        "dynamic_gt_boxes": [b["dynamic_gt_boxes"] for b in batch],
        "dynamic_gt_labels": [b["dynamic_gt_labels"] for b in batch],
    }


class StreamingGroupSampler(Sampler):
    """
    Batch sampler for SparseDrive-style streaming temporal training with a
    per-slot instance memory bank.

    동작 요약
    ---------
    - dataset.items 를 seq_id 로 묶어 시퀀스(청크)를 만든다. 시퀀스 내부 프레임 순서는
      frame_offset 오름차순으로 보존한다 (절대 shuffle 하지 않는다).
    - 전체 시퀀스를 batch_size(=B)개의 "트랙"(=batch slot)에 LPT(longest-processing-time)
      그리디로 분배한다. 시퀀스 길이 multiset 이 고정이므로 트랙별 총 프레임 수(=track_len)는
      epoch 과 무관하게 결정적 → __len__ 안정.
    - step s 마다 각 트랙의 s번째 프레임을 모아 배치를 만든다. 배치의 b번째 원소는 항상
      트랙 b 에서 나오므로 batch position ↔ instance bank slot 매핑이 안정적이다.
    - 한 트랙 안에서 시퀀스가 끝나고 다음 시퀀스가 시작되면, 그 프레임은 청크의 frame0
      (is_seq_start=1) 이므로 슬롯 교체 시 bank reset 이 자연스럽게 트리거된다.
    - shuffle=True(train): epoch 마다 "트랙 안에서 시퀀스 그룹의 순서"만 섞는다
      (프레임 내부 순서는 유지, 트랙 길이도 유지). shuffle=False(val): 완전 결정적.

    coverage 모드
    -------------
    - drop_uneven_tail=False (기본): 모든 프레임을 epoch당 정확히 1회 서빙. 짧은 트랙이 먼저
      끝나는 마지막 몇 스텝은 배치 크기가 B보다 작아진다(ragged tail). 누락/중복 없음.
    - drop_uneven_tail=True: 가장 짧은 트랙이 끝나는 순간 epoch 종료 → 모든 배치가 정확히 B개,
      포지션 완전 안정(긴 트랙의 tail 프레임 일부는 드롭). instance bank 활성 시 권장.

    DataLoader 연결
    ---------------
        sampler = StreamingGroupSampler(ds, batch_size=B, shuffle=True)
        loader  = DataLoader(ds, batch_sampler=sampler,
                             collate_fn=morai_temporal_collate_fn, num_workers=0)
        # 매 epoch: sampler.set_epoch(epoch)
    """

    def __init__(self, dataset, batch_size, shuffle=False, seed=0,
                 drop_uneven_tail=False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_uneven_tail = bool(drop_uneven_tail)
        self.epoch = 0

        if not hasattr(dataset, "items"):
            raise TypeError("StreamingGroupSampler는 .items를 가진 MoraiTemporalDataset이 필요합니다.")
        if self.batch_size < 1:
            raise ValueError("batch_size >= 1 이어야 합니다.")

        # seq_id -> [(frame_offset, global_idx)]  (첫 등장 순서 보존)
        seq_map = {}
        seq_order = []
        for gidx, item in enumerate(dataset.items):
            sid = item["seq_id"]
            if sid not in seq_map:
                seq_map[sid] = []
                seq_order.append(sid)
            seq_map[sid].append((item["frame_offset"], gidx))

        # 시퀀스별 프레임 인덱스(프레임 순서 보존)
        self.sequences = []  # list of (seq_id, [global_idx ...])
        for sid in seq_order:
            frames = sorted(seq_map[sid], key=lambda t: t[0])
            self.sequences.append((sid, [g for _, g in frames]))

        # LPT 트랙 길이(결정적). 그룹 배정 자체도 캐시(순서만 epoch마다 섞음).
        self._base_track_groups, self._track_len = self._assign_tracks()
        if self.drop_uneven_tail:
            self._num_batches = min(self._track_len) if self._track_len else 0
        else:
            self._num_batches = max(self._track_len) if self._track_len else 0

        total_frames = sum(len(s[1]) for s in self.sequences)
        print(
            f"[StreamingGroupSampler] B={self.batch_size} shuffle={self.shuffle} "
            f"drop_tail={self.drop_uneven_tail} | 시퀀스 {len(self.sequences)}개 "
            f"프레임 {total_frames:,} | 트랙길이 {self._track_len} | 배치 {self._num_batches}"
        )

    def _assign_tracks(self):
        """LPT 그리디로 시퀀스를 B트랙에 배정. 반환: (트랙별 그룹리스트, 트랙길이)."""
        B = self.batch_size
        order = sorted(range(len(self.sequences)),
                       key=lambda i: len(self.sequences[i][1]), reverse=True)
        track_groups = [[] for _ in range(B)]   # 각 원소: 프레임 인덱스 리스트(=한 시퀀스)
        track_len = [0] * B
        for i in order:
            frames = self.sequences[i][1]
            b = min(range(B), key=lambda k: (track_len[k], k))  # 최단 트랙(동률 시 낮은 인덱스)
            track_groups[b].append(frames)
            track_len[b] += len(frames)
        return track_groups, track_len

    def _build_tracks(self, epoch):
        """이번 epoch의 트랙(=평탄화된 프레임 인덱스 시퀀스) 생성."""
        # 그룹 배정/길이는 고정, 그룹 '순서'만 트랙 내부에서 섞는다(프레임 내부순서는 유지).
        groups = [list(track) for track in self._base_track_groups]
        if self.shuffle:
            rng = random.Random(self.seed + epoch)
            for b in range(self.batch_size):
                rng.shuffle(groups[b])
        tracks = [[gi for group in groups[b] for gi in group]
                  for b in range(self.batch_size)]
        return tracks

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        tracks = self._build_tracks(self.epoch)
        lengths = [len(t) for t in tracks]
        steps = min(lengths) if self.drop_uneven_tail else max(lengths)
        for s in range(steps):
            batch = []
            for b in range(self.batch_size):
                if s < lengths[b]:
                    batch.append(tracks[b][s])
            if batch:
                yield batch


if __name__ == "__main__":
    root = "./dataset" if os.path.isdir("./dataset") else "/data/dataset"
    ds_tr = MoraiDataset(dataset_root=root, split='train')
    ds_va = MoraiDataset(dataset_root=root, split='val')

    loader = DataLoader(ds_tr, batch_size=2, shuffle=True,
                        collate_fn=morai_collate_fn, num_workers=0)
    batch  = next(iter(loader))
    print(f"images     : {batch['images'].shape}")
    print(f"intrinsics : {batch['intrinsics'].shape}")
    print(f"extrinsics : {batch['extrinsics'].shape}")
    print(f"GT boxes   : {batch['dynamic_gt_boxes'][0].shape}")
    print(f"stems      : {batch['stem']}")
    ds_tmp = MoraiTemporalDataset(dataset_root=root, split="train")
    tmp_loader = DataLoader(ds_tmp, batch_size=2, shuffle=False,
                            collate_fn=morai_temporal_collate_fn, num_workers=0)
    tmp_batch = next(iter(tmp_loader))
    print(f"temporal images : {tmp_batch['images'].shape}")
    print(f"temporal T      : {tmp_batch['T_global'].shape}")
    print(f"temporal boxes  : {tmp_batch['gt_boxes'][0].shape}")
    print(f"temporal depth  : {[x.shape for x in tmp_batch['gt_depth']]}")
    print(f"temporal seq    : {tmp_batch['seq_id'][:2]} start={tmp_batch['is_seq_start'].tolist()}")
    print("✅ 데이터셋 정상!")
