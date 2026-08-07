try:
    import wandb
except ImportError:  # 병렬 워커/최소 환경에서도 train.py import 가능하도록(WANDB_MODE와 무관한 안전장치)
    class _WandbStub:
        def __getattr__(self, _name):
            return lambda *a, **k: None
    wandb = _WandbStub()
import math
import os
import csv
import json
import hashlib
import time

import numpy as np

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from morai_dataset import (
    IMG_HEIGHT,
    IMG_WIDTH,
    MoraiDataset,
    MoraiTemporalDataset,
    morai_collate_fn,
    morai_temporal_collate_fn,
    StreamingGroupSampler,
)
from resnet_fpn import ResNet50_FPN, Bottleneck
from anchor_generator import NUM_ANCHORS, generate_anchors_full
from decoder import FFNDecoder
from loss_calculator import CustomLoss, ParityLoss
from make_kmeans import (
    DEFAULT_FULL_OUT,
    DEFAULT_K,
    DEFAULT_META_OUT,
    DEFAULT_VAL_SCENARIOS,
    DEFAULT_XY_OUT,
    anchor_meta_matches_run,
    ensure_kmeans_files,
    hash_train_label_files,
    label_dir_for_gt,
    list_scenarios,
    resolve_val_scenarios,
    sha256_file,
)
from sparsedrive_ops import deformable_aggregation_function, feature_maps_format
from torch.utils.data import DataLoader


# ===========================================================
# [P0-#1] 멀티스케일 샘플링 — per-point visible mask 반환
# ===========================================================
def sample_from_multiscale(features_list, grid_2d, valid_mask, N, level_logits=None):
    """
    features_list : list of [1, 256, H, W]  (각 스케일)
    grid_2d       : [1, 1, N, 2]            (normalized -1~1)
    valid_mask    : [N] bool                (depth > 0.1)
    N             : 점 개수
    반환          : [N, 256] sampled feature (invalid는 0)
    """
    combined = torch.zeros(N, 256, device=features_list[0].device)
    if level_logits is None:
        level_weights = torch.full(
            (len(features_list),),
            1.0 / len(features_list),
            device=features_list[0].device,
            dtype=features_list[0].dtype,
        )
    else:
        level_weights = torch.softmax(level_logits[:len(features_list)], dim=0)

    for level_idx, feat in enumerate(features_list):
        sampled = F.grid_sample(feat, grid_2d, align_corners=False)
        sampled = sampled.view(256, N).T            # [N, 256]
        sampled = sampled * valid_mask.float().unsqueeze(1)
        combined = combined + level_weights[level_idx] * sampled
    return combined


# ===========================================================
# [P0-#2] anchor 박스 → 5개 키포인트 (중심 + BEV 4 corner)
# ===========================================================
def generate_5_keypoints(anchors_full):
    """
    anchors_full : [N, 11] = x,y,z,ln_w,ln_l,ln_h,sin_yaw,cos_yaw,vx,vy,vz
    반환         : [N, 5, 3] = (중심, FL, FR, RL, RR)
                   각 점은 (x,y,z) ego 좌표
    """
    device = anchors_full.device
    N = anchors_full.shape[0]

    xyz = anchors_full[:, 0:3]           # [N, 3]
    w = torch.exp(anchors_full[:, 3])    # [N]  (좌우 폭)
    l = torch.exp(anchors_full[:, 4])    # [N]  (전후 길이)
    sin_y = anchors_full[:, 6]           # [N]
    cos_y = anchors_full[:, 7]           # [N]

    half_l = (l * 0.5).unsqueeze(-1)     # [N, 1]
    half_w = (w * 0.5).unsqueeze(-1)     # [N, 1]

    # BEV 평면 4 꼭짓점 (local frame: x=forward, y=left)
    corners_local = torch.stack([
        torch.cat([ half_l,  half_w], dim=-1),   # FL
        torch.cat([ half_l, -half_w], dim=-1),   # FR
        torch.cat([-half_l,  half_w], dim=-1),   # RL
        torch.cat([-half_l, -half_w], dim=-1),   # RR
    ], dim=1)                                    # [N, 4, 2]

    # yaw 회전 적용 (rel_yaw 정의는 GT와 같음)
    cos_e = cos_y.unsqueeze(-1).unsqueeze(-1)    # [N, 1, 1]
    sin_e = sin_y.unsqueeze(-1).unsqueeze(-1)    # [N, 1, 1]
    x_l = corners_local[..., 0:1]                # [N, 4, 1]
    y_l = corners_local[..., 1:2]                # [N, 4, 1]
    x_r = cos_e * x_l - sin_e * y_l              # [N, 4, 1]
    y_r = sin_e * x_l + cos_e * y_l              # [N, 4, 1]
    corners_rot = torch.cat([x_r, y_r], dim=-1)  # [N, 4, 2]

    # ego 좌표로 평행이동 + z 추가
    corners_xy = corners_rot + xyz[:, 0:2].unsqueeze(1)              # [N, 4, 2]
    corners_z  = xyz[:, 2:3].unsqueeze(1).expand(-1, 4, 1)           # [N, 4, 1]
    corners_3d = torch.cat([corners_xy, corners_z], dim=-1)          # [N, 4, 3]

    # 중심 + 4 corner = 5 keypoints
    center_3d = xyz.unsqueeze(1)                                     # [N, 1, 3]
    keypoints = torch.cat([center_3d, corners_3d], dim=1)            # [N, 5, 3]
    return keypoints


# ===========================================================
# [Deformable] 학습 가능한 키포인트 생성기
#   고정 5점 대신, anchor 기하 + instance_feature 오프셋으로
#   "어디를 봐야 할지"를 모델이 스스로 찾도록 함
# ===========================================================
class KeypointGenerator(nn.Module):
    def __init__(self, num_pts=13, hidden_dim=256):
        super().__init__()
        self.num_pts = num_pts
        self.num_base = 5   # generate_5_keypoints 재사용분 (중심1 + BEV 4corner)
        # instance_feature → 점마다 3D 오프셋 (학습으로 주목 위치 탐색)
        self.offset_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_pts * 3),
        )
        # 오프셋 0 초기화 → 처음엔 기본 키포인트와 동일하게 시작 (요구사항: NaN/발산 방지)
        nn.init.zeros_(self.offset_mlp[-1].weight)
        nn.init.zeros_(self.offset_mlp[-1].bias)

    def forward(self, anchor, instance_feature):
        """
        anchor           : [N, 11]
        instance_feature : [N, 256]
        반환             : key_points [N, num_pts, 3]
        """
        N = anchor.shape[0]
        # 기본 5점은 기존 generate_5_keypoints 그대로 재사용 (요구사항)
        base5 = generate_5_keypoints(anchor)                 # [N, 5, 3]
        center = anchor[:, 0:3].unsqueeze(1)                 # [N, 1, 3]
        # 나머지 점의 기본 위치는 anchor 중심 → 학습 오프셋이 위치를 결정
        base_rest = center.expand(-1, self.num_pts - self.num_base, 3)
        base = torch.cat([base5, base_rest], dim=1)          # [N, num_pts, 3]

        # 학습 오프셋 (처음엔 0)
        offset = self.offset_mlp(instance_feature).view(N, self.num_pts, 3)
        # 박스 크기로 스케일 → 오프셋이 박스 기하 단위가 되도록 (x=l, y=w, z=h)
        dims = torch.exp(anchor[:, 3:6])                     # [N, 3] = (w, l, h)
        scale = dims[:, [1, 0, 2]].unsqueeze(1)              # [N, 1, 3] = (l, w, h)
        key_points = base + offset * scale                   # [N, num_pts, 3]
        return key_points


# ===========================================================
# [Deformable] 논문 기반 Deformable Feature Aggregation
#   anchor/카메라/스케일/키포인트/group 별 개별 가중치로
#   멀티뷰·멀티스케일 feature를 weighted sum
# ===========================================================
class DeformableAggregation(nn.Module):
    def __init__(self, hidden_dim=256, num_groups=8, num_levels=4,
                 num_cams=3, num_pts=13, use_deformable_func=True,
                 use_camera_embed=True):
        super().__init__()
        # hidden_dim을 group으로 균등 분할해야 group별 가중이 성립
        assert hidden_dim % num_groups == 0, "hidden_dim must be divisible by num_groups"
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.num_cams = num_cams
        self.num_pts = num_pts
        self.group_dim = hidden_dim // num_groups
        self.use_deformable_func = use_deformable_func

        self.kps_generator = KeypointGenerator(num_pts=num_pts, hidden_dim=hidden_dim)
        # 11D anchor → 256D embedding (gen_sineembed 대신 단순 Linear+ReLU+LayerNorm, 요구사항)
        self.anchor_encoder = nn.Sequential(
            nn.Linear(11, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        if use_camera_embed:
            self.camera_encoder = nn.Sequential(
                nn.Linear(12, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            # official 방식: camera embedding을 더한 뒤 camera별 weight를 예측한다.
            self.weights_fc = nn.Linear(hidden_dim, num_groups * num_levels * num_pts)
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                hidden_dim, num_groups * num_cams * num_levels * num_pts
            )
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self._cuda_daf_failed = False

    @staticmethod
    def _build_projection_mats(intrinsics, extrinsics):
        """
        Our camera convention uses cam_x as depth and projects
          u = fx * (-cam_y) / cam_x + cx
          v = fy * (-cam_z) / cam_x + cy
        so the equivalent 3x4 projection is P_cam @ ego_to_cam.
        """
        C = intrinsics.shape[0]
        proj_cam = intrinsics.new_zeros(C, 3, 4)
        fx = intrinsics[:, 0, 0]
        fy = intrinsics[:, 1, 1]
        cx = intrinsics[:, 0, 2]
        cy = intrinsics[:, 1, 2]
        proj_cam[:, 0, 0] = cx
        proj_cam[:, 0, 1] = -fx
        proj_cam[:, 1, 0] = cy
        proj_cam[:, 1, 2] = -fy
        proj_cam[:, 2, 0] = 1.0
        return torch.matmul(proj_cam, extrinsics)

    def _project_keypoints(self, key_points, intrinsics, extrinsics, image_h, image_w):
        device = key_points.device
        dtype = key_points.dtype
        N, P = key_points.shape[:2]
        C = self.num_cams
        kp_homo = torch.cat(
            [key_points, torch.ones(N, P, 1, device=device, dtype=dtype)],
            dim=-1,
        ).view(N * P, 4)

        sampling_location = torch.zeros(N, P, C, 2, device=device, dtype=dtype)
        visible = torch.zeros(N, C, P, device=device, dtype=dtype)
        for c in range(C):
            E = extrinsics[c]
            K = intrinsics[c]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            pts = (E @ kp_homo.T).T
            depth = pts[:, 0]
            u = fx * (-pts[:, 1]) / (depth + 1e-6) + cx
            v = fy * (-pts[:, 2]) / (depth + 1e-6) + cy
            loc_w = (u + 0.5) / float(image_w)
            loc_h = (v + 0.5) / float(image_h)
            valid = (
                (depth > 0.1) &
                (loc_w > 0.0) & (loc_w < 1.0) &
                (loc_h > 0.0) & (loc_h < 1.0)
            )
            loc = torch.stack([loc_w, loc_h], dim=-1).view(N, P, 2)
            sampling_location[:, :, c, :] = torch.where(
                valid.view(N, P, 1),
                loc,
                torch.zeros_like(loc),
            )
            visible[:, c, :] = valid.view(N, P).to(dtype=dtype)
        return sampling_location, visible

    def _get_weights(self, instance_feature, anchor_embed, intrinsics, extrinsics):
        N = instance_feature.shape[0]
        C, L, P, G = self.num_cams, self.num_levels, self.num_pts, self.num_groups
        if self.camera_encoder is not None:
            projection = self._build_projection_mats(intrinsics, extrinsics)
            camera_embed = self.camera_encoder(projection.reshape(C, 12))
            feature = instance_feature[:, None, :] + anchor_embed[:, None, :]
            feature = feature + camera_embed[None, :, :]
            weights = self.weights_fc(feature).view(N, C, L, P, G)
        else:
            feature = instance_feature + anchor_embed
            weights = self.weights_fc(feature).view(N, C, L, P, G)
        return weights

    def _normalize_weights(self, weights, visible):
        N, C, L, P, G = weights.shape
        mask = visible.view(N, C, 1, P, 1)
        weights = weights.masked_fill(mask <= 0, -1e4)
        weights = weights.permute(0, 4, 1, 2, 3).reshape(N, G, C * L * P)
        weights = torch.softmax(weights, dim=-1)
        return weights.reshape(N, G, C, L, P).permute(0, 2, 3, 4, 1)

    def _forward_cuda_daf(self, features_list, sampling_location, weights):
        if not self.use_deformable_func or self._cuda_daf_failed:
            return None
        if not features_list[0].is_cuda:
            return None
        try:
            feature_maps = [
                feat.unsqueeze(0).contiguous()
                for feat in features_list[:self.num_levels]
            ]
            col_feats, spatial_shape, scale_start_index = feature_maps_format(feature_maps)
            sampling = sampling_location.unsqueeze(0).contiguous()
            daf_weights = weights.permute(0, 3, 1, 2, 4).unsqueeze(0).contiguous()
            return deformable_aggregation_function(
                col_feats,
                spatial_shape,
                scale_start_index,
                sampling,
                daf_weights,
            ).squeeze(0)
        except Exception as exc:
            self._cuda_daf_failed = True
            print(f"[DeformableAggregation] CUDA op 비활성화, grid_sample fallback 사용: {exc}")
            return None

    def _forward_grid_sample(self, features_list, sampling_location, visible, weights):
        N = sampling_location.shape[0]
        C, L, P, G, gd = (
            self.num_cams,
            self.num_levels,
            self.num_pts,
            self.num_groups,
            self.group_dim,
        )
        sampled = torch.zeros(
            N, C, L, P, self.hidden_dim,
            device=sampling_location.device,
            dtype=features_list[0].dtype,
        )
        for c in range(C):
            loc = sampling_location[:, :, c, :].reshape(N * P, 2)
            grid = (loc * 2.0 - 1.0).view(1, 1, N * P, 2)
            valid_f = visible[:, c, :].reshape(N * P, 1).to(dtype=features_list[0].dtype)
            for l in range(L):
                feat = features_list[l][c:c + 1]
                s = F.grid_sample(feat, grid, align_corners=False)
                s = s.view(self.hidden_dim, N * P).T * valid_f
                sampled[:, c, l, :, :] = s.view(N, P, self.hidden_dim)

        sampled_g = sampled.view(N, C, L, P, G, gd)
        fused = (sampled_g * weights.unsqueeze(-1)).sum(dim=(1, 2, 3))
        return fused.reshape(N, self.hidden_dim)

    def forward(self, instance_feature, anchor, features_list,
                intrinsics, extrinsics, image_h=IMG_HEIGHT, image_w=IMG_WIDTH):
        """
        instance_feature : [N, 256]
        anchor           : [N, 11]
        features_list    : list of [num_cams, 256, H, W]  (스케일 4개)
        intrinsics       : [num_cams, 3, 3]
        extrinsics       : [num_cams, 4, 4]
        반환             : [N, 256]
        """
        device = anchor.device
        N = anchor.shape[0]

        # 1) 학습 가능한 키포인트 (anchor 기하 + instance_feature 오프셋)
        key_points = self.kps_generator(anchor, instance_feature)        # [N, P, 3]

        # 2) anchor embedding
        anchor_embed = self.anchor_encoder(anchor)                       # [N, 256]

        # 3) camera embedding 기반 가중치 생성 + 카메라 투영
        weights = self._get_weights(instance_feature, anchor_embed, intrinsics, extrinsics)
        sampling_location, visible = self._project_keypoints(
            key_points,
            intrinsics,
            extrinsics,
            image_h,
            image_w,
        )
        weights = self._normalize_weights(weights, visible)

        # 4) official CUDA deformable aggregation 우선, 실패 시 grid_sample fallback
        fused = self._forward_cuda_daf(features_list, sampling_location, weights)
        if fused is None:
            fused = self._forward_grid_sample(
                features_list,
                sampling_location,
                visible,
                weights,
            )

        # 5) output projection + residual (instance_feature)
        return self.output_proj(fused) + instance_feature


class CameraFeatureFusion(nn.Module):
    """
    Lightweight learned camera fusion for the 3 front cameras.
    Invalid cameras receive zero weight, so anchors outside every camera stay zero.
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, camera_features, camera_visible, query_feature):
        """
        camera_features : [C, N, D]
        camera_visible  : [C, N] float/bool
        query_feature   : [N, D]
        """
        C, N, _ = camera_features.shape
        query = query_feature.unsqueeze(0).expand(C, N, -1)
        logits = self.score_net(torch.cat([camera_features, query], dim=-1)).squeeze(-1)

        visible = camera_visible.float()
        logits = logits.masked_fill(visible <= 0, -1e4)
        weights = torch.softmax(logits, dim=0) * visible
        weights = weights / weights.sum(dim=0, keepdim=True).clamp(min=1e-6)
        return (camera_features * weights.unsqueeze(-1)).sum(dim=0)


class GridMask(nn.Module):
    """
    SparseDrive 공식 GridMask(models/grid_mask.py) 축약 이식.
    학습 시 prob 확률로 이미지에 격자 마스크를 곱한다 (mode=1: 격자 밴드만 0으로).
    공식 config: use_h/use_w=True, rotate=1(→회전 없음), offset=False, ratio=0.5,
    mode=1, prob=0.7. 배치 전체에 같은 마스크를 적용한다(공식과 동일).
    """
    def __init__(self, use_h=True, use_w=True, ratio=0.5, mode=1, prob=0.7):
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def forward(self, x):
        # x: [N, C, H, W]
        if (not self.training) or np.random.rand() > self.prob:
            return x
        n, c, h, w = x.shape
        hh, ww = int(1.5 * h), int(1.5 * w)
        d = np.random.randint(2, h)
        band = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + band, hh)
                mask[s:t, :] = 0.0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + band, ww)
                mask[:, s:t] = 0.0
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h,
                    (ww - w) // 2:(ww - w) // 2 + w]
        mask_t = torch.from_numpy(mask).to(device=x.device, dtype=x.dtype)
        if self.mode == 1:
            mask_t = 1.0 - mask_t
        return x * mask_t.view(1, 1, h, w)


class DenseDepthNet(nn.Module):
    """
    SparseDrive 공식 DenseDepthNet(models/blocks.py:264-322) 이식.
    FPN 앞 num_depth_layers개 레벨(stride 4/8/16)에 1x1 conv를 얹어 dense depth를
    회귀하는 학습 전용 보조 태스크. depth = exp(conv(feat)) * focal / equal_focal.
    Loss: GT>0 픽셀만 masked L1, sum / max(1, n_pts * n_levels) * loss_weight (fp32).
    추론 시에는 호출하지 않는다(비용 0).
    """
    def __init__(self, embed_dims=256, num_depth_layers=3,
                 equal_focal=100.0, max_depth=60.0, loss_weight=0.2):
        super().__init__()
        self.equal_focal = equal_focal
        self.max_depth = max_depth
        self.loss_weight = loss_weight
        self.num_depth_layers = num_depth_layers
        self.depth_layers = nn.ModuleList([
            nn.Conv2d(embed_dims, 1, kernel_size=1) for _ in range(num_depth_layers)
        ])

    def forward(self, feature_maps, focal):
        """
        feature_maps: list of [N_cam, C, H, W] (앞 num_depth_layers개 사용)
        focal       : [N_cam] (리사이즈 반영 fx)
        반환        : list of [N_cam, H, W] depth (m)
        """
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.float()).exp()          # [N,1,H,W]
            depth = depth * (focal.float() / self.equal_focal).view(-1, 1, 1, 1)
            depths.append(depth.squeeze(1))
        return depths

    def loss(self, depth_preds, gt_depths):
        """
        depth_preds: list of [B, N_cam, H, W]
        gt_depths  : list of [B, N_cam, H, W]  (빈 픽셀 = -1, 유효 depth > 0)
        """
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.reshape(-1)
            gt = gt.to(device=pred.device).reshape(-1)
            fg_mask = torch.logical_and(gt > 0.0, torch.logical_not(torch.isnan(pred)))
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with torch.cuda.amp.autocast(enabled=False):
                error = torch.abs(pred.float() - gt.float()).sum()
                _loss = error / max(1.0, len(gt) * len(depth_preds)) * self.loss_weight
            loss = loss + _loss
        return loss


def normalize_box_sincos(box):
    sin_cos = box[..., 6:8]
    norm = sin_cos.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    sin_cos = sin_cos / norm
    return torch.cat([box[..., :6], sin_cos, box[..., 8:]], dim=-1)


def quality_centerness(det_quality):
    if det_quality is None:
        return None
    if det_quality.ndim == 1:
        return det_quality
    return det_quality[..., 0]


def rot2d_from_yaw(yaw):
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    return torch.stack([
        torch.stack([c, -s], dim=-1),
        torch.stack([s,  c], dim=-1),
    ], dim=-2)


def align_anchor_prev_to_current(prev_anchor, prev_ego_pose, cur_ego_pose, dt=None):
    """
    Convert anchors from previous ego frame into current ego frame.
    ego_pose: [timestamp, ego_x, ego_y, ego_z, ego_yaw_rad, valid]
    Anchor convention: x forward, y left, z up in ego/body frame.

    dt(초, = t_cur - t_prev)가 주어지면 공식 SparseDrive anchor_projection처럼
    rigid 변환 전에 prev 프레임에서 velocity 모션 보상(center += v * dt)을 적용한다.
    (공식: center = center - vel * (t_prev - t_cur) 후 T_temp2cur 적용 — 동일 수식)
    """
    if (
        prev_ego_pose is None or cur_ego_pose is None or
        prev_ego_pose.numel() < 6 or cur_ego_pose.numel() < 6 or
        prev_ego_pose[-1] <= 0.5 or cur_ego_pose[-1] <= 0.5
    ):
        return None

    dtype = prev_anchor.dtype
    device = prev_anchor.device
    prev_pose = prev_ego_pose.to(device=device, dtype=dtype)
    cur_pose = cur_ego_pose.to(device=device, dtype=dtype)

    prev_t = prev_pose[1:3]
    cur_t = cur_pose[1:3]
    prev_z = prev_pose[3]
    cur_z = cur_pose[3]
    prev_yaw = prev_pose[4]
    cur_yaw = cur_pose[4]

    R_prev = rot2d_from_yaw(prev_yaw)
    R_cur = rot2d_from_yaw(cur_yaw)
    R_cur_inv = R_cur.transpose(-1, -2)

    aligned = prev_anchor.clone()

    prev_xy = prev_anchor[:, 0:2]
    prev_zc = prev_anchor[:, 2]
    if dt is not None and prev_anchor.shape[-1] >= 10:
        dt_t = torch.as_tensor(dt, device=device, dtype=dtype)
        prev_xy = prev_xy + prev_anchor[:, 8:10] * dt_t
        if prev_anchor.shape[-1] >= 11:
            prev_zc = prev_zc + prev_anchor[:, 10] * dt_t

    xy_global = (R_prev @ prev_xy.unsqueeze(-1)).squeeze(-1) + prev_t
    xy_cur = (R_cur_inv @ (xy_global - cur_t).unsqueeze(-1)).squeeze(-1)
    aligned[:, 0:2] = xy_cur
    aligned[:, 2] = prev_zc + prev_z - cur_z

    yaw_prev_box = torch.atan2(prev_anchor[:, 6], prev_anchor[:, 7])
    yaw_cur_box = yaw_prev_box + prev_yaw - cur_yaw
    aligned[:, 6] = torch.sin(yaw_cur_box)
    aligned[:, 7] = torch.cos(yaw_cur_box)

    if prev_anchor.shape[-1] >= 10:
        vel_global = (R_prev @ prev_anchor[:, 8:10].unsqueeze(-1)).squeeze(-1)
        vel_cur = (R_cur_inv @ vel_global.unsqueeze(-1)).squeeze(-1)
        aligned[:, 8:10] = vel_cur

    return normalize_box_sincos(aligned)


class SparseRefinementDecoderLayer(nn.Module):
    """
    SparseDrive perception decoder를 이 코드베이스에 맞게 축약 구현한 layer.
    operation order:
      optional temp_gnn -> query self-attention -> deformable multi-view aggregation
      -> FFN -> anchor refinement
    """
    def __init__(
        self,
        hidden_dim=256,
        num_classes=2,
        num_heads=8,
        num_groups=8,
        num_levels=4,
        num_cams=3,
        num_pts=13,
        dropout=0.1,
        use_temp_gnn=False,
    ):
        super().__init__()
        self.use_temp_gnn = use_temp_gnn
        if use_temp_gnn:
            self.temp_q_norm = nn.LayerNorm(hidden_dim)
            self.temp_kv_norm = nn.LayerNorm(hidden_dim)
            self.temp_anchor_encoder = nn.Sequential(
                nn.Linear(11, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
            )
            self.temp_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            # A1-b: temporal fusion을 0에서 서서히 여는 learnable scalar gate(ReZero식).
            # 0-init 이라 warm-start 직후(gate=0)엔 temp_out 기여가 0 → A0 출력과 동일.
            # 학습되며 gate가 열려 temporal이 점진 반영. A0 checkpoint엔 이 키가 없어
            # filtered warm-start가 0-init을 그대로 유지한다.
            self.temp_gate = nn.Parameter(torch.zeros(1))
            # A1-a: sampler/bank는 유지하되 temp_gnn fusion만 완전히 끄는 런타임 스위치.
            # 학습 파라미터 아님(외부에서 set). 기본 True(=gated fusion 사용).
            self.temp_gnn_enabled = True
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.agg_norm = nn.LayerNorm(hidden_dim)
        self.deformable_agg = DeformableAggregation(
            hidden_dim=hidden_dim,
            num_groups=num_groups,
            num_levels=num_levels,
            num_cams=num_cams,
            num_pts=num_pts,
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.det_decoder = FFNDecoder(hidden_dim=hidden_dim, num_classes=num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        instance_feature,
        anchor,
        features_list,
        intrinsics,
        extrinsics,
        image_h,
        image_w,
        temp_instance_feature=None,
        temp_anchor=None,
    ):
        if (
            self.use_temp_gnn and
            self.temp_gnn_enabled and
            temp_instance_feature is not None and
            temp_anchor is not None and
            temp_instance_feature.numel() > 0 and
            temp_anchor.numel() > 0
        ):
            cur_anchor_embed = self.temp_anchor_encoder(anchor)
            temp_anchor_embed = self.temp_anchor_encoder(temp_anchor)
            q_temp = self.temp_q_norm(instance_feature + cur_anchor_embed).unsqueeze(0)
            kv_temp = self.temp_kv_norm(
                temp_instance_feature + temp_anchor_embed
            ).unsqueeze(0)
            temp_out, _ = self.temp_attn(q_temp, kv_temp, kv_temp, need_weights=False)
            # 0-init gate: gate=0 이면 A0 출력 보존, 학습되며 서서히 temporal 반영.
            instance_feature = instance_feature + self.temp_gate * self.dropout(temp_out.squeeze(0))

        q = self.attn_norm(instance_feature).unsqueeze(0)
        attn_out, _ = self.self_attn(q, q, q, need_weights=False)
        instance_feature = instance_feature + self.dropout(attn_out.squeeze(0))

        instance_feature = self.deformable_agg(
            self.agg_norm(instance_feature),
            anchor,
            features_list,
            intrinsics,
            extrinsics,
            image_h=image_h,
            image_w=image_w,
        )

        ffn_out = self.ffn(self.ffn_norm(instance_feature))
        instance_feature = instance_feature + self.dropout(ffn_out)

        det_cls, det_off, det_quality = self.det_decoder(self.out_norm(instance_feature))
        refined_anchor = normalize_box_sincos(anchor + det_off)
        return instance_feature, refined_anchor, det_cls, det_quality


# ===========================================================
# 모델
# ===========================================================
class AutoNavModel(nn.Module):
    def __init__(
        self,
        hidden_dim=256,
        num_classes=2,
        num_decoder_layers=6,
        pretrained_backbone=True,
        use_temporal_memory=True,
        num_temp_instances=600,
        temporal_confidence_decay=0.6,
        max_time_interval=2.0,
        # 기본 False: 구 스크립트(eval_distance/inference)의 v10 strict 로드 호환.
        # train.py(v11)에서 명시적으로 켠다.
        use_grid_mask=False,
        use_dense_depth=False,
    ):
        super().__init__()
        anchors_full = generate_anchors_full()
        if anchors_full.shape != (NUM_ANCHORS, 11):
            raise ValueError(f"anchors_full shape 이상: {anchors_full.shape}")

        self.num_classes = num_classes
        self.use_temporal_memory = use_temporal_memory
        self.num_temp_instances = min(int(num_temp_instances), NUM_ANCHORS)
        self.temporal_confidence_decay = temporal_confidence_decay
        self.max_time_interval = float(max_time_interval)
        self.use_grid_mask = use_grid_mask
        self.grid_mask = GridMask(prob=0.7)
        self.use_dense_depth = use_dense_depth
        self.depth_net = DenseDepthNet(
            embed_dims=256, num_depth_layers=3,
            equal_focal=100.0, max_depth=60.0, loss_weight=0.2,
        ) if use_dense_depth else None
        self.backbone = ResNet50_FPN(Bottleneck, pretrained=pretrained_backbone)
        self.decoder_layers = nn.ModuleList([
            SparseRefinementDecoderLayer(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                num_heads=8,
                num_groups=8,
                num_levels=4,
                num_cams=3,
                num_pts=13,
                dropout=0.1,
                use_temp_gnn=(layer_idx >= 1),
            )
            for layer_idx in range(num_decoder_layers)
        ])
        self.instance_feature = nn.Parameter(torch.empty(NUM_ANCHORS, hidden_dim))
        self.register_buffer('det_anchors_full', anchors_full)
        nn.init.xavier_uniform_(self.instance_feature)
        self.reset_temporal_memory()

    def freeze_backbone_bn(self):
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad_(False)

    def reset_temporal_memory(self):
        """배치 슬롯별 instance bank 상태 초기화. 슬롯 수는 첫 forward에서 확정."""
        self._bank = None   # list[dict|None] 길이 B; dict keys: feature/anchor/confidence/ego_pose/context
        # telemetry (preflight/epoch 로그용): bank 무효화 횟수, 직전 forward 의 temporal 활성 수
        self._bank_reset_count = 0
        self._last_temporal_active = 0
        self._last_batch_size = 0

    def _ensure_bank(self, batch_size):
        if self._bank is None or len(self._bank) != batch_size:
            self._bank = [None] * batch_size

    @staticmethod
    def _context_from_stem(stem):
        if stem is None:
            return None
        return str(stem).split('/')[0]

    @torch.no_grad()
    def _get_temporal_memory(self, slot, context, cur_ego_pose, cur_timestamp, device, dtype):
        """
        공식 InstanceBank.get 대응 (배치 슬롯별).
        유효 조건: 같은 시나리오(context) + 0 < dt <= max_time_interval + ego valid.
        스트리밍 샘플러에서 같은 슬롯의 연속 청크(같은 시나리오, dt~0.1s)는 전파가
        유지되고, 다른 청크/시나리오는 dt 또는 context에서 자동 리셋된다.
        dt는 반드시 float64 timestamp(cur_timestamp)로 계산한다 — ego_pose 텐서는
        float32라 유닉스 시각(1.7e9)에서 분해능이 ~256s로 무너져 dt가 0이 된다.
        반환: (temp_feature[600,C], aligned_anchor[600,11], prev_confidence[600]) 또는 (None,None,None)
        """
        entry = self._bank[slot] if (self.use_temporal_memory and self._bank is not None) else None
        if (
            entry is None or context is None or entry["context"] != context or
            cur_ego_pose is None or cur_ego_pose.numel() < 6 or cur_ego_pose[-1] <= 0.5 or
            cur_timestamp is None
        ):
            if self._bank is not None and entry is not None and entry["context"] != context:
                self._bank[slot] = None
                self._bank_reset_count += 1
            return None, None, None

        dt = float(cur_timestamp) - float(entry["timestamp"])
        if not (0.0 < dt <= self.max_time_interval):
            self._bank[slot] = None
            self._bank_reset_count += 1
            return None, None, None

        temp_feature = entry["feature"].to(device=device, dtype=dtype)
        prev_anchor = entry["anchor"].to(device=device, dtype=dtype)
        prev_ego_pose = entry["ego_pose"].to(device=device, dtype=dtype)
        aligned_anchor = align_anchor_prev_to_current(
            prev_anchor,
            prev_ego_pose,
            cur_ego_pose.to(device=device, dtype=dtype),
            dt=dt,
        )
        if aligned_anchor is None:
            self._bank[slot] = None
            self._bank_reset_count += 1
            return None, None, None
        prev_conf = entry["confidence"].to(device=device)
        return temp_feature, aligned_anchor, prev_conf

    @torch.no_grad()
    def _cache_temporal_memory(
        self,
        slot,
        instance_feature,
        anchor,
        det_cls,
        context,
        cur_ego_pose,
        cur_timestamp,
        prev_confidence,
        temporal_active,
    ):
        """
        공식 InstanceBank.cache 대응 (배치 슬롯별).
        - confidence = 최종 layer cls sigmoid max (공식과 동일하게 cls만 사용)
        - temporal 전파가 있었으면 working set 앞 num_temp 자리(=전파 인스턴스)의
          confidence를 max(prev * decay, new)로 융합
        - 900개 중 top-600을 detach해 저장 (프레임 간 gradient 차단)
        """
        if (
            not self.use_temporal_memory or
            context is None or
            self.num_temp_instances <= 0 or
            cur_ego_pose is None or
            cur_ego_pose.numel() < 6 or
            cur_ego_pose[-1] <= 0.5 or
            cur_timestamp is None
        ):
            return

        confidence = det_cls.detach().float().sigmoid().max(dim=-1).values   # [900]
        if temporal_active and prev_confidence is not None:
            n_temp = self.num_temp_instances
            confidence[:n_temp] = torch.maximum(
                prev_confidence.float() * self.temporal_confidence_decay,
                confidence[:n_temp],
            )

        topk = min(self.num_temp_instances, confidence.shape[0])
        top_conf, indices = torch.topk(confidence, k=topk, largest=True)
        self._bank[slot] = {
            "feature": instance_feature.detach()[indices].float(),
            "anchor": anchor.detach()[indices].float(),
            "confidence": top_conf,
            "ego_pose": cur_ego_pose.detach().float(),
            "timestamp": float(cur_timestamp),   # float64 유지 (dt 정밀도)
            "context": context,
        }

    def forward(
        self,
        images,
        intrinsics,
        extrinsics,
        stems=None,
        ego_poses=None,
        focal=None,
        timestamps=None,
        return_intermediate=False,
    ):
        """
        images     : [B, 3, 3, H, W]
        intrinsics : [B, 3, 3, 3] resized-input coordinate system
        extrinsics : [B, 3, 4, 4]
        focal      : [B, 3] 리사이즈 반영 fx (dense depth 보조 태스크용, 학습 시만 사용)
        timestamps : [B] float64 (temporal dt 계산용 — ego_pose의 float32 ts는
                     유닉스 시각에서 정밀도 부족. 미지정 시 ego_pose[0]로 fallback)
        반환       : dict with final and per-decoder outputs (+ 'depth_pred' 학습 시)

        Temporal(공식 SparseDrive instance bank semantics, 배치 슬롯별):
          layer0(single-frame) → update: temporal 600개가 현재 900개 중
          confidence 하위 600개를 교체(top-300 fresh만 유지) → layer1..5(temp_gnn
          cross-attn, K/V=전파 인스턴스) → cache: top-600 저장(+confidence decay 융합).
          배치 position b ↔ bank slot b 매핑은 StreamingGroupSampler가 보장한다.
        """
        B = images.shape[0]
        image_h = images.shape[-2]
        image_w = images.shape[-1]
        self._ensure_bank(B)

        # GridMask (학습 시만; 공식은 [B*N,C,H,W]에 배치 공통 마스크 적용)
        if self.use_grid_mask and self.training:
            flat = images.flatten(0, 1)                     # [B*3, 3, H, W]
            images = self.grid_mask(flat).view_as(images) if flat.numel() else images

        # 배치별 출력 누적
        batch_det_classes = []
        batch_det_boxes   = []
        batch_det_quality = []
        batch_all_classes = []
        batch_all_boxes = []
        batch_all_quality = []
        depth_preds_per_level = None

        n_fresh = NUM_ANCHORS - self.num_temp_instances     # temporal 시 유지할 fresh 수

        for b in range(B):
            context = self._context_from_stem(stems[b]) if stems is not None else None
            cam_imgs = images[b]                            # [3, 3, H, W]
            all_features = self.backbone(cam_imgs)          # list of [3, C, H, W]

            # dense depth 보조 태스크 (학습 시만, stride 4/8/16 레벨)
            if self.use_dense_depth and self.training and focal is not None:
                depths_b = self.depth_net(all_features, focal[b])
                if depth_preds_per_level is None:
                    depth_preds_per_level = [[] for _ in range(len(depths_b))]
                for li, d in enumerate(depths_b):
                    depth_preds_per_level[li].append(d)

            cur_ego_pose = ego_poses[b] if ego_poses is not None else None
            if timestamps is not None:
                cur_ts = float(timestamps[b].item())
            elif cur_ego_pose is not None and cur_ego_pose.numel() >= 1:
                cur_ts = float(cur_ego_pose[0].item())   # fallback (float32 정밀도 한계)
            else:
                cur_ts = None
            det_feat = self.instance_feature
            det_box = self.det_anchors_full
            temp_feat, temp_box, prev_conf = self._get_temporal_memory(
                b,
                context,
                cur_ego_pose,
                cur_ts,
                det_feat.device,
                det_feat.dtype,
            )
            temporal_active = temp_feat is not None
            if b == 0:
                self._last_temporal_active = 0
                self._last_batch_size = B
            self._last_temporal_active += int(temporal_active)

            det_cls = None
            det_quality = None
            layer_classes = []
            layer_boxes = []
            layer_quality = []
            for layer_idx, layer in enumerate(self.decoder_layers):
                det_feat, det_box, det_cls, det_quality = layer(
                    det_feat,
                    det_box,
                    all_features,
                    intrinsics[b],
                    extrinsics[b],
                    image_h=image_h,
                    image_w=image_w,
                    temp_instance_feature=temp_feat,
                    temp_anchor=temp_box,
                )
                layer_classes.append(det_cls)
                layer_boxes.append(det_box)
                layer_quality.append(det_quality)

                # 공식 InstanceBank.update: single-frame decoder(layer0) 직후
                # temporal 600개가 confidence 하위 600개를 교체 (top-n_fresh만 유지)
                if layer_idx == 0 and temporal_active:
                    conf0 = det_cls.detach().float().sigmoid().max(dim=-1).values   # [900]
                    _, fresh_idx = torch.topk(conf0, k=n_fresh, largest=True)
                    det_feat = torch.cat(
                        [temp_feat.to(det_feat.dtype), det_feat[fresh_idx]], dim=0)
                    det_box = torch.cat(
                        [temp_box.to(det_box.dtype), det_box[fresh_idx]], dim=0)

            batch_det_classes.append(det_cls)
            batch_det_boxes.append(det_box)
            batch_det_quality.append(det_quality)
            batch_all_classes.append(torch.stack(layer_classes))
            batch_all_boxes.append(torch.stack(layer_boxes))
            batch_all_quality.append(torch.stack(layer_quality))
            self._cache_temporal_memory(
                b,
                det_feat,
                det_box,
                det_cls,
                context,
                cur_ego_pose,
                cur_ts,
                prev_conf,
                temporal_active,
            )

        output = {
            'det_cls': torch.stack(batch_det_classes),
            'det_box': torch.stack(batch_det_boxes),
            'det_quality': torch.stack(batch_det_quality),
            'all_det_cls': torch.stack(batch_all_classes),
            'all_det_box': torch.stack(batch_all_boxes),
            'all_det_quality': torch.stack(batch_all_quality),
        }
        if depth_preds_per_level is not None:
            output['depth_pred'] = [
                torch.stack(level_list) for level_list in depth_preds_per_level
            ]                                                # list of [B, 3, H, W]
        if return_intermediate:
            return output
        return output


# ===========================================================
# Val 헬퍼 — score/class-aware Precision & Recall
# ===========================================================
CLASS_ID_NAMES = {0: "vehicle", 1: "pedestrian"}

# v10: 평가 전용 — ego 방사거리(sqrt(x^2+y^2)) 기준 3구간. [lo, hi) 반개구간.
# 55m 이상은 어느 버킷에도 집계되지 않는다(학습/체크포인트 로직과 무관, 로깅 전용).
DISTANCE_BUCKETS = ((0.0, 20.0), (20.0, 40.0), (40.0, 55.0))


def _distance_bucket_idx(dist, buckets=DISTANCE_BUCKETS):
    for i, (lo, hi) in enumerate(buckets):
        if lo <= dist < hi:
            return i
    return None


@torch.no_grad()
def bev_nms_axis_aligned(boxes, scores, labels, iou_thresh=0.3, center_dist_thresh=1.5):
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    keep_all = []
    for cls_id in labels.unique():
        cls_idx = torch.where(labels == cls_id)[0]
        cls_scores = scores[cls_idx]
        order = torch.argsort(cls_scores, descending=True)

        while order.numel() > 0:
            current_local = order[0]
            current_idx = cls_idx[current_local]
            keep_all.append(current_idx)
            if order.numel() == 1:
                break

            rest_local = order[1:]
            rest_idx = cls_idx[rest_local]

            cur = boxes[current_idx]
            rest = boxes[rest_idx]

            cur_w = torch.exp(cur[3])
            cur_l = torch.exp(cur[4])
            rest_w = torch.exp(rest[:, 3])
            rest_l = torch.exp(rest[:, 4])

            inter_x = (
                torch.minimum(cur[0] + cur_l * 0.5, rest[:, 0] + rest_l * 0.5) -
                torch.maximum(cur[0] - cur_l * 0.5, rest[:, 0] - rest_l * 0.5)
            ).clamp(min=0)
            inter_y = (
                torch.minimum(cur[1] + cur_w * 0.5, rest[:, 1] + rest_w * 0.5) -
                torch.maximum(cur[1] - cur_w * 0.5, rest[:, 1] - rest_w * 0.5)
            ).clamp(min=0)
            inter = inter_x * inter_y
            union = cur_w * cur_l + rest_w * rest_l - inter + 1e-6
            center_dist = torch.norm(rest[:, :2] - cur[:2], dim=-1)
            suppress = ((inter / union) >= iou_thresh) | (center_dist < center_dist_thresh)
            order = rest_local[~suppress]

    if not keep_all:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    return torch.stack(keep_all)


# ===========================================================
# [v9] 회전 박스 기반 BEV NMS
#   기존 bev_nms_axis_aligned는 axis-aligned IoU만 계산해서 비스듬한 차량의
#   중복 탐지를 정확히 억제하지 못했다. v9는 box[6]/box[7]로 실제 yaw를 복원해
#   BEV 상 회전 직사각형 간 IoU(Sutherland-Hodgman polygon clipping)를 계산하고,
#   center-distance suppression threshold도 2.0m로 상향한다.
# ===========================================================
def _bev_rbox_corners(boxes):
    """boxes: [N, >=8] (x,y,z,ln_w,ln_l,ln_h,sin_yaw,cos_yaw...).
    반환: BEV 4 corner [N, 4, 2] (ego 좌표, x=forward, y=left)."""
    x = boxes[:, 0]
    y = boxes[:, 1]
    half_w = torch.exp(boxes[:, 3]) * 0.5   # 좌우 폭(y축)
    half_l = torch.exp(boxes[:, 4]) * 0.5   # 전후 길이(x축)
    sin_y = boxes[:, 6]
    cos_y = boxes[:, 7]
    norm = torch.sqrt(sin_y * sin_y + cos_y * cos_y).clamp(min=1e-6)
    sin_y = sin_y / norm
    cos_y = cos_y / norm
    # local corner (x, y): FL, FR, RR, RL — 하나의 방향으로 감기는 순서
    lx = torch.stack([half_l, half_l, -half_l, -half_l], dim=-1)
    ly = torch.stack([half_w, -half_w, -half_w, half_w], dim=-1)
    cos_e = cos_y.unsqueeze(-1)
    sin_e = sin_y.unsqueeze(-1)
    gx = cos_e * lx - sin_e * ly + x.unsqueeze(-1)
    gy = sin_e * lx + cos_e * ly + y.unsqueeze(-1)
    return torch.stack([gx, gy], dim=-1)     # [N, 4, 2]


def _poly_signed_area(poly):
    """poly: [..., K, 2] → signed area [...] (shoelace)."""
    x = poly[..., 0]
    y = poly[..., 1]
    x2 = torch.roll(x, shifts=-1, dims=-1)
    y2 = torch.roll(y, shifts=-1, dims=-1)
    return 0.5 * (x * y2 - x2 * y).sum(dim=-1)


def _clip_against_edge(subj, a, b):
    """subj polygon [M, K, 2]을 CCW clip 다각형의 한 edge(a→b)로 자른다.
    interior는 edge 왼쪽. 잘려나간 vertex는 직전 유효 vertex로 중복 채워
    (면적 0 기여) 고정 크기 [M, 2K, 2]로 반환 → 완전 벡터화."""
    M, K, _ = subj.shape
    ax = a[:, 0:1]
    ay = a[:, 1:2]
    bx = b[:, 0:1]
    by = b[:, 1:2]
    ex = bx - ax
    ey = by - ay

    px = subj[..., 0]
    py = subj[..., 1]
    cross = ex * (py - ay) - ey * (px - ax)   # >=0 이면 interior
    inside = cross >= 0

    px2 = torch.roll(px, shifts=-1, dims=-1)
    py2 = torch.roll(py, shifts=-1, dims=-1)
    inside2 = torch.roll(inside, shifts=-1, dims=-1)
    cross2 = ex * (py2 - ay) - ey * (px2 - ax)

    denom = cross - cross2
    t = cross / torch.where(denom.abs() < 1e-9, torch.ones_like(denom), denom)
    ix = px + t * (px2 - px)
    iy = py + t * (py2 - py)
    crossing = inside ^ inside2

    slot1 = torch.stack([px, py], dim=-1)     # 현재 vertex (inside면 유지)
    slot2 = torch.stack([ix, iy], dim=-1)     # edge 교차점 (crossing이면 추가)
    out = torch.stack([slot1, slot2], dim=2).reshape(M, 2 * K, 2)
    valid = torch.stack([inside, crossing], dim=2).reshape(M, 2 * K)

    idx = torch.arange(2 * K, device=subj.device).view(1, -1).expand(M, -1)
    valid_idx = torch.where(valid, idx, torch.full_like(idx, -1))
    last_valid, _ = torch.cummax(valid_idx, dim=1)
    overall_last = last_valid[:, -1:].clamp(min=0)
    last_valid = torch.where(last_valid < 0, overall_last, last_valid)
    safe = last_valid.clamp(min=0)
    return torch.gather(out, 1, safe.unsqueeze(-1).expand(-1, -1, 2))


def _rotated_iou_one_to_many(cur_corners, rest_corners):
    """cur_corners [4, 2] vs rest_corners [M, 4, 2] → 회전 IoU [M]."""
    M = rest_corners.shape[0]
    if M == 0:
        return rest_corners.new_zeros(0)

    clip = rest_corners
    sa = _poly_signed_area(clip)
    clip_rev = torch.flip(clip, dims=[1])
    clip = torch.where((sa < 0).view(M, 1, 1), clip_rev, clip)   # CCW 통일

    subj = cur_corners.unsqueeze(0).expand(M, -1, -1).contiguous()
    for j in range(4):
        a = clip[:, j, :]
        b = clip[:, (j + 1) % 4, :]
        subj = _clip_against_edge(subj, a, b)

    inter = _poly_signed_area(subj).abs()
    area_cur = _poly_signed_area(cur_corners.unsqueeze(0)).abs()
    area_rest = _poly_signed_area(rest_corners).abs()
    union = area_cur + area_rest - inter
    return (inter / union.clamp(min=1e-6)).clamp(min=0.0, max=1.0)


@torch.no_grad()
def bev_nms_rotated(boxes, scores, labels, iou_thresh=0.3, center_dist_thresh=2.0):
    """회전 박스 기반 BEV NMS. box[6]/box[7]의 sin/cos으로 실제 yaw를 복원해
    회전 직사각형 IoU를 계산하고, center-distance(<2.0m) suppression을 병행한다."""
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)

    corners = _bev_rbox_corners(boxes)       # [N, 4, 2]
    keep_all = []
    for cls_id in labels.unique():
        cls_idx = torch.where(labels == cls_id)[0]
        cls_scores = scores[cls_idx]
        order = torch.argsort(cls_scores, descending=True)

        while order.numel() > 0:
            current_local = order[0]
            current_idx = cls_idx[current_local]
            keep_all.append(current_idx)
            if order.numel() == 1:
                break

            rest_local = order[1:]
            rest_idx = cls_idx[rest_local]

            iou = _rotated_iou_one_to_many(corners[current_idx], corners[rest_idx])
            center_dist = torch.norm(
                boxes[rest_idx, :2] - boxes[current_idx, :2], dim=-1
            )
            suppress = (iou >= iou_thresh) | (center_dist < center_dist_thresh)
            order = rest_local[~suppress]

    if not keep_all:
        return torch.zeros(0, dtype=torch.long, device=boxes.device)
    return torch.stack(keep_all)


@torch.no_grad()
def decode_detections(
    det_classes,
    det_boxes,
    det_quality=None,
    score_thresh=0.10,
    bg_idx=2,
    nms_iou=0.3,
    pre_nms_topk=300,
    apply_quality=True,
    quality_power=1.0,
):
    probs = det_classes.sigmoid()
    fg_probs = probs
    scores, labels = fg_probs.max(dim=-1)
    if apply_quality and det_quality is not None:
        centerness = quality_centerness(det_quality).view(-1)
        quality = torch.sigmoid(centerness).clamp(min=1e-6)
        scores = scores * quality.pow(float(quality_power))
    keep = scores >= score_thresh
    if keep.sum() == 0:
        empty_long = torch.zeros(0, dtype=torch.long, device=det_boxes.device)
        empty_float = torch.zeros(0, dtype=det_boxes.dtype, device=det_boxes.device)
        return det_boxes[:0], empty_long, empty_float

    boxes = det_boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    if scores.numel() > pre_nms_topk:
        topk_idx = torch.topk(scores, k=pre_nms_topk, largest=True).indices
        boxes = boxes[topk_idx]
        labels = labels[topk_idx]
        scores = scores[topk_idx]
    # v9: 회전 박스 기반 NMS (기존 bev_nms_axis_aligned는 호환용으로 유지)
    keep_nms = bev_nms_rotated(boxes, scores, labels, iou_thresh=nms_iou)
    return boxes[keep_nms], labels[keep_nms], scores[keep_nms]


@torch.no_grad()
def compute_detection_counts(
    det_classes,
    det_boxes,
    gt_classes,
    gt_boxes,
    det_quality=None,
    distance_thr=2.0,
    score_thresh=0.10,
    apply_quality=True,
    quality_power=1.0,
):
    pred_boxes, pred_labels, pred_scores = decode_detections(
        det_classes,
        det_boxes,
        det_quality=det_quality,
        score_thresh=score_thresh,
        apply_quality=apply_quality,
        quality_power=quality_power,
    )

    if gt_boxes.numel() == 0:
        return 0, int(pred_boxes.shape[0]), 0

    gt_classes = gt_classes.long().view(-1)
    matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)
    tp = 0
    fp = 0

    order = torch.argsort(pred_scores, descending=True)
    for pred_idx in order:
        same_cls = gt_classes == pred_labels[pred_idx]
        candidates = torch.where(same_cls & (~matched_gt))[0]
        if candidates.numel() == 0:
            fp += 1
            continue

        distances = torch.norm(gt_boxes[candidates, :2] - pred_boxes[pred_idx, :2], dim=-1)
        best_dist, best_local = distances.min(dim=0)
        if best_dist <= distance_thr:
            matched_gt[candidates[best_local]] = True
            tp += 1
        else:
            fp += 1

    fn = int((~matched_gt).sum().item())
    return tp, fp, fn


@torch.no_grad()
def compute_detection_counts_by_class(
    det_classes,
    det_boxes,
    gt_classes,
    gt_boxes,
    det_quality=None,
    distance_thr=2.0,
    score_thresh=0.10,
    apply_quality=True,
    quality_power=1.0,
    class_ids=(0, 1),
):
    """클래스별 (tp, fp, fn)을 dict로 반환. 매칭은 class-aware라 각 클래스가
    독립적이므로 클래스별 합은 compute_detection_counts의 합산값과 동일하다."""
    pred_boxes, pred_labels, pred_scores = decode_detections(
        det_classes,
        det_boxes,
        det_quality=det_quality,
        score_thresh=score_thresh,
        apply_quality=apply_quality,
        quality_power=quality_power,
    )
    gt_classes = gt_classes.long().view(-1)

    result = {}
    for cls in class_ids:
        p_sel = pred_labels == cls
        g_sel = gt_classes == cls
        p_boxes = pred_boxes[p_sel]
        p_scores = pred_scores[p_sel]
        g_boxes = gt_boxes[g_sel]

        n_gt = int(g_boxes.shape[0])
        matched = torch.zeros(n_gt, dtype=torch.bool, device=det_boxes.device)
        tp = 0
        fp = 0
        order = torch.argsort(p_scores, descending=True)
        for pred_idx in order:
            if n_gt == 0:
                fp += 1
                continue
            candidates = torch.where(~matched)[0]
            if candidates.numel() == 0:
                fp += 1
                continue
            distances = torch.norm(
                g_boxes[candidates, :2] - p_boxes[pred_idx, :2], dim=-1
            )
            best_dist, best_local = distances.min(dim=0)
            if best_dist <= distance_thr:
                matched[candidates[best_local]] = True
                tp += 1
            else:
                fp += 1
        fn = int((~matched).sum().item())
        result[cls] = (tp, fp, fn)
    return result


@torch.no_grad()
def compute_detection_counts_by_distance(
    det_classes,
    det_boxes,
    gt_classes,
    gt_boxes,
    det_quality=None,
    distance_thr=2.0,
    score_thresh=0.15,
    apply_quality=True,
    quality_power=0.5,
    buckets=DISTANCE_BUCKETS,
):
    """v10 평가 전용: ego 방사거리(sqrt(x^2+y^2)) 3구간별 tp/fp/fn과
    매칭쌍 center distance 합/개수를 반환한다. 매칭 알고리즘은
    compute_detection_counts와 동일(greedy, class-aware, score 내림차순).
    TP/FN은 GT 방사거리로, FP는 매칭 대상 GT가 없으므로 예측 자신의
    방사거리로 버킷팅한다. 학습/손실/체크포인트 로직에는 쓰이지 않는다.
    """
    pred_boxes, pred_labels, pred_scores = decode_detections(
        det_classes,
        det_boxes,
        det_quality=det_quality,
        score_thresh=score_thresh,
        apply_quality=apply_quality,
        quality_power=quality_power,
    )
    gt_classes = gt_classes.long().view(-1)

    result = {
        i: {'tp': 0, 'fp': 0, 'fn': 0, 'dist_sum': 0.0, 'dist_n': 0}
        for i in range(len(buckets))
    }

    if gt_boxes.numel() == 0:
        for pred_idx in range(pred_boxes.shape[0]):
            pdist = torch.norm(pred_boxes[pred_idx, :2]).item()
            b = _distance_bucket_idx(pdist, buckets)
            if b is not None:
                result[b]['fp'] += 1
        return result

    gt_dist = torch.norm(gt_boxes[:, :2], dim=-1)
    matched_gt = torch.zeros(gt_boxes.shape[0], dtype=torch.bool, device=gt_boxes.device)

    order = torch.argsort(pred_scores, descending=True)
    for pred_idx in order:
        same_cls = gt_classes == pred_labels[pred_idx]
        candidates = torch.where(same_cls & (~matched_gt))[0]
        if candidates.numel() == 0:
            pdist = torch.norm(pred_boxes[pred_idx, :2]).item()
            b = _distance_bucket_idx(pdist, buckets)
            if b is not None:
                result[b]['fp'] += 1
            continue

        distances = torch.norm(gt_boxes[candidates, :2] - pred_boxes[pred_idx, :2], dim=-1)
        best_dist, best_local = distances.min(dim=0)
        if best_dist <= distance_thr:
            g_idx = candidates[best_local]
            matched_gt[g_idx] = True
            b = _distance_bucket_idx(gt_dist[g_idx].item(), buckets)
            if b is not None:
                result[b]['tp'] += 1
                result[b]['dist_sum'] += float(best_dist.item())
                result[b]['dist_n'] += 1
        else:
            pdist = torch.norm(pred_boxes[pred_idx, :2]).item()
            b = _distance_bucket_idx(pdist, buckets)
            if b is not None:
                result[b]['fp'] += 1

    fn_idx = torch.where(~matched_gt)[0]
    for g_idx in fn_idx:
        b = _distance_bucket_idx(gt_dist[g_idx].item(), buckets)
        if b is not None:
            result[b]['fn'] += 1

    return result


@torch.no_grad()
def compute_metrics_sweep(det_classes, det_boxes, gt_classes, gt_boxes,
                          det_quality, metric_modes, score_thresholds,
                          distance_thr=2.0, class_ids=(0, 1)):
    """검증 metric 가속 (결과 보존).

    기존: mode(3) × threshold(6) 마다 compute_detection_counts_by_class 를 호출 →
          프레임당 decode/NMS + greedy 매칭을 18회 반복.
    개선: mode 당 '최저 threshold(예: 0.01)'로 decode/NMS 를 1회만 수행하고,
          클래스별 greedy 매칭을 score 내림차순으로 1회 sweep 하면서 각 threshold
          경계에서 (tp,fp,fn) 스냅샷을 남긴다 → mode 당 decode 1회 + 클래스 sweep.

    동치 근거:
      - score 내림차순 NMS(+ top-k by score) 에서 낮은 점수 예측은 높은 점수 예측의
        생존을 못 바꾸므로, {decode(0.01) 결과를 score>=T 로 필터} == {decode(T) 결과}.
      - greedy 매칭도 score 내림차순이라 threshold T 의 결과 = 0.01 sweep 의 'score>=T
        prefix' 와 동일. 따라서 sweep 중 T 경계 스냅샷이 by_class(T) 와 일치한다.

    반환: {mode: {thr: {cls: (tp, fp, fn)}}} — overall 은 클래스 합으로 유도(기존과 동일).
    """
    gt_classes = gt_classes.long().view(-1)
    thr_desc = sorted(score_thresholds, reverse=True)
    min_thr = float(min(score_thresholds))
    out = {}
    for mode, (apply_quality, quality_power) in metric_modes.items():
        pred_boxes, pred_labels, pred_scores = decode_detections(
            det_classes, det_boxes, det_quality=det_quality,
            score_thresh=min_thr, apply_quality=apply_quality,
            quality_power=quality_power,
        )
        mode_out = {thr: {} for thr in score_thresholds}
        for cls in class_ids:
            p_sel = pred_labels == cls
            g_boxes = gt_boxes[gt_classes == cls]
            n_gt = int(g_boxes.shape[0])
            p_boxes = pred_boxes[p_sel]
            order = torch.argsort(pred_scores[p_sel], descending=True)
            p_boxes = p_boxes[order]
            scores_sorted = pred_scores[p_sel][order].tolist()

            matched = torch.zeros(n_gt, dtype=torch.bool)
            cum_tp = 0
            cum_fp = 0
            ti = 0
            for i, s in enumerate(scores_sorted):
                # 이 예측 점수보다 '엄격히 큰' threshold 는 이 예측을 포함하지 않음 → 여기서 확정
                while ti < len(thr_desc) and thr_desc[ti] > s:
                    mode_out[thr_desc[ti]][cls] = (cum_tp, cum_fp, n_gt - cum_tp)
                    ti += 1
                if n_gt == 0:
                    cum_fp += 1
                    continue
                cand = torch.where(~matched)[0]
                if cand.numel() == 0:
                    cum_fp += 1
                    continue
                dists = torch.norm(g_boxes[cand, :2] - p_boxes[i, :2], dim=-1)
                best_dist, best_local = dists.min(dim=0)
                if best_dist <= distance_thr:
                    matched[cand[best_local]] = True
                    cum_tp += 1
                else:
                    cum_fp += 1
            # 남은(작은) threshold: 모든 예측이 반영된 최종 상태
            while ti < len(thr_desc):
                mode_out[thr_desc[ti]][cls] = (cum_tp, cum_fp, n_gt - cum_tp)
                ti += 1
        out[mode] = mode_out
    return out


# 검증 metric 계약 상수 — val 함수와 멀티프로세싱 워커가 동일하게 공유(드리프트 방지).
_VAL_METRIC_MODES = {
    "calibrated": (True, 1.0),      # foreground * quality
    "softcalibrated": (True, 0.5),  # foreground * sqrt(quality)
    "raw": (False, 1.0),            # foreground only
}
_VAL_SCORE_THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.15, 0.25)


@torch.no_grad()
def _val_frame_metric(det_cls, det_box, det_q, gt_cls, gt_box, recall_thr, class_ids):
    """한 프레임의 검증 metric 전부(score 통계 + threshold sweep + 거리버킷)를 계산한다.
    joblib 멀티프로세싱 워커로 쓰이며, 순수 CPU 연산이라 프레임 간 독립·결정적이다.
    반환은 모두 '합산 가능한' 카운트/스칼라라 병합 순서와 무관하게 결과가 같다.
    (입력 텐서는 이미 CPU 로 옮겨진 det/gt 이므로 CUDA fork 문제 없음.)"""
    fg = det_cls.sigmoid()
    raw_scores = fg.max(dim=-1).values
    quality = torch.sigmoid(quality_centerness(det_q).view(-1)).clamp(min=1e-6)
    soft_scores = raw_scores * quality.sqrt()
    n = int(raw_scores.numel())
    stats = {
        "quality_sum": float(quality.sum()) if n else 0.0,
        "raw_sum": float(raw_scores.sum()) if n else 0.0,
        "soft_sum": float(soft_scores.sum()) if n else 0.0,
        "quality_max": float(quality.max()) if n else 0.0,
        "raw_max": float(raw_scores.max()) if n else 0.0,
        "soft_max": float(soft_scores.max()) if n else 0.0,
        "count": n,
    }
    swept = compute_metrics_sweep(
        det_cls, det_box, gt_cls, gt_box, det_q,
        _VAL_METRIC_MODES, _VAL_SCORE_THRESHOLDS,
        distance_thr=recall_thr, class_ids=class_ids,
    )
    dist = compute_detection_counts_by_distance(
        det_cls, det_box, gt_cls, gt_box, det_quality=det_q,
        distance_thr=recall_thr, score_thresh=0.15,
        apply_quality=True, quality_power=0.5,
    )
    return stats, swept, dist


def _tensor_health(t):
    """텐서(또는 텐서 리스트/튜플)의 (nan개수, inf개수, 원소수). None/비텐서면 None."""
    if t is None:
        return None
    if isinstance(t, (list, tuple)):
        parts = [p for p in (_tensor_health(x) for x in t) if p is not None]
        if not parts:
            return None
        return (sum(p[0] for p in parts), sum(p[1] for p in parts), sum(p[2] for p in parts))
    if not torch.is_tensor(t):
        return None
    tf = t.detach()
    return (int(torch.isnan(tf).sum().item()), int(torch.isinf(tf).sum().item()), int(tf.numel()))


def diagnose_nonfinite(tag, named_tensors, per_layer=None):
    """비유한(NaN/Inf) loss 발생 시, 입력·모델출력 중 어디서 처음 비유한이 나왔는지 추적 로그.
    named_tensors: [(name, tensor|list), ...] 파이프라인 순서. per_layer: (name, [B,L,...] 텐서)."""
    print(f"[NaN-DIAG] {tag}")
    any_bad = False
    for name, t in named_tensors:
        h = _tensor_health(t)
        if h is None:
            continue
        nan, inf, n = h
        flag = "⚠️" if (nan or inf) else "  "
        if nan or inf:
            any_bad = True
            print(f"   {flag} {name}: nan={nan} inf={inf} / {n}")
    # decoder refinement 레이어별로 어느 layer 부터 비유한인지 (aggregation vs refinement 구분)
    if per_layer is not None:
        pname, pt = per_layer
        if torch.is_tensor(pt) and pt.ndim >= 2:
            for li in range(pt.shape[1]):
                h = _tensor_health(pt[:, li])
                if h and (h[0] or h[1]):
                    print(f"      ↳ {pname} layer {li}: nan={h[0]} inf={h[1]}  (여기부터 비유한)")
                    break
    if not any_bad:
        print("   (검사 텐서엔 비유한 없음 → loss 내부 계산에서 발생 추정)")


def compute_auxiliary_detection_loss(model_out, batch, criterion, device, aux_weight=0.5):
    """
    SparseDrive처럼 모든 decoder refinement 출력에 loss를 건다.
    마지막 layer는 1.0, 이전 layer들은 aux_weight로 반영하고 weight 합으로 정규화한다.
    """
    all_cls = model_out['all_det_cls']          # [B, L, N, C]
    all_box = model_out['all_det_box']          # [B, L, N, 11]
    all_quality = model_out['all_det_quality']  # [B, L, N, 2]
    B, L = all_cls.shape[:2]
    layer_weights = all_cls.new_full((L,), float(aux_weight))
    layer_weights[-1] = 1.0
    normalizer = layer_weights.sum().clamp(min=1.0) * max(B, 1)

    total_loss = all_cls.new_tensor(0.0)
    cls_loss_sum = all_cls.new_tensor(0.0)
    box_loss_sum = all_cls.new_tensor(0.0)
    quality_loss_sum = all_cls.new_tensor(0.0)

    # P2 velocity validity: raw 속도만 vx/vy loss에 기여. motion/불확실 source는 mask.
    # (temporal loader가 batch['velocity_valid']를 gt_boxes와 동일 순서로 제공. 없으면 None
    #  → 기존 동작과 동일. matcher/cost에는 영향 없음.)
    vel_valid_list = batch.get('velocity_valid', None)
    for b in range(B):
        gt_boxes = batch['dynamic_gt_boxes'][b].to(device)
        gt_classes = batch['dynamic_gt_labels'][b].to(device)
        vel_valid_b = None
        if vel_valid_list is not None and b < len(vel_valid_list) and vel_valid_list[b] is not None:
            vv = vel_valid_list[b]
            vel_valid_b = vv.to(device) if torch.is_tensor(vv) else torch.as_tensor(vv, device=device)
        for layer_idx in range(L):
            weight = layer_weights[layer_idx]
            det_loss, cls_loss, box_loss, quality_loss = criterion(
                all_cls[b, layer_idx],
                all_box[b, layer_idx],
                gt_classes,
                gt_boxes,
                all_quality[b, layer_idx],
                velocity_valid=vel_valid_b,
            )
            total_loss = total_loss + weight * det_loss
            cls_loss_sum = cls_loss_sum + weight * cls_loss.detach()
            box_loss_sum = box_loss_sum + weight * box_loss.detach()
            quality_loss_sum = quality_loss_sum + weight * quality_loss.detach()

    return (
        total_loss / normalizer,
        cls_loss_sum / normalizer,
        box_loss_sum / normalizer,
        quality_loss_sum / normalizer,
    )


def compute_parity_detection_loss(model_out, batch, criterion, device, aux_weight=None):
    """Phase 2 공식 Stage-1 계약 집계.

    compute_auxiliary_detection_loss 와 signature/반환 형태 호환(드롭인 교체).
    차이:
      - 정규화: 프레임별 평균이 아니라 **배치 전체 num_pos** 로 한 번 나눈다
        (공식 head 435-437행 — per-object gradient 가 프레임 GT 밀도와 무관해짐)
      - decoder 결합: aux_weight 가중평균이 아니라 6개 동일가중 **합산**
      - criterion 은 ParityLoss (frame 단위 비정규화 합 반환)
    aux_weight 인자는 호환용으로만 받고 무시한다.
    진단용 부가 통계는 compute_parity_detection_loss.last_stats 에 기록된다.
    """
    all_cls = model_out['all_det_cls']          # [B, L, N, C]
    all_box = model_out['all_det_box']          # [B, L, N, 11]
    all_quality = model_out['all_det_quality']  # [B, L, N, 2]
    B, L = all_cls.shape[:2]

    zero = all_cls.new_tensor(0.0)
    cls_sums = [zero] * L
    box_sums = [zero] * L
    qual_sums = [zero] * L
    num_pos_batch = 0
    num_gate_batch = 0

    vel_valid_list = batch.get('velocity_valid', None)
    for b in range(B):
        gt_boxes = batch['dynamic_gt_boxes'][b].to(device)
        gt_classes = batch['dynamic_gt_labels'][b].to(device)
        vel_valid_b = None
        if vel_valid_list is not None and b < len(vel_valid_list) and vel_valid_list[b] is not None:
            vv = vel_valid_list[b]
            vel_valid_b = vv.to(device) if torch.is_tensor(vv) else torch.as_tensor(vv, device=device)
        for li in range(L):
            out = criterion(
                all_cls[b, li], all_box[b, li], gt_classes, gt_boxes,
                all_quality[b, li], velocity_valid=vel_valid_b,
            )
            cls_sums[li] = cls_sums[li] + out['cls_sum']
            box_sums[li] = box_sums[li] + out['box_sum']
            qual_sums[li] = qual_sums[li] + out['cns_sum'] + out['yns_sum']
            if li == L - 1:                     # num_pos 는 layer 무관(동일 GT)
                num_pos_batch += out['num_pos']
                num_gate_batch += out['num_gate']

    denom = float(max(num_pos_batch, 1))
    cls_loss = sum(cls_sums) / denom
    box_loss = sum(box_sums) / denom
    quality_loss = sum(qual_sums) / denom
    total = cls_loss + box_loss + quality_loss

    compute_parity_detection_loss.last_stats = {
        'num_pos': num_pos_batch,
        'num_gate': num_gate_batch,
        'gate_rate': num_gate_batch / max(num_pos_batch, 1),
    }
    return total, cls_loss.detach(), box_loss.detach(), quality_loss.detach()


compute_parity_detection_loss.last_stats = {}

# 학습/검증 공용 loss 집계 dispatch. main 에서 LOSS_CONTRACT=parity 면 재바인딩된다.
_AUX_LOSS_FN = compute_auxiliary_detection_loss


def load_v10_weights(
    model,
    ckpt_path,
    verbose=True,
    load_anchor=False,
    reinit_instance_feature=False,
    zero_reg_branch6=True,
):
    """
    v10 단일프레임 체크포인트(best_model.pth, 순수 state_dict)를 v11 temporal 모델로
    전이한다. MIGRATION_PLAN §5 규칙 (Stage A0 진단 후 개정):
      - 로드   : backbone, deformable_agg, self_attn/norms/ffn, cls_branch,
                 quality_branch, reg_branch[0-4](offset 생성 앞단), instance_feature
      - 로드안함(zero-init): reg_branch.6.{weight,bias} **전체**.
                 앵커를 신규 K-means로 재생성했으므로 v10이 옛 앵커 기준으로 학습한
                 최종 offset은 무효다(Stage A0 진단: 예측 box가 GT에서 6~10m 이탈 →
                 매칭 불안정 → cls confidence 붕괴). anchor에서 refine을 다시 시작하도록
                 마지막 Linear를 0으로 초기화한다(decoder.py 원래 관례와 동일).
      - 스킵(재초기화): temp_*(layer1-5, temporal off로 학습된 적 없음),
                 det_anchors_full(신규 K-means 앵커를 buffer로 유지 — 덮어쓰면 안 됨)
      - 신규   : depth_net.*(v10에 없음, 새 init 유지)
    반환: dict(loaded, skipped_rule, skipped_shape, missing) 키 리스트.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    src = ckpt["model_state"] if (isinstance(ckpt, dict) and "model_state" in ckpt) else ckpt

    model_sd = model.state_dict()
    new_sd = {}
    loaded, skipped_rule, skipped_shape, missing = [], [], [], []

    def skip_reason(k):
        if k == "det_anchors_full" and not load_anchor:
            return "anchor(신규 kmeans 유지)"
        if k == "instance_feature" and reinit_instance_feature:
            return "instance_feature(reinit for new anchor)"
        if ".temp_" in k:
            return "temp_*(재초기화)"
        return None

    for k, v in model_sd.items():
        reason = skip_reason(k)
        if reason is not None:
            new_sd[k] = v                      # model init 유지
            skipped_rule.append((k, reason))
        elif k in src and src[k].shape == v.shape:
            new_sd[k] = src[k].clone()         # 전이
            loaded.append(k)
        elif k in src:
            new_sd[k] = v
            skipped_shape.append((k, tuple(src[k].shape), tuple(v.shape)))
        else:
            new_sd[k] = v                      # 신규(depth_net 등)
            missing.append(k)

    # reg_branch 마지막 Linear(index 6) 전체 zero-init (weight+bias 전부).
    # loaded 목록에서 제거하고 재초기화 사유로 기록.
    n_reg_zeroed = 0
    for k in list(new_sd.keys()):
        if k.endswith("det_decoder.reg_branch.6.weight"):
            new_sd[k] = torch.zeros_like(new_sd[k]); n_reg_zeroed += 1
            if k in loaded:
                loaded.remove(k); skipped_rule.append((k, "reg_branch.6(anchor재refine 위해 zero)"))
        elif k.endswith("det_decoder.reg_branch.6.bias"):
            new_sd[k] = torch.zeros_like(new_sd[k])
            if k in loaded:
                loaded.remove(k); skipped_rule.append((k, "reg_branch.6(anchor재refine 위해 zero)"))

    if not zero_reg_branch6:
        for k in list(new_sd.keys()):
            if (
                k.endswith("det_decoder.reg_branch.6.weight")
                or k.endswith("det_decoder.reg_branch.6.bias")
            ):
                if k in src and src[k].shape == new_sd[k].shape:
                    new_sd[k] = src[k].clone()
        n_reg_zeroed = 0

    model.load_state_dict(new_sd)

    if verbose:
        print(f"   options: load_anchor={load_anchor}, "
              f"reinit_instance_feature={reinit_instance_feature}, "
              f"zero_reg_branch6={zero_reg_branch6}")
        print(f"[전이] v10→v11 from {ckpt_path}")
        print(f"   로드 {len(loaded)} | 스킵(규칙) {len(skipped_rule)} "
              f"| 스킵(shape불일치) {len(skipped_shape)} | 신규 {len(missing)}")
        print(f"   reg_branch.6 전체 zero-init: {n_reg_zeroed}개 layer (anchor에서 refine 재시작)")
        if skipped_rule:
            from collections import Counter
            rc = Counter(r for _, r in skipped_rule)
            print(f"   스킵(규칙) 내역: {dict(rc)}")
        if skipped_shape:
            print(f"   ⚠️ shape 불일치 스킵: {skipped_shape[:3]}")
        print(f"   신규(새 init) 예: {missing[:4]}")
    return {"loaded": loaded, "skipped_rule": skipped_rule,
            "skipped_shape": skipped_shape, "missing": missing}


def build_transfer_param_groups(
    model, base_lr=2e-4,
    backbone_mult=0.1, loaded_mult=0.5, new_mult=1.0,
    wd_backbone=1e-3, wd_other=1e-2,
):
    """
    v11 전이학습용 3단 옵티마이저 그룹 (MIGRATION_PLAN §5/§6):
      backbone (×0.1)          : 도메인 적응 완료 — 미세조정
      loaded   (×0.5)          : v10에서 전이된 디코더 공통부 + instance_feature
      new      (×1.0)          : 신규/재초기화 — depth_net, temp_*, reg_branch.6(velocity)
    reg_branch.6은 텐서 단위라 velocity 행만 분리 못하므로 마지막 Linear 전체를 new로.
    """
    def classify(name):
        if name.startswith("backbone."):
            return "backbone"
        if (name.startswith("depth_net.") or ".temp_" in name or
                "det_decoder.reg_branch.6." in name):
            return "new"
        return "loaded"

    buckets = {"backbone": [], "loaded": [], "new": []}
    names = {"backbone": [], "loaded": [], "new": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        g = classify(name)
        buckets[g].append(p)
        names[g].append(name)

    groups = [
        {"params": buckets["backbone"], "lr": base_lr * backbone_mult, "weight_decay": wd_backbone},
        {"params": buckets["loaded"],   "lr": base_lr * loaded_mult,   "weight_decay": wd_other},
        {"params": buckets["new"],      "lr": base_lr * new_mult,      "weight_decay": wd_other},
    ]
    return groups, names


def count_ego_pose_files(dataset):
    total = len(dataset.items)
    count = 0
    for item in dataset.items:
        if isinstance(item, dict):
            scen_name = item.get('scen_name')
            stem = item.get('stem')
            if (
                hasattr(dataset, 'scene_infos') and
                scen_name in dataset.scene_infos and
                stem in dataset.scene_infos[scen_name] and
                'T_ego2global' in dataset.scene_infos[scen_name][stem]
            ):
                count += 1
        else:
            scen_dir, stem = item
            pose_path = os.path.join(scen_dir, 'ego_pose', f"{stem}.csv")
            if os.path.isfile(pose_path):
                count += 1
    return count, total


@torch.no_grad()
def validate(model, loader, criterion, device, compute_metric=False, recall_thr=2.0,
             max_frames=0):
    """
    val set 전체에 대해 평균 detection loss를 계산하고,
    compute_metric=True면 score/class-aware Precision/Recall도 계산.
    max_frames>0이면 val_loader 앞쪽에서 그만큼의 프레임만 처리한다(fast 실험용
    subsample). val_loader는 shuffle=False라 항상 동일한 앞부분을 사용 → 비교 가능.
    반환: (val_loss, metrics_or_None)
    """
    model.eval()
    if hasattr(model, "reset_temporal_memory"):
        model.reset_temporal_memory()
    loss_sum = 0.0
    n_batches = 0
    main_score_threshold = 0.05
    metric_score_thresholds = _VAL_SCORE_THRESHOLDS   # 워커와 동일 상수 공유(드리프트 방지)
    metric_modes = _VAL_METRIC_MODES
    metric_counts = {
        mode: {
            thr: {'tp': 0, 'fp': 0, 'fn': 0}
            for thr in metric_score_thresholds
        }
        for mode in metric_modes
    }
    # v9: 클래스별(vehicle/pedestrian) 카운트 추가 집계
    class_ids = tuple(sorted(CLASS_ID_NAMES.keys()))
    metric_counts_cls = {
        mode: {
            thr: {cls: {'tp': 0, 'fp': 0, 'fn': 0} for cls in class_ids}
            for thr in metric_score_thresholds
        }
        for mode in metric_modes
    }
    score_sums = {'quality': 0.0, 'raw': 0.0, 'soft': 0.0}
    score_max = {'quality': 0.0, 'raw': 0.0, 'soft': 0.0}
    score_count = 0

    # v10: softcalibrated@0.15 ego 방사거리 3구간 P/R/F1 + 매칭쌍 평균 center distance (로깅 전용)
    distance_bucket_counts = {
        i: {'tp': 0, 'fp': 0, 'fn': 0, 'dist_sum': 0.0, 'dist_n': 0}
        for i in range(len(DISTANCE_BUCKETS))
    }

    # ── 검증 metric 병렬화 (프레임 단위 독립 → 멀티프로세싱). env VAL_METRIC_WORKERS 로 제어.
    #    joblib 이 입력 순서대로 결과를 반환하고 병합은 합산이라, 직렬 경로와 bit-단위 동일.
    #    VAL_METRIC_WORKERS<=1 또는 joblib 없음 → 직렬 fallback. VAL_METRIC_BLOCK: 메모리 상한.
    _val_workers = int(os.environ.get(
        "VAL_METRIC_WORKERS", str(max(1, min(10, (os.cpu_count() or 2) - 2)))))
    _use_parallel = bool(compute_metric and _val_workers > 1 and _HAS_JOBLIB)
    _val_block = max(1, int(os.environ.get("VAL_METRIC_BLOCK", "512")))
    _pending = []

    def _merge_frame(result):
        nonlocal score_count
        stats, swept, dist = result
        score_sums['quality'] += stats['quality_sum']
        score_sums['raw'] += stats['raw_sum']
        score_sums['soft'] += stats['soft_sum']
        score_max['quality'] = max(score_max['quality'], stats['quality_max'])
        score_max['raw'] = max(score_max['raw'], stats['raw_max'])
        score_max['soft'] = max(score_max['soft'], stats['soft_max'])
        score_count += stats['count']
        for mode in metric_modes:
            for score_thr in metric_score_thresholds:
                tp = fp = fn = 0
                for cls in class_ids:
                    ctp, cfp, cfn = swept[mode][score_thr][cls]
                    metric_counts_cls[mode][score_thr][cls]['tp'] += ctp
                    metric_counts_cls[mode][score_thr][cls]['fp'] += cfp
                    metric_counts_cls[mode][score_thr][cls]['fn'] += cfn
                    tp += ctp
                    fp += cfp
                    fn += cfn
                metric_counts[mode][score_thr]['tp'] += tp
                metric_counts[mode][score_thr]['fp'] += fp
                metric_counts[mode][score_thr]['fn'] += fn
        for b, counts in dist.items():
            for k in ('tp', 'fp', 'fn', 'dist_sum', 'dist_n'):
                distance_bucket_counts[b][k] += counts[k]

    def _flush_pending():
        if not _pending:
            return
        for r in Parallel(n_jobs=_val_workers, prefer='processes')(
            delayed(_val_frame_metric)(dc, db, dq, gc, gb, recall_thr, class_ids)
            for (dc, db, dq, gc, gb) in _pending
        ):
            _merge_frame(r)
        _pending.clear()

    frames_done = 0
    for batch in loader:
        if max_frames and frames_done >= max_frames:
            break
        images     = batch['images'].to(device)
        intrinsics = batch['intrinsics'].to(device)
        extrinsics = batch['extrinsics'].to(device)
        ego_poses  = batch['ego_pose'].to(device)
        n = images.shape[0]

        model_out = model(
            images,
            intrinsics,
            extrinsics,
            stems=batch['stem'],
            ego_poses=ego_poses,
            timestamps=batch.get('timestamp'),
            return_intermediate=True,
        )
        # metric matching은 예측 개수만큼 python 루프 + 텐서 인덱싱이라, GPU에 두면
        # 반복마다 sync가 걸려 극도로 느리다(특히 낮은 score_thresh). CPU로 옮겨 sync 제거.
        det_classes_b = model_out['det_cls'].detach().cpu()
        det_boxes_b = model_out['det_box'].detach().cpu()
        det_quality_b = model_out['det_quality'].detach().cpu()
        batch_loss, _, _, _ = _AUX_LOSS_FN(
            model_out, batch, criterion, device
        )

        for i in range(n):
            if not compute_metric:
                continue
            # det_*_b가 CPU이므로 GT도 CPU에 둔다(metric matching 전용, sync 제거).
            # clone: 워커로 넘길 프레임 조각을 batch 텐서에서 분리(배치 조기 해제 + pickling 안전).
            frame = (
                det_classes_b[i].clone(), det_boxes_b[i].clone(), det_quality_b[i].clone(),
                batch['dynamic_gt_labels'][i].cpu().clone(),
                batch['dynamic_gt_boxes'][i].cpu().clone(),
            )
            if _use_parallel:
                _pending.append(frame)
            else:
                # 직렬 경로: 워커 함수를 그대로 호출(병렬과 동일 계산) → bit-단위 동일
                dc, db, dq, gc, gb = frame
                _merge_frame(_val_frame_metric(dc, db, dq, gc, gb, recall_thr, class_ids))

        if _use_parallel and len(_pending) >= _val_block:
            _flush_pending()

        loss_sum += batch_loss.item()
        n_batches += 1
        frames_done += n

    if _use_parallel:
        _flush_pending()   # 남은 프레임 병합

    avg_loss = loss_sum / max(n_batches, 1)
    if not compute_metric:
        return avg_loss, None

    by_mode = {}
    for mode, counts_by_thr in metric_counts.items():
        by_mode[mode] = {}
        for score_thr, counts in counts_by_thr.items():
            tp = counts['tp']
            fp = counts['fp']
            fn = counts['fn']
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            by_mode[mode][score_thr] = {
                'precision': precision,
                'recall': recall,
                'f1': 2.0 * precision * recall / max(precision + recall, 1e-12),
                'tp': tp,
                'fp': fp,
                'fn': fn,
            }

    by_mode_class = {}
    for mode, counts_by_thr in metric_counts_cls.items():
        by_mode_class[mode] = {}
        for score_thr, counts_by_cls in counts_by_thr.items():
            by_mode_class[mode][score_thr] = {}
            for cls, counts in counts_by_cls.items():
                tp = counts['tp']
                fp = counts['fp']
                fn = counts['fn']
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                by_mode_class[mode][score_thr][cls] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': 2.0 * precision * recall / max(precision + recall, 1e-12),
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                }

    main = by_mode['calibrated'][main_score_threshold]
    score_stats = {
        key: value / max(score_count, 1)
        for key, value in score_sums.items()
    }
    for key, value in score_max.items():
        score_stats[f'{key}_max'] = value

    # v10: softcalibrated@0.15 거리구간별 P/R/F1 + 평균 matched center distance
    by_distance = {}
    for b, (lo, hi) in enumerate(DISTANCE_BUCKETS):
        c = distance_bucket_counts[b]
        d_precision = c['tp'] / max(c['tp'] + c['fp'], 1)
        d_recall = c['tp'] / max(c['tp'] + c['fn'], 1)
        by_distance[(lo, hi)] = {
            'precision': d_precision,
            'recall': d_recall,
            'f1': 2.0 * d_precision * d_recall / max(d_precision + d_recall, 1e-12),
            'mean_center_dist': (c['dist_sum'] / c['dist_n']) if c['dist_n'] > 0 else None,
            'tp': c['tp'],
            'fp': c['fp'],
            'fn': c['fn'],
        }

    return avg_loss, {
        'precision': main['precision'],
        'recall': main['recall'],
        'score_thresh': main_score_threshold,
        'by_score': by_mode['calibrated'],
        'by_mode': by_mode,
        'by_mode_class': by_mode_class,
        'score_stats': score_stats,
        'by_distance': by_distance,
    }


HISTORY_SCORE_THRESHOLDS = (0.01, 0.03, 0.05, 0.10, 0.15, 0.25)
HISTORY_METRIC_MODES = ("calibrated", "softcalibrated", "raw")
HISTORY_METRIC_KEYS = ("precision", "recall", "f1")


def _score_suffix(score_thr):
    return f"{int(round(score_thr * 100)):03d}"


HISTORY_FIELDS = [
    'epoch',
    'train_loss',
    'train_cls_loss',
    'train_box_loss',
    'train_quality_loss',
    'train_depth_loss',
    'train_det_loss',   # = train_loss - train_depth_loss (val_loss 와 비교 가능한 detection-only)
    'val_loss',
    'lr',
    'val_quality_mean',
    'val_raw_score_mean',
    'val_soft_score_mean',
    'val_quality_score_max',
    'val_raw_score_max',
    'val_soft_score_max',
]
for _mode in HISTORY_METRIC_MODES:
    for _thr in HISTORY_SCORE_THRESHOLDS:
        _suffix = _score_suffix(_thr)
        for _key in HISTORY_METRIC_KEYS:
            HISTORY_FIELDS.append(f'{_mode}_{_key}_{_suffix}')


def _history_metric(val_metrics, mode, score_thr, key):
    if val_metrics is None:
        return float('nan')
    try:
        return float(val_metrics['by_mode'][mode][score_thr][key])
    except KeyError:
        return float('nan')


def make_history_record(epoch, train_loss, train_cls_loss, train_box_loss,
                        train_quality_loss, train_depth_loss, val_loss, lr, val_metrics):
    score_stats = (val_metrics or {}).get('score_stats', {})
    record = {
        'epoch': int(epoch),
        'train_loss': float(train_loss),
        'train_cls_loss': float(train_cls_loss),
        'train_box_loss': float(train_box_loss),
        'train_quality_loss': float(train_quality_loss),
        'train_depth_loss': float(train_depth_loss),
        # detection-only train loss = total - depth. val_loss 는 depth 를 안 쓰므로(eval 에선
        # depth head 미실행) train_loss(=det+depth)와 직접 비교하면 안 되고 이 값과 비교해야 한다.
        'train_det_loss': float(train_loss) - float(train_depth_loss),
        'val_loss': float(val_loss),
        'lr': float(lr),
        'val_quality_mean': float(score_stats.get('quality', float('nan'))),
        'val_raw_score_mean': float(score_stats.get('raw', float('nan'))),
        'val_soft_score_mean': float(score_stats.get('soft', float('nan'))),
        'val_quality_score_max': float(score_stats.get('quality_max', float('nan'))),
        'val_raw_score_max': float(score_stats.get('raw_max', float('nan'))),
        'val_soft_score_max': float(score_stats.get('soft_max', float('nan'))),
    }
    for mode in HISTORY_METRIC_MODES:
        for score_thr in HISTORY_SCORE_THRESHOLDS:
            suffix = _score_suffix(score_thr)
            for key in HISTORY_METRIC_KEYS:
                record[f'{mode}_{key}_{suffix}'] = _history_metric(
                    val_metrics, mode, score_thr, key
                )
    return record


def load_training_history(csv_path, max_epoch=None):
    if not os.path.isfile(csv_path):
        return []

    records = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epoch = int(float(row.get('epoch', 'nan')))
            except (TypeError, ValueError):
                continue
            if max_epoch is not None and epoch > max_epoch:
                continue

            record = {'epoch': epoch}
            for field in HISTORY_FIELDS:
                if field == 'epoch':
                    continue
                try:
                    record[field] = float(row.get(field, 'nan'))
                except (TypeError, ValueError):
                    record[field] = float('nan')
            records.append(record)
    return records


def append_epoch_loss_log(path, epoch, train_stats, lr):
    """매 epoch(=완료된 학습 epoch)마다 train loss 분해를 텍스트(CSV)로 append.
    training_history.csv 와 달리 validation 여부와 무관하게 모든 epoch를 남긴다."""
    if not path:
        return
    is_new = not os.path.isfile(path)
    with open(path, 'a', newline='') as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(['epoch', 'det_loss', 'cls_loss', 'box_loss',
                             'quality_loss', 'depth_loss', 'lr'])
        writer.writerow([
            int(epoch),
            f"{train_stats['loss']:.6f}", f"{train_stats['cls']:.6f}",
            f"{train_stats['box']:.6f}", f"{train_stats['quality']:.6f}",
            f"{train_stats['depth']:.6f}", f"{lr:.3e}",
        ])


def save_training_history(records, csv_path, plot_path):
    if not records:
        return

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, float('nan')) for field in HISTORY_FIELDS})

    epochs = [int(record['epoch']) for record in records]
    train_losses = [float(record['train_loss']) for record in records]                     # total(det+depth)
    train_det = [float(record.get('train_det_loss', float('nan'))) for record in records]   # det-only
    val_losses = [float(record['val_loss']) for record in records]                          # det-only

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 9), sharex=True)
    # val_loss 는 detection-only(eval 에서 depth head 미실행)이므로, 공정 비교는 train_det 와의 비교.
    # train total(det+depth)은 참고용 얇은 선.
    ax.plot(epochs, train_det, marker='o', linewidth=1.8, label='Train (det-only)')
    ax.plot(epochs, val_losses, marker='o', linewidth=2.2, label='Val (det-only)')
    ax.plot(epochs, train_losses, marker='.', linewidth=1.0, alpha=0.45, label='Train total (det+depth)')

    finite_val = [(idx, loss) for idx, loss in enumerate(val_losses) if math.isfinite(loss)]
    if finite_val:
        best_idx, best_loss = min(finite_val, key=lambda item: item[1])
        ax.scatter(
            [epochs[best_idx]],
            [best_loss],
            s=80,
            zorder=5,
            label=f'Best val {best_loss:.4f}',
        )

    ax.set_title('Detection loss: Train(det-only) vs Val  [+ train total]')
    ax.set_ylabel('loss')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # train loss 성분 분해 (det=cls+box+quality+depth)
    for field, lbl in (('train_cls_loss', 'Cls'), ('train_box_loss', 'Box'),
                       ('train_quality_loss', 'Quality'), ('train_depth_loss', 'Depth')):
        ys = [float(record.get(field, float('nan'))) for record in records]
        ax2.plot(epochs, ys, marker='.', linewidth=1.5, label=lbl)
    ax2.set_title('Train loss breakdown (cls / box / quality / depth)')
    ax2.set_xlabel('Epoch (validation index)')
    ax2.set_ylabel('Component loss')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def metric_value(val_metrics, mode, score_thr, key):
    if val_metrics is None:
        return float('-inf')
    return float(val_metrics['by_mode'][mode][score_thr][key])


# 학습 전용 보조 모듈 — 추론/평가/전이 소스 state_dict에서는 제외한다.
# (depth_net은 dense depth 보조 태스크용으로 추론 forward에서 실행되지 않으며,
#  기존 eval_distance.py/inference.py는 use_dense_depth=False 모델로 strict load 하므로
#  이 키가 남아 있으면 unexpected key 로 로드가 실패한다.)
AUX_ONLY_PREFIXES = ("depth_net.",)


def export_inference_state_dict(model):
    """추론/평가/전이 소스용 순수 state_dict (학습 전용 모듈 제외)."""
    return {k: v for k, v in model.state_dict().items()
            if not k.startswith(AUX_ONLY_PREFIXES)}


def save_best_with_epoch(model, fixed_path, epoch, tag, score):
    export_sd = export_inference_state_dict(model)
    out_dir = os.path.dirname(fixed_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    torch.save(export_sd, fixed_path)
    root, ext = os.path.splitext(fixed_path)
    root_name = os.path.basename(root)
    for name in os.listdir(out_dir):
        if name.startswith(f"{root_name}_epoch") and name.endswith(ext):
            os.remove(os.path.join(out_dir, name))
    epoch_path = f"{root}_epoch{epoch:03d}_{tag}_{score:.4f}{ext}"
    torch.save(export_sd, epoch_path)
    return epoch_path


# ===========================================================
# 학습 루프 (3전방 카메라 동적 객체 detection 전용)
# ===========================================================
if __name__ == "__main__":
    # ─── Config ────────────────────────────────────────
    DATASET_ROOT = os.environ.get('DATASET_ROOT', './dataset')
    # 디스크에 존재하고 labels_3d_v2를 가진 모든 scen*을 사용한다.
    # 검증 기준 유지를 위해 DEFAULT_VAL_SCENARIOS만 val로 빼고 나머지는 train.
    # ⚠️ make_kmeans.py 의 DEFAULT_VAL_SCENARIOS와 반드시 일치시킬 것.
    # env VAL_SCENARIOS(쉼표구분)로 override 가능 — 소규모 검증용 루트(scen02~06 등)에서
    # 기본 val 시나리오가 디스크에 없을 때 사용. 미설정 시 기존 동작 그대로.
    _vs_env = os.environ.get('VAL_SCENARIOS')
    VAL_SCENARIOS = ([s for s in _vs_env.split(',') if s] if _vs_env
                     else list(DEFAULT_VAL_SCENARIOS))

    # 재현성: env SEED 가 있으면 전역 시드를 고정한다(모델 init·DataLoader shuffle 결정성).
    # 기본 미설정 → 기존 동작 그대로(시드 미설정). 대표 A/B 는 동일 SEED 로 init·sampler
    # 순서를 맞춘다. GPU 커널 비결정성까지 제거하지는 않는다(재실행 차이는 preflight 에서 측정).
    SEED = os.environ.get('SEED')
    if SEED not in (None, ''):
        import random as _random
        _seed = int(SEED)
        _random.seed(_seed)
        np.random.seed(_seed)
        torch.manual_seed(_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(_seed)
        print(f"[seed] 전역 시드 고정: {_seed}")

    NUM_EPOCHS           = int(os.environ.get('NUM_EPOCHS', '100'))
    # Hardware-dependent knobs are environment-overridable, but the historical
    # defaults stay unchanged.  Production wrappers keep the effective batch at
    # eight unless a measured OOM/throughput preflight justifies another pair.
    BATCH_SIZE           = int(os.environ.get('BATCH_SIZE', '4'))
    GRAD_ACCUM_STEPS     = int(os.environ.get('GRAD_ACCUM_STEPS', '2'))
    NUM_WORKERS          = int(os.environ.get('NUM_WORKERS', '0'))
    if BATCH_SIZE < 1 or GRAD_ACCUM_STEPS < 1 or NUM_WORKERS < 0:
        raise ValueError(
            "BATCH_SIZE/GRAD_ACCUM_STEPS must be >=1 and NUM_WORKERS must be >=0: "
            f"batch={BATCH_SIZE}, accum={GRAD_ACCUM_STEPS}, workers={NUM_WORKERS}"
        )
    # early-stop patience. env override 가능(A/B 는 매우 크게 설정해 baseline/candidate 의
    # optimizer update 수를 동일하게 고정 — 한쪽만 조기 종료해 update 수가 어긋나는 것 방지).
    EARLY_STOP_PATIENCE  = int(os.environ.get('EARLY_STOP_PATIENCE', '10'))
    EARLY_STOP_MIN_DELTA = 1e-4
    METRIC_EVERY         = 1      # scheduled validation마다 primary metric 계산
    # Full validation is expensive on the 150-scene split.  This only changes
    # validation/checkpoint cadence; it never truncates the training loader.
    VALIDATE_EVERY_EPOCHS = int(os.environ.get('VALIDATE_EVERY_EPOCHS', '1'))
    if VALIDATE_EVERY_EPOCHS < 1:
        raise ValueError("VALIDATE_EVERY_EPOCHS must be >= 1")
    RECALL_THR           = 2.0    # distance match threshold
    KMEANS_K             = DEFAULT_K
    # 기본 True(신규 학습 시 앵커 재생성). env로 끄면 기존 anchor_kmeans_*.npy 재사용
    # (재시작 시 epoch1과 동일 앵커를 쓰려면 0으로). resume 시엔 아래에서 자동 False.
    FORCE_REMAKE_KMEANS  = (os.environ.get('FORCE_REMAKE_KMEANS', '1').strip().lower()
                            not in ('0', '', 'false', 'no', 'off'))
    AUX_LOSS_WEIGHT      = 0.5
    USE_TEMPORAL_DATASET = True   # labels_3d_v2 + scene_info + depth_gt 사용
    # A1(temporal) 스위치. 기본 False(=Stage A0 단일프레임). env USE_TEMPORAL_MEMORY=1 로 A1.
    # 켜면 아래 USE_STREAMING_SAMPLER 도 자동 True(배치 슬롯별 시간순 스트리밍 필수).
    # 전제: segments.json 생성 + dataset segment-aware(ego 텔레포트가 seq 안에 없어야 함).
    USE_TEMPORAL_MEMORY  = (os.environ.get('USE_TEMPORAL_MEMORY', '0').strip().lower()
                            not in ('0', '', 'false', 'no', 'off'))
    # temp_gnn ablation 스위치(USE_TEMPORAL_MEMORY=1 일 때만 의미 있음):
    #   'gated'(기본) = A1-b: temp_gnn ON + 0-init learnable gate(A0 출력 보존 후 서서히 open)
    #   'off'         = A1-a: sampler/bank는 유지하되 temp_gnn fusion만 완전히 OFF
    #   'on'          = 구 동작(gate 강제 1.0, zero-gate 없이 바로 더함) — 붕괴 재현/디버그용
    TEMP_GNN_MODE = os.environ.get('TEMP_GNN_MODE', 'gated').strip().lower()
    # 스트리밍 샘플러: instance bank(temporal memory)를 켜려면 batch slot별 시간순
    # 스트리밍이 필수다. 기본은 USE_TEMPORAL_MEMORY 를 따라간다(=지금은 off → 기존
    # shuffle DataLoader 경로 유지, 런타임 무변경). 강제로 켜서 검증하려면 True.
    # ablation: env STREAMING_SAMPLER 로 temporal memory와 독립 제어(sampler만 격리).
    #   미설정 → USE_TEMPORAL_MEMORY 추종(기존 동작)
    #   0/off  → shuffle DataLoader 강제(bank는 켜둔 채 sampler만 shuffle로 대조)
    #   1/on   → streaming 강제
    _stream_env = os.environ.get('STREAMING_SAMPLER')
    if _stream_env is None:
        USE_STREAMING_SAMPLER = USE_TEMPORAL_MEMORY
    else:
        USE_STREAMING_SAMPLER = _stream_env.strip().lower() not in ('0', '', 'false', 'no', 'off')
    SEQUENCE_LENGTH      = 150    # MoraiTemporalDataset chunk 길이와 일치시킬 것
    NUM_TEMP_INSTANCES   = 600
    # v11 checklist 7: dense depth 보조 태스크(loss weight 0.2는 DenseDepthNet 내부),
    # GridMask(p=0.7, 학습 시만), PhotoMetric(시퀀스-일관, train split만)
    # dense depth 보조 태스크. 기본은 temporal dataset 사용 시 ON(기존 동작 불변).
    # task H: env USE_DENSE_DEPTH 로 명시적 override(실험 스위치). OFF 로 두면 depth_gt I/O 와
    # DenseDepthNet/loss 를 모두 끈다(label-correction isolation preflight 용).
    _udd_env = os.environ.get('USE_DENSE_DEPTH')
    USE_DENSE_DEPTH      = (USE_TEMPORAL_DATASET if _udd_env is None
                            else _udd_env.strip().lower() not in ('0', '', 'false', 'no', 'off'))
    USE_GRID_MASK        = True
    USE_PHOTOMETRIC_AUG  = True
    # occlusion(num_lidar_pts) 필터: 박스 안 LiDAR 점 < 이 값이면 관측불가 GT로 제외(drop).
    # train+val 둘 다 적용 → val 지표도 '관측 가능한 객체' 기준으로 재측정된다.
    # 기본 1(유령/amodal 라벨만 제거). env OCCLUSION_MIN_PTS로 override(0=비활성).
    # 사전 생성 필요: python3 generate_occlusion_gt.py --all
    OCCLUSION_MIN_PTS    = int(os.environ.get('OCCLUSION_MIN_PTS', '1'))
    # GT 버전: v2(기존) 또는 v3(front-camera source-time 보정 GT + scene_info_v3).
    # 기본 v2 → 기존 학습 동작 불변. env GT_VERSION=v3 로 보정 GT 학습.
    # label 과 scene_info 는 항상 같은 버전으로만 짝지어진다(MoraiTemporalDataset 강제).
    #
    # task B: train GT 와 validation GT 를 분리한다.
    #   TRAIN_GT_VERSION / VAL_GT_VERSION 이 있으면 그것을, 없으면 기존 GT_VERSION 으로 fallback
    #   (→ 둘 다 미설정이면 기존과 완전히 동일한 단일-GT 동작).
    #   대표 A/B:  baseline = train v2 / val v3,  candidate = train v3 / val v3.
    #   ⚠️ validation metric 은 항상 동일한 v3 GT 로 측정해야 A/B 가 공정하다(VAL_GT_VERSION=v3).
    GT_VERSION           = os.environ.get('GT_VERSION', 'v2').strip().lower()
    TRAIN_GT_VERSION     = os.environ.get('TRAIN_GT_VERSION', GT_VERSION).strip().lower()
    VAL_GT_VERSION       = os.environ.get('VAL_GT_VERSION', GT_VERSION).strip().lower()
    for _gv_name, _gv in (('TRAIN_GT_VERSION', TRAIN_GT_VERSION),
                          ('VAL_GT_VERSION', VAL_GT_VERSION)):
        if _gv not in ('v2', 'v3'):
            raise ValueError(f"{_gv_name}은 'v2' 또는 'v3'이어야 합니다: {_gv}")
    PRIMARY_BEST_MODE    = "softcalibrated"
    PRIMARY_BEST_THR     = 0.15
    PRIMARY_BEST_KEY     = "f1"
    BEST_METRIC_MODE     = "raw"  # legacy/resume 호환용
    FREEZE_BACKBONE_BN   = True   # batch=sample별 3 cameras라 backbone BN은 고정
    USE_AMP              = (os.environ.get('USE_AMP', '0').strip().lower()
                            not in ('0', '', 'false', 'no', 'off'))
    ALLOW_TF32           = (os.environ.get('ALLOW_TF32', '0').strip().lower()
                            not in ('0', '', 'false', 'no', 'off'))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = ALLOW_TF32
        torch.backends.cudnn.allow_tf32 = ALLOW_TF32
    # RUN_DIR: 설정 시 이 run 의 모든 산출(checkpoint/history/plot/periodic ckpt/run_config)을
    # 한 디렉터리에 격리한다. 기존 checkpoint/log 를 덮어쓰지 않기 위한 A/B run 격리용(Phase 0).
    # 미설정 시 기존 기본 동작 그대로. resume 은 같은 RUN_DIR 안에서만 자동 탐색된다.
    RUN_DIR              = os.environ.get('RUN_DIR')
    if RUN_DIR:
        os.makedirs(RUN_DIR, exist_ok=True)
        OUTPUT_CKPT_DIR  = os.environ.get('OUTPUT_CKPT_DIR', RUN_DIR)
    else:
        OUTPUT_CKPT_DIR  = os.environ.get(
            'OUTPUT_CKPT_DIR', os.path.join("checkpoints", "v11_transfer"))
    V10_SOURCE_CKPT      = os.path.join("checkpoints", "v10_source", "best_model.pth")
    BEST_MODEL_PATH      = os.path.join(OUTPUT_CKPT_DIR, "best_model.pth")
    BEST_RAW_F1_025_PATH = os.path.join(OUTPUT_CKPT_DIR, "best_model_raw_f1_025.pth")
    BEST_VAL_LOSS_PATH   = os.path.join(OUTPUT_CKPT_DIR, "best_model_val_loss.pth")
    LAST_CHECKPOINT_PATH = os.path.join(OUTPUT_CKPT_DIR, "last_checkpoint.pth")
    FINAL_WEIGHTS_PATH   = os.path.join(OUTPUT_CKPT_DIR, "morai_autonav_weights.pth")
    THROUGHPUT_JSONL_PATH = os.path.join(OUTPUT_CKPT_DIR, "throughput.jsonl")
    os.makedirs(OUTPUT_CKPT_DIR, exist_ok=True)
    # ── v11 전이학습 (MIGRATION_PLAN §5) ──────────────────
    # TRANSFER_FROM_V10 이 설정되면 v10 단일프레임 가중치를 규칙 기반 부분전이하고
    # 3단 옵티마이저(backbone×0.1 / 전이부×0.5 / 신규×1.0)로 처음부터 학습한다.
    # 이때 RESUME_FROM(full-checkpoint 재개)은 무시된다. None이면 기존 동작 유지.
    TRANSFER_FROM_V10    = os.environ.get('TRANSFER_FROM_V10') or None  # 예: "best_model.pth"
    TRANSFER_BASE_LR     = 2e-4
    _transfer_requested  = (
        TRANSFER_FROM_V10 is not None and os.path.isfile(str(TRANSFER_FROM_V10))
    )

    # ── resume vs transfer 명확 분리 (요구사항 D) ────────────────────
    # 두 모드는 상호배타다. resume이 선택되면 TRANSFER_FROM_V10은 절대 사용하지 않는다.
    # RESUME env:
    #   미설정/auto/1  → last_checkpoint.pth 있으면 그것으로 full-resume.
    #                    (없고 transfer 요청도 없으면 best_model.pth로 weight warm-start)
    #   <경로>         → 그 경로에서 resume
    #   0/none/off     → resume 안 함 (scratch 또는 transfer)
    # 우선순위: resume(명시/자동 checkpoint) > transfer > scratch.
    _resume_env = (os.environ.get('RESUME') or '').strip()
    _rl = _resume_env.lower()
    if _rl in ('0', 'none', 'off', 'false', 'no'):
        RESUME_FROM = None
    elif _rl in ('', '1', 'auto', 'true', 'yes'):
        if os.path.isfile(LAST_CHECKPOINT_PATH):
            RESUME_FROM = LAST_CHECKPOINT_PATH
        elif (not _transfer_requested) and os.path.isfile(BEST_MODEL_PATH):
            RESUME_FROM = BEST_MODEL_PATH   # 기존 fallback: weight warm-start
        else:
            RESUME_FROM = None
        if _rl in ('1', 'true', 'yes') and RESUME_FROM is None:
            print(f"[학습] RESUME 요청됐지만 재개할 checkpoint 없음: {LAST_CHECKPOINT_PATH}")
    else:
        RESUME_FROM = _resume_env   # 사용자 지정 경로

    RESUMING = RESUME_FROM is not None and os.path.isfile(RESUME_FROM)
    if RESUMING and _transfer_requested:
        print(f"⚠️  [학습] resume 모드 활성 → TRANSFER_FROM_V10({TRANSFER_FROM_V10}) 무시. "
              f"checkpoint에서 재개: {RESUME_FROM}")
        TRANSFER_FROM_V10 = None
        _transfer_requested = False
    TRANSFER_MODE = _transfer_requested

    # ── Phase 2: loss 계약 선택 + weight-only 초기화 ──────────────────
    # LOSS_CONTRACT:
    #   legacy (기본) → 기존 CustomLoss + compute_auxiliary_detection_loss (동작 불변)
    #   parity        → 공식 Stage-1 계약 (ParityLoss + batch num_pos 정규화,
    #                   decoder 동일가중 합산) — Phase 2 repair pilot / 재학습용
    # INIT_WEIGHTS: model weight 만 로드 (optimizer/scheduler/step 은 초기화 상태 유지).
    #   repair pilot 계약: RESUME 과 상호배타 — resume 은 optimizer 까지 복원하므로
    #   "새 objective + 새 optimizer" 라는 pilot 정의와 충돌한다.
    LOSS_CONTRACT = (os.environ.get('LOSS_CONTRACT', 'legacy') or 'legacy').strip().lower()
    if LOSS_CONTRACT not in ('legacy', 'parity'):
        raise ValueError(f"LOSS_CONTRACT must be legacy|parity: {LOSS_CONTRACT}")
    INIT_WEIGHTS = os.environ.get('INIT_WEIGHTS') or None
    if INIT_WEIGHTS and not os.path.isfile(INIT_WEIGHTS):
        raise FileNotFoundError(f"INIT_WEIGHTS 파일 없음: {INIT_WEIGHTS}")
    if INIT_WEIGHTS and RESUMING:
        raise ValueError("INIT_WEIGHTS 와 RESUME 은 함께 쓸 수 없습니다 (pilot 계약).")
    # preflight: 처음 N optimizer update 동안 grad norm/clip/loss 성분/num_pos 기록
    PREFLIGHT_UPDATES = int(os.environ.get('PREFLIGHT_UPDATES', '300'))
    # 진단용 조기 종료: N optimizer update 후 정상 checkpoint 저장하고 종료.
    # scheduler horizon(NUM_EPOCHS 기반 cosine 길이)은 그대로 유지되므로,
    # 확장 preflight 를 "본 학습과 동일한 LR 궤적의 앞부분"으로 관측할 수 있다.
    # (1-epoch 축소 horizon 으로 돌리면 후반이 cosine 급감쇠라 grad 안정화와
    #  LR 강제 하락을 구분할 수 없음 — 2026-07-31 동료 리뷰.) 0=비활성.
    STOP_AFTER_UPDATES = int(os.environ.get('STOP_AFTER_UPDATES', '0'))

    # optimizer 계약 (scratch parity run 용):
    #   legacy (기본) → backbone 1e-5/wd 1e-3, head 5e-5/wd 1e-2, warmup 0→1 (기존 동작)
    #   parity        → 공식 batch-scaled: backbone 2.5e-5(scale 0.5), 전체 wd 1e-3,
    #                   warmup 을 1/3 에서 시작 (mmcv warmup_ratio=1/3)
    OPTIM_CONTRACT = (os.environ.get('OPTIM_CONTRACT', 'legacy') or 'legacy').strip().lower()
    if OPTIM_CONTRACT not in ('legacy', 'parity'):
        raise ValueError(f"OPTIM_CONTRACT must be legacy|parity: {OPTIM_CONTRACT}")
    WARMUP_START_RATIO = (1.0 / 3.0) if OPTIM_CONTRACT == 'parity' else 0.0
    # 정기 checkpoint 주기(epoch). 기본 10 = 기존 동작. scratch 30ep run 은 5 로 설정해
    # epoch 5/10/.../30 스냅샷을 남기고 AP 평가에 사용한다.
    CHECKPOINT_EVERY_EPOCHS = int(os.environ.get('CHECKPOINT_EVERY_EPOCHS', '10'))
    if CHECKPOINT_EVERY_EPOCHS < 1:
        raise ValueError("CHECKPOINT_EVERY_EPOCHS must be >= 1")

    HISTORY_CSV_PATH     = (os.path.join(RUN_DIR, "training_history.csv")
                            if RUN_DIR else "training_history.csv")
    HISTORY_PLOT_PATH    = (os.path.join(RUN_DIR, "training_curves.png")
                            if RUN_DIR else "training_curves.png")
    # 매 epoch train loss 분해(det/cls/box/quality/depth)를 남기는 텍스트 로그.
    # training_history.csv 는 validation cadence(예: 5 epoch)마다만 기록되므로 별도로 둔다.
    EPOCH_LOSS_LOG_PATH  = (os.path.join(RUN_DIR, "epoch_losses.csv")
                            if RUN_DIR else "epoch_losses.csv")
    WARMUP_STEPS         = 500
    MIN_LR_RATIO         = 1e-3
    if RESUMING:
        FORCE_REMAKE_KMEANS = False   # resume 시 anchor 재생성 금지 (checkpoint 정합)

    # ── 실험용 step 기반 루프 / fast mode (요구사항 A,B) ──────────────
    # 기본값(모두 미설정)이면 기존 full-epoch 동작 그대로. FAST_MODE=1이면
    # 단일 GPU 실험용 기본값(짧은 val 주기 + val subsample)을 켠다.
    def _envflag(name, default=False):
        v = os.environ.get(name)
        if v is None:
            return default
        return v.strip().lower() not in ('0', '', 'false', 'no', 'off')

    FAST_MODE = _envflag('FAST_MODE', False)
    # optimizer update N회마다 val+checkpoint (0 = 기존 epoch 단위 검증만).
    VAL_EVERY_STEPS = int(os.environ.get(
        'VAL_EVERY_STEPS', '1000' if FAST_MODE else '0'))
    # 한 epoch에서 최대 dataloader step 수 (0 = 제한 없음 / full epoch).
    MAX_STEPS_PER_EPOCH = int(os.environ.get('MAX_STEPS_PER_EPOCH', '0'))
    # validation에서 사용할 최대 프레임 수 (0 = 전체 val). fast 검증 속도용.
    FAST_VAL_MAX_FRAMES = int(os.environ.get(
        'FAST_VAL_MAX_FRAMES', '1000' if FAST_MODE else '0'))
    # ───────────────────────────────────────────────────

    print("SparseDrive-style 3-camera detection 학습 시작! [vehicle + pedestrian]")
    print(f"   - val 시나리오: {'auto(last 5)' if VAL_SCENARIOS is None else VAL_SCENARIOS}")
    print(f"   - kmeans anchor: train split only, K={KMEANS_K}, force_remake={FORCE_REMAKE_KMEANS}")
    print(
        f"   - best 기준   : {PRIMARY_BEST_MODE} "
        f"{PRIMARY_BEST_KEY}@score>={PRIMARY_BEST_THR:.2f} "
        f"(raw_f1@0.25/val_loss 별도 저장)"
    )
    print(f"   - early stop  : patience={EARLY_STOP_PATIENCE}, min_delta={EARLY_STOP_MIN_DELTA}")
    print(f"   - metric      : Precision/Recall@{RECALL_THR}m, 매 {METRIC_EVERY} validation")
    print(f"   - validation  : every {VALIDATE_EVERY_EPOCHS} epoch(s), final epoch always")
    print(f"   - input size  : {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"   - decoder     : SparseDrive-style 6 refinement layers + auxiliary loss")
    print(f"   - aggregation : CUDA deformable op + learned camera embedding, grid_sample fallback")
    print(f"   - quality     : centerness/yawness head, score = foreground * centerness")
    print(f"   - best metric : primary={PRIMARY_BEST_MODE}/{PRIMARY_BEST_KEY}@{PRIMARY_BEST_THR:.2f}")
    print(f"   - temporal    : instance bank={USE_TEMPORAL_MEMORY}, temp={NUM_TEMP_INSTANCES}")
    print(f"   - backbone BN : freeze={FREEZE_BACKBONE_BN}")
    print(f"   - AMP/TF32    : amp={USE_AMP}, tf32={ALLOW_TF32}")
    print(f"   - batch       : {BATCH_SIZE} x accum {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"   - dataloader  : workers={NUM_WORKERS}")
    print(f"   - gt_version  : train={TRAIN_GT_VERSION} / val={VAL_GT_VERSION} "
          f"(v3=source-time 보정 GT+scene_info_v3, v2=기존 GT)"
          f" | velocity_valid mask=raw-only")
    print(f"   - dense_depth : {USE_DENSE_DEPTH} (env override={_udd_env!r})\n")
    _loop_desc = (
        f"STEP (val+ckpt {VAL_EVERY_STEPS} update마다)" if VAL_EVERY_STEPS > 0
        else "FULL-EPOCH"
    )
    print(f"   - loop mode   : {_loop_desc}"
          f"{' | max_steps/epoch=%d' % MAX_STEPS_PER_EPOCH if MAX_STEPS_PER_EPOCH > 0 else ''}"
          f"{' | val_max_frames=%d' % FAST_VAL_MAX_FRAMES if FAST_VAL_MAX_FRAMES > 0 else ' | val=full'}")
    print(f"   - run mode    : "
          f"{'RESUME(%s)' % RESUME_FROM if RESUMING else ('TRANSFER(%s)' % TRANSFER_FROM_V10 if TRANSFER_MODE else 'SCRATCH')}")
    print(f"   - occlusion   : min_lidar_pts={OCCLUSION_MIN_PTS} "
          f"({'OFF' if OCCLUSION_MIN_PTS <= 0 else 'train+val GT에서 관측불가 박스 drop'})")
    print(f"   - graph       : {HISTORY_PLOT_PATH} (csv: {HISTORY_CSV_PATH})\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device}\n")

    # ── task C: 대표 검증 split 을 명시적으로 해석·출력·기록한다 ──────────────
    # VAL_SCENARIOS 없이 묵시적으로 잘못된 split 으로 도는 것을 막기 위해, 시작 시
    # 디스크의 시나리오로부터 train/val 을 확정해 출력하고 SPLIT_RECORD 에 남긴다.
    _all_scen = list_scenarios(DATASET_ROOT, label_dir_name=('labels_3d_v2'
                               if TRAIN_GT_VERSION == 'v2' else 'labels_3d_v3'))
    _resolved_val = resolve_val_scenarios(DATASET_ROOT, VAL_SCENARIOS,
                                          label_dir_name=('labels_3d_v2'
                                          if TRAIN_GT_VERSION == 'v2' else 'labels_3d_v3'))
    _resolved_train = [s for s in _all_scen if s not in _resolved_val]
    SPLIT_RECORD = {
        'dataset_root': os.path.abspath(DATASET_ROOT),
        'all_scenarios': _all_scen,
        'train_scenarios': _resolved_train,
        'val_scenarios': _resolved_val,
        'train_gt_version': TRAIN_GT_VERSION,
        'val_gt_version': VAL_GT_VERSION,
    }
    if not _resolved_train:
        raise RuntimeError(f"[split] train 시나리오가 비었습니다. resolved={SPLIT_RECORD}")
    print(f"   - resolved split: TRAIN={_resolved_train} (gt={TRAIN_GT_VERSION}) | "
          f"VAL={_resolved_val} (gt={VAL_GT_VERSION})")

    # ── task D: anchor 정책 ───────────────────────────────────────────────
    #  ANCHOR_DIR 가 있으면: 사전 생성된 versioned anchor 를 **검증만** 하고 사용한다.
    #    - meta 의 anchor SHA-256 과 실제 .npy SHA 가 다르면 자동 재생성하지 않고 fail-fast.
    #    - 모델(generate_anchors_full)이 그 파일을 읽도록 ANCHOR_*_FILE env 를 절대경로로 지정.
    #    - 대표 A/B: 같은 ANCHOR_DIR 를 baseline/candidate 가 공유 → anchor tensor 동일.
    #  없으면: 기존 동작(train GT 로 root anchor 생성). FORCE_REMAKE_KMEANS 로 재생성 여부 제어.
    ANCHOR_DIR = os.environ.get('ANCHOR_DIR')
    if ANCHOR_DIR:
        _axy = os.path.join(ANCHOR_DIR, 'anchor_kmeans_xy.npy')
        _afull = os.path.join(ANCHOR_DIR, 'anchor_kmeans_full.npy')
        _ameta = os.path.join(ANCHOR_DIR, 'anchor_kmeans_meta.json')
        for _p in (_axy, _afull, _ameta):
            if not os.path.isfile(_p):
                raise FileNotFoundError(f"[anchor] ANCHOR_DIR에 파일 없음(자동 생성 안 함): {_p}")
        with open(_ameta, 'r', encoding='utf-8') as _f:
            ANCHOR_META = json.load(_f)
        _got_full = sha256_file(_afull)
        _got_xy = sha256_file(_axy)
        if ANCHOR_META.get('anchor_full_sha256') != _got_full:
            raise SystemExit(
                f"[anchor] SHA-256 mismatch (full): meta={ANCHOR_META.get('anchor_full_sha256')} "
                f"!= actual={_got_full}. 자동 재생성하지 않음 — anchor를 재생성하거나 ANCHOR_DIR를 확인하세요.")
        if ANCHOR_META.get('anchor_xy_sha256') != _got_xy:
            raise SystemExit(
                f"[anchor] SHA-256 mismatch (xy): meta={ANCHOR_META.get('anchor_xy_sha256')} "
                f"!= actual={_got_xy}. 자동 재생성하지 않음.")
        # 파일 무결성(SHA)만으로는 '엉뚱한 split 의 올바른 파일'을 못 막는다. meta 가 현재
        # resolved train/val, k, seed, anchor GT version, 입력 train-label aggregate hash 와
        # 일치하는지도 검사한다(3-scene anchor 를 150-scene split 에 지정 시 GPU 이전 fail-fast).
        ANCHOR_SEED = int(os.environ.get('ANCHOR_SEED', '42'))
        # anchor 는 '자기 자신의 GT'(meta.gt_version, 예: v3)로 검증한다. 대표 A/B 는
        # baseline(train v2)·candidate(train v3)가 동일한 v3 anchor 를 공유하므로 anchor GT 를
        # run 의 TRAIN_GT_VERSION 에 묶지 않는다. 입력 label hash 도 anchor 의 GT dir 로 재계산해
        # wrong-split(train_scenarios·hash)·stale-label(hash)·wrong-seed 를 GPU 이전에 잡는다.
        _anchor_gt = str(ANCHOR_META.get('gt_version'))
        _expected_anchor_gt = os.environ.get('ANCHOR_GT_VERSION') or _anchor_gt
        _alabel = label_dir_for_gt(_anchor_gt)
        _cur_input_sha, _ = hash_train_label_files(DATASET_ROOT, VAL_SCENARIOS, _alabel)
        _ok, _mm = anchor_meta_matches_run(
            ANCHOR_META, k=KMEANS_K, gt_version=_expected_anchor_gt,
            train_scenarios=_resolved_train, val_scenarios=_resolved_val,
            seed=ANCHOR_SEED, input_label_sha256=_cur_input_sha)
        if not _ok:
            raise SystemExit(
                "[anchor] ANCHOR_DIR meta 가 현재 split/데이터와 불일치 — 자동 재생성하지 않음:\n  "
                + "\n  ".join(_mm))
        os.environ['ANCHOR_FULL_FILE'] = os.path.abspath(_afull)
        os.environ['ANCHOR_XY_FILE'] = os.path.abspath(_axy)
        SPLIT_RECORD['anchor_dir'] = os.path.abspath(ANCHOR_DIR)
        SPLIT_RECORD['anchor_full_sha256'] = _got_full
        SPLIT_RECORD['anchor_gt_version'] = ANCHOR_META.get('gt_version')
        SPLIT_RECORD['anchor_train_scenarios'] = ANCHOR_META.get('train_scenarios')
        print(f"[anchor] versioned anchor 검증 통과: {ANCHOR_DIR} | "
              f"gt={ANCHOR_META.get('gt_version')} k={ANCHOR_META.get('k')} "
              f"sha={_got_full[:16]} train={ANCHOR_META.get('train_scenarios')}")
    else:
        ANCHOR_META = None
        ensure_kmeans_files(
            dataset_root=DATASET_ROOT,
            val_scenarios=VAL_SCENARIOS,
            k=KMEANS_K,
            xy_out=DEFAULT_XY_OUT,
            full_out=DEFAULT_FULL_OUT,
            meta_out=DEFAULT_META_OUT,
            force=FORCE_REMAKE_KMEANS,
            gt_version=TRAIN_GT_VERSION,
        )
        if os.path.isfile(DEFAULT_FULL_OUT):
            SPLIT_RECORD['anchor_full_sha256'] = sha256_file(DEFAULT_FULL_OUT)
        SPLIT_RECORD['anchor_dir'] = None

    # ── run provenance JSON(Phase 2): 시작 시 실행 설정을 OUTPUT_CKPT_DIR 에 남긴다 ──
    def _git_out(*a):
        try:
            import subprocess
            return subprocess.run(['git', *a], capture_output=True, text=True).stdout.strip()
        except Exception:
            return None
    try:
        _input_label_sha, _ = hash_train_label_files(
            DATASET_ROOT, VAL_SCENARIOS, label_dir_for_gt(TRAIN_GT_VERSION))
    except Exception:
        _input_label_sha = None
    RUN_CONFIG = {
        'git_head': _git_out('rev-parse', 'HEAD'),
        'git_dirty_diff_sha256': (hashlib.sha256((_git_out('diff') or '').encode()).hexdigest()),
        'output_ckpt_dir': os.path.abspath(OUTPUT_CKPT_DIR),
        'run_dir': (os.path.abspath(RUN_DIR) if RUN_DIR else None),
        'train_scenarios': _resolved_train, 'val_scenarios': _resolved_val,
        'train_gt_version': TRAIN_GT_VERSION, 'val_gt_version': VAL_GT_VERSION,
        'anchor_dir': SPLIT_RECORD.get('anchor_dir'),
        'anchor_full_sha256': SPLIT_RECORD.get('anchor_full_sha256'),
        'anchor_gt_version': SPLIT_RECORD.get('anchor_gt_version'),
        'input_label_sha256': _input_label_sha,
        'seed': (int(SEED) if SEED not in (None, '') else None),
        'use_dense_depth': USE_DENSE_DEPTH,
        'batch_size': BATCH_SIZE, 'grad_accum_steps': GRAD_ACCUM_STEPS,
        'num_workers': NUM_WORKERS,
        'num_epochs': NUM_EPOCHS, 'kmeans_k': KMEANS_K, 'recall_thr': RECALL_THR,
        'early_stop_patience': EARLY_STOP_PATIENCE,
        'metric_every': METRIC_EVERY,
        'validate_every_epochs': VALIDATE_EVERY_EPOCHS,
        'fast_val_max_frames': FAST_VAL_MAX_FRAMES,
        'use_temporal_memory': USE_TEMPORAL_MEMORY,
        'use_streaming_sampler': USE_STREAMING_SAMPLER,
        'temp_gnn_mode': TEMP_GNN_MODE,
        'sequence_length': SEQUENCE_LENGTH,
        'num_temp_instances': NUM_TEMP_INSTANCES,
        'use_amp': USE_AMP, 'allow_tf32': ALLOW_TF32,
        'freeze_backbone_bn': FREEZE_BACKBONE_BN,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'device': str(device),
        'gpu_name': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        # 실제 모드는 첫 forward 때 결정된다(CUDA op 성공 시 CUDA, 실패 시 grid_sample fallback).
        # 예전엔 무조건 'grid_sample (no nvcc)'로 잘못 고정돼 있었음(RTX 5070에선 CUDA op 정상 동작).
        'deformable_agg': 'CUDA deformable op preferred; grid_sample fallback only on CUDA op failure',
    }
    with open(os.path.join(OUTPUT_CKPT_DIR, 'run_config.json'), 'w', encoding='utf-8') as _f:
        json.dump(RUN_CONFIG, _f, indent=2, ensure_ascii=False)
    print(f"[run_config] {os.path.join(OUTPUT_CKPT_DIR, 'run_config.json')} "
          f"seed={RUN_CONFIG['seed']} depth={USE_DENSE_DEPTH} anchor_sha={str(SPLIT_RECORD.get('anchor_full_sha256'))[:16]}")

    model = AutoNavModel(
        num_decoder_layers=6,
        pretrained_backbone=True,
        use_temporal_memory=USE_TEMPORAL_MEMORY,
        num_temp_instances=NUM_TEMP_INSTANCES,
        use_grid_mask=USE_GRID_MASK,
        use_dense_depth=USE_DENSE_DEPTH,
    ).to(device)
    if FREEZE_BACKBONE_BN:
        model.freeze_backbone_bn()

    # temp_gnn ablation 적용(temporal memory ON일 때만 실질 효과).
    #   off  : temp_gnn fusion 완전 비활성(A1-a) — sampler/bank 안전성 검증용
    #   on   : gate 강제 1.0(구 붕괴 동작 재현)
    #   gated: 0-init learnable gate 유지(기본, A1-b)
    if USE_TEMPORAL_MEMORY:
        _n_gnn = 0
        for _layer in model.decoder_layers:
            if not getattr(_layer, 'use_temp_gnn', False):
                continue
            _n_gnn += 1
            if TEMP_GNN_MODE == 'off':
                _layer.temp_gnn_enabled = False
            elif TEMP_GNN_MODE == 'on':
                _layer.temp_gnn_enabled = True
                with torch.no_grad():
                    _layer.temp_gate.fill_(1.0)
            else:  # 'gated'
                _layer.temp_gnn_enabled = True
        print(f"   - temp_gnn    : mode={TEMP_GNN_MODE} "
              f"(gnn layers={_n_gnn}, gate={'1.0(forced)' if TEMP_GNN_MODE=='on' else ('disabled' if TEMP_GNN_MODE=='off' else '0-init learnable')})")

    # v11 전이: v10 가중치를 규칙 기반 부분전이 (옵티마이저 그룹 구성보다 먼저)
    if TRANSFER_MODE:
        load_v10_weights(model, TRANSFER_FROM_V10)

    # Phase 2 repair pilot: model weight 만 로드. optimizer/scheduler/step 은 그대로
    # 초기 상태(pilot 계약: "weight = checkpoint, 나머지 = 새로").
    if INIT_WEIGHTS:
        _ck = torch.load(INIT_WEIGHTS, map_location='cpu', weights_only=False)
        _sd = _ck.get('model_state', _ck) if isinstance(_ck, dict) else _ck
        _missing, _unexpected = model.load_state_dict(_sd, strict=False)
        # depth head 유무 차이만 허용 — 그 외 불일치는 다른 아키텍처이므로 즉시 중단
        _bad_m = [k for k in _missing if not k.startswith('depth_net.')]
        _bad_u = [k for k in _unexpected if not k.startswith('depth_net.')]
        if _bad_m or _bad_u:
            raise RuntimeError(
                f"INIT_WEIGHTS 아키텍처 불일치: missing={_bad_m[:5]}, unexpected={_bad_u[:5]}")
        print(f"[학습] INIT_WEIGHTS 로드: {INIT_WEIGHTS} "
              f"(depth_net 제외 missing={len(_missing)-len(_bad_m)}, "
              f"unexpected={len(_unexpected)-len(_bad_u)})")

    dataset_cls = MoraiTemporalDataset if USE_TEMPORAL_DATASET else MoraiDataset
    collate_fn = morai_temporal_collate_fn if USE_TEMPORAL_DATASET else morai_collate_fn
    if USE_TEMPORAL_DATASET:
        # task B: train 은 TRAIN_GT_VERSION, val 은 VAL_GT_VERSION 으로 각각 로드한다.
        # task H: load_depth=USE_DENSE_DEPTH → depth OFF 시 depth_gt I/O 도 생략(동일 조건 양쪽 적용).
        train_ds = dataset_cls(dataset_root=DATASET_ROOT, split='train', val_scenarios=VAL_SCENARIOS,
                               photometric_aug=USE_PHOTOMETRIC_AUG, load_depth=USE_DENSE_DEPTH,
                               occlusion_min_pts=OCCLUSION_MIN_PTS, gt_version=TRAIN_GT_VERSION)
        val_ds   = dataset_cls(dataset_root=DATASET_ROOT, split='val',   val_scenarios=VAL_SCENARIOS,
                               load_depth=USE_DENSE_DEPTH,
                               occlusion_min_pts=OCCLUSION_MIN_PTS, gt_version=VAL_GT_VERSION)
    else:
        train_ds = dataset_cls(dataset_root=DATASET_ROOT, split='train', val_scenarios=VAL_SCENARIOS)
        val_ds   = dataset_cls(dataset_root=DATASET_ROOT, split='val',   val_scenarios=VAL_SCENARIOS)

    if USE_TEMPORAL_MEMORY:
        train_pose_count, train_pose_total = count_ego_pose_files(train_ds)
        val_pose_count, val_pose_total = count_ego_pose_files(val_ds)
        print(
            f"[temporal] ego_pose 파일: "
            f"train {train_pose_count}/{train_pose_total}, "
            f"val {val_pose_count}/{val_pose_total}"
        )
        if train_pose_count == 0:
            print(
                "[temporal] 현재 dataset에는 ego_pose가 없어 "
                "정확한 ego-motion alignment/temp_gnn은 비활성 상태로 학습됩니다. "
                "morai_3d_live.py로 새로 수집한 데이터부터 활성화됩니다.\n"
            )

    if USE_STREAMING_SAMPLER:
        # temporal memory용 스트리밍: batch slot별 시간순, position↔bank slot 안정.
        # train은 매 배치 정확히 B개(drop_uneven_tail=True)로 포지션을 고정,
        # val은 full coverage(ragged tail) deterministic 스트리밍.
        train_sampler = StreamingGroupSampler(
            train_ds, batch_size=BATCH_SIZE, shuffle=True,
            seed=0, drop_uneven_tail=True,
        )
        val_sampler = StreamingGroupSampler(
            val_ds, batch_size=BATCH_SIZE, shuffle=False,
            seed=0, drop_uneven_tail=False,
        )
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                                  collate_fn=collate_fn, num_workers=NUM_WORKERS)
        val_loader   = DataLoader(val_ds, batch_sampler=val_sampler,
                                  collate_fn=collate_fn, num_workers=NUM_WORKERS)
    else:
        train_sampler = None
        # non-streaming 분기 = streaming을 안 쓰는 경우이므로 항상 shuffle(표준 학습).
        # (기존 도달 케이스는 USE_TEMPORAL_MEMORY=0뿐이라 이전에도 True였음 → 동작 불변.
        #  STREAMING_SAMPLER=0 강제 시 sampler만 shuffle로 바꾸는 대조군을 위해 명시 True.)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=collate_fn, num_workers=NUM_WORKERS)
        val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                                  collate_fn=collate_fn, num_workers=NUM_WORKERS)

    # Dataset/sampler 규모는 실제 wall-time 산정의 분모이며, temporal train sampler는
    # drop_uneven_tail=True라 raw dataset 길이와 epoch당 유효 frame 수가 다를 수 있다.
    _train_frames_per_epoch = (
        len(train_loader) * BATCH_SIZE if USE_STREAMING_SAMPLER else len(train_ds)
    )
    RUN_CONFIG.update({
        'train_dataset_frames': len(train_ds),
        'val_dataset_frames': len(val_ds),
        'train_batches_per_epoch': len(train_loader),
        'val_batches_per_eval': len(val_loader),
        'train_frames_per_epoch': _train_frames_per_epoch,
        'streaming_train_dropped_frames': max(len(train_ds) - _train_frames_per_epoch, 0),
        'loss_contract': LOSS_CONTRACT,
        'optim_contract': OPTIM_CONTRACT,
        'cls_prior_prob': float(os.environ.get('CLS_PRIOR_PROB', '0.1')),
        'checkpoint_every_epochs': CHECKPOINT_EVERY_EPOCHS,
        'init_weights': INIT_WEIGHTS,
        'preflight_updates': PREFLIGHT_UPDATES,
    })
    with open(os.path.join(OUTPUT_CKPT_DIR, 'run_config.json'), 'w', encoding='utf-8') as _f:
        json.dump(RUN_CONFIG, _f, indent=2, ensure_ascii=False)
    print(
        f"[loader] train raw={len(train_ds):,}, effective/epoch={_train_frames_per_epoch:,}, "
        f"batches={len(train_loader):,}, dropped={RUN_CONFIG['streaming_train_dropped_frames']:,} | "
        f"val raw={len(val_ds):,}, batches={len(val_loader):,}"
    )

    # num_classes=2: vehicle, pedestrian (sigmoid focal, 배경 채널 없음)
    # v9: quality_weight=0.2 명시 — centerness/confidence 학습 강화
    if LOSS_CONTRACT == 'parity':
        det_criterion = ParityLoss(num_classes=2).to(device)
        _AUX_LOSS_FN = compute_parity_detection_loss
        print("   - loss 계약  : parity (공식 Stage-1: focal-cost matcher, "
              "batch num_pos 정규화, decoder 동일가중 합산, cls>0.05 reg/qual gate)")
    else:
        det_criterion = CustomLoss(num_classes=2, quality_weight=0.2).to(device)

    if TRANSFER_MODE:
        # 3단 그룹: backbone×0.1 / 전이부×0.5 / 신규(depth·temp·reg velocity)×1.0
        param_groups, group_names = build_transfer_param_groups(
            model, base_lr=TRANSFER_BASE_LR,
        )
        n_by_group = {g: len(names) for g, names in group_names.items()}
        print(f"[전이] 옵티마이저 3단 그룹 (base_lr={TRANSFER_BASE_LR}): "
              f"backbone×0.1={n_by_group['backbone']}, "
              f"loaded×0.5={n_by_group['loaded']}, new×1.0={n_by_group['new']} 텐서")
        wandb_cfg = {
            "transfer_from": TRANSFER_FROM_V10, "base_lr": TRANSFER_BASE_LR,
            "lr_backbone": TRANSFER_BASE_LR * 0.1, "lr_loaded": TRANSFER_BASE_LR * 0.5,
            "lr_new": TRANSFER_BASE_LR, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM_STEPS,
        }
        wandb_name, wandb_tags = "v11-transfer", ["v11", "transfer"]
        optimizer = optim.AdamW(param_groups)
    else:
        backbone_params = list(model.backbone.parameters())
        backbone_ids    = set(id(p) for p in backbone_params)
        other_params    = [p for p in model.parameters() if id(p) not in backbone_ids]
        # wandb 메타데이터는 실제 optimizer(아래 AdamW) 값과 반드시 일치시킨다.
        # (과거 2e-5/1e-4 로 잘못 기록돼 있었음 — 실제 LR 은 1e-5/5e-5)
        wandb_cfg = {
            "lr_backbone": 1e-5, "lr_head": 5e-5, "wd_backbone": 1e-3,
            "wd_head": 1e-2, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM_STEPS,
        }
        wandb_name, wandb_tags = "v10", ["v10"]
        if OPTIM_CONTRACT == 'parity':
            # 공식 stage-1 을 effective batch 8/64 로 스케일: head 4e-4×1/8=5e-5,
            # backbone lr_mult 0.5 → 2.5e-5, weight decay 는 전 그룹 1e-3.
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': 2.5e-5, 'weight_decay': 1e-3},
                {'params': other_params,    'lr': 5e-5,   'weight_decay': 1e-3},
            ])
            wandb_cfg.update({"lr_backbone": 2.5e-5, "wd_head": 1e-3,
                              "optim_contract": "parity"})
            print("   - optim 계약 : parity (backbone 2.5e-5/wd 1e-3, head 5e-5/wd 1e-3, "
                  "warmup 1/3 시작)")
        else:
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': 1e-5,  'weight_decay': 1e-3},
                {'params': other_params,    'lr': 5e-5,  'weight_decay': 1e-2},
            ])

    wandb.init(
        project="morai-3d-detection",
        name=wandb_name,
        tags=wandb_tags,
        config=wandb_cfg,
    )

    updates_per_epoch = max(math.ceil(len(train_loader) / GRAD_ACCUM_STEPS), 1)
    total_update_steps = max(updates_per_epoch * NUM_EPOCHS, 1)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            # legacy: 0→1 선형 (WARMUP_START_RATIO=0, 기존과 동일).
            # parity: 공식 warmup_ratio=1/3 → base·1/3 에서 시작해 base 까지 선형.
            lin = float(step + 1) / float(max(WARMUP_STEPS, 1))
            return max(WARMUP_START_RATIO + (1.0 - WARMUP_START_RATIO) * lin, MIN_LR_RATIO)
        denom = max(total_update_steps - WARMUP_STEPS, 1)
        progress = min(float(step - WARMUP_STEPS) / float(denom), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return MIN_LR_RATIO + (1.0 - MIN_LR_RATIO) * cosine

    base_lrs = [group['lr'] for group in optimizer.param_groups]

    def set_optimizer_lr(step):
        factor = lr_lambda(step)
        for group, base_lr in zip(optimizer.param_groups, base_lrs):
            group['lr'] = base_lr * factor

    global_update_step = 0
    set_optimizer_lr(global_update_step)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and USE_AMP))

    best_recall       = -1.0          # best 기준을 val recall로 변경 (높을수록 좋음)
    best_val_loss     = float('inf')  # recall 동률 시 tie-break용
    best_scores = {
        'primary': -1.0,
        'raw_f1_025': -1.0,
        'val_loss': float('inf'),
    }
    epochs_no_improve = 0
    start_epoch = 0
    history_records = []

    if RESUME_FROM is not None and os.path.isfile(RESUME_FROM):
        ckpt = torch.load(RESUME_FROM, map_location=device)
        if isinstance(ckpt, dict) and 'model_state' in ckpt:
            # cross-run resume 방지(Phase 0): checkpoint 가 다른 split/anchor/GT 로 만들어졌으면
            # 조용히 이어받지 않고 fail-fast 한다(잘못된 run checkpoint resume 차단).
            if ckpt.get('train_scenarios') is not None:
                _rmm = []
                if list(ckpt.get('train_scenarios') or []) != list(_resolved_train):
                    _rmm.append(f"train_scenarios ckpt={ckpt.get('train_scenarios')} run={_resolved_train}")
                if list(ckpt.get('val_scenarios') or []) != list(_resolved_val):
                    _rmm.append(f"val_scenarios ckpt={ckpt.get('val_scenarios')} run={_resolved_val}")
                if str(ckpt.get('train_gt_version')) != TRAIN_GT_VERSION:
                    _rmm.append(f"train_gt ckpt={ckpt.get('train_gt_version')} run={TRAIN_GT_VERSION}")
                if (ckpt.get('val_gt_version') is not None and
                        str(ckpt.get('val_gt_version')) != VAL_GT_VERSION):
                    _rmm.append(f"val_gt ckpt={ckpt.get('val_gt_version')} run={VAL_GT_VERSION}")
                _ck_a = ckpt.get('anchor_full_sha256'); _run_a = SPLIT_RECORD.get('anchor_full_sha256')
                if _ck_a and _run_a and _ck_a != _run_a:
                    _rmm.append(f"anchor_sha ckpt={_ck_a[:12]} run={_run_a[:12]}")
                _resume_contract = {
                    'batch_size': BATCH_SIZE,
                    'grad_accum_steps': GRAD_ACCUM_STEPS,
                    'use_temporal_memory': USE_TEMPORAL_MEMORY,
                    'use_streaming_sampler': USE_STREAMING_SAMPLER,
                    'temp_gnn_mode': TEMP_GNN_MODE,
                    'sequence_length': SEQUENCE_LENGTH,
                    'use_dense_depth': USE_DENSE_DEPTH,
                    'use_amp': USE_AMP,
                    'allow_tf32': ALLOW_TF32,
                    'validate_every_epochs': VALIDATE_EVERY_EPOCHS,
                    'fast_val_max_frames': FAST_VAL_MAX_FRAMES,
                    'max_steps_per_epoch': MAX_STEPS_PER_EPOCH,
                    # Phase 2 계약: legacy checkpoint 를 parity 로(또는 반대로) 조용히
                    # 이어받거나, NUM_EPOCHS 변경으로 cosine 길이가 달라지는 것을 차단.
                    'loss_contract': LOSS_CONTRACT,
                    'optim_contract': OPTIM_CONTRACT,
                    'num_epochs': NUM_EPOCHS,
                    'warmup_steps': WARMUP_STEPS,
                    'min_lr_ratio': MIN_LR_RATIO,
                    # 실효 LR/WD 값 자체도 비교 — OPTIM_CONTRACT 이름은 같아도 코드에서
                    # parity 값이 바뀌면 감지된다 (계약 버전/hash 의 역할을 값 비교로 수행).
                    'base_lrs': base_lrs,
                    'weight_decays': [g['weight_decay'] for g in optimizer.param_groups],
                }
                for _key, _run_value in _resume_contract.items():
                    if _key in ckpt and ckpt.get(_key) != _run_value:
                        _rmm.append(
                            f"{_key} ckpt={ckpt.get(_key)!r} run={_run_value!r}"
                        )
                if _rmm:
                    raise SystemExit(
                        "[resume] checkpoint 가 현재 run 과 불일치 — 다른 run 의 checkpoint 를 "
                        "resume 하려 함(fail-fast):\n  " + "\n  ".join(_rmm))
            ckpt_state = ckpt['model_state']
            model_state = model.state_dict()
            filtered = {k: v for k, v in ckpt_state.items()
                        if k in model_state and v.shape == model_state[k].shape}
            # v9: cls_branch는 v8 가중치를 그대로 이어받는다.
            #     (v8에서 하던 prior_prob 재초기화용 명시적 제거를 비활성화)
            # for k in [k for k in list(filtered) if 'cls_branch' in k]:
            #     del filtered[k]
            skipped = [k for k in ckpt_state if k not in filtered]
            model_state.update(filtered)
            model.load_state_dict(model_state)
            print(f"[체크포인트] {len(filtered)}/{len(ckpt_state)} 파라미터 로드, {len(skipped)}개 스킵: {skipped[:3]}")
            if ckpt.get('optimizer_state') is not None:
                optimizer.load_state_dict(ckpt['optimizer_state'])
                print("[체크포인트] optimizer state 복원")
            if 'scaler_state' in ckpt and ckpt['scaler_state'] is not None:
                scaler.load_state_dict(ckpt['scaler_state'])
            saved_epoch = int(ckpt.get('epoch', -1))
            # epoch_completed=False(step 모드 mid-epoch 저장)면 해당 epoch을 다시 시작.
            # 구 포맷 checkpoint엔 이 키가 없으므로 True(=epoch 끝에서 저장)로 간주.
            epoch_completed = bool(ckpt.get('epoch_completed', True))
            start_epoch = saved_epoch + 1 if epoch_completed else max(saved_epoch, 0)
            global_update_step = int(ckpt.get('global_update_step', global_update_step))
            best_recall = float(ckpt.get('best_recall', best_recall))
            best_val_loss = float(ckpt.get('best_val_loss', best_val_loss))
            best_scores.update(ckpt.get('best_scores', {}))
            # (best_scores['val_loss'] 는 체크포인트 값을 그대로 사용 — primary tie-break용
            #  best_val_loss 와 섞지 않는다. 예전엔 min 으로 커플링돼 두 지표가 오염됐음.)
            epochs_no_improve = int(ckpt.get('epochs_no_improve', epochs_no_improve))
            # resume 는 같은 실험의 '연속'이므로 patience 카운터를 그대로 유지한다.
            # (예전엔 여기서 0으로 강제 리셋해 재시작마다 early-stop 이 처음부터 다시 세는 결함이
            #  있었음 — 중단되어야 할 학습이 계속 이어질 수 있었다.)
            history_records = ckpt.get('history_records', [])
            if not history_records:
                history_records = load_training_history(HISTORY_CSV_PATH, max_epoch=start_epoch)
            set_optimizer_lr(global_update_step)
            print(
                f"[학습] full checkpoint에서 재개: {RESUME_FROM} | "
                f"start_epoch={start_epoch + 1}, update_step={global_update_step}\n"
            )
        else:
            # 순수 state_dict warm-start (optimizer/LR/epoch 새로 시작).
            # best_model.pth는 export_inference_state_dict라 depth_net 키가 없다 →
            # strict 로드는 실패하므로 shape 일치 키만 필터 로드(depth_net은 init 유지).
            model_state = model.state_dict()
            filtered = {k: v for k, v in ckpt.items()
                        if k in model_state and v.shape == model_state[k].shape}
            skipped = [k for k in ckpt if k not in filtered]
            missing = [k for k in model_state if k not in ckpt]
            model_state.update(filtered)
            model.load_state_dict(model_state)
            print(
                f"[학습] 모델 가중치 warm-start: {RESUME_FROM} "
                f"(optimizer/LR/epoch 새로 시작)\n"
                f"   로드 {len(filtered)}/{len(ckpt)} | ckpt에만 있어 스킵 {len(skipped)} | "
                f"모델 신규(init 유지) {len(missing)}개 예: {missing[:3]}\n"
            )
    else:
        if RESUME_FROM is not None:
            print(f"[학습] resume 파일 없음: {RESUME_FROM}")
        print("[학습] 처음부터 학습\n")

    # 검증+로깅+best저장+resume체크포인트를 한 곳에 모은다. full-epoch(끝에서 1회)와
    # step 기반(VAL_EVERY_STEPS update마다) 양쪽에서 동일 코드로 호출한다.
    # 참고: 이 블록은 module-level(__main__) 스코프라 재바인딩은 global 선언이 필요하다.
    eval_counter = len(history_records)   # history CSV/plot x축 (단조 증가)

    def append_throughput_record(record):
        payload = dict(record)
        payload.update({
            'gpu_name': RUN_CONFIG.get('gpu_name'),
            'batch_size': BATCH_SIZE,
            'grad_accum_steps': GRAD_ACCUM_STEPS,
            'num_workers': NUM_WORKERS,
            'use_temporal_memory': USE_TEMPORAL_MEMORY,
            'use_streaming_sampler': USE_STREAMING_SAMPLER,
            'temp_gnn_mode': TEMP_GNN_MODE,
            'use_dense_depth': USE_DENSE_DEPTH,
            'use_amp': USE_AMP,
            'allow_tf32': ALLOW_TF32,
        })
        with open(THROUGHPUT_JSONL_PATH, 'a', encoding='utf-8') as _f:
            _f.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def save_resume_checkpoint(epoch_idx, epoch_completed):
        """Save the resumable state even on epochs where validation is skipped."""
        torch.save({
            'epoch': epoch_idx,
            'epoch_completed': epoch_completed,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict() if scaler is not None else None,
            'global_update_step': global_update_step,
            'best_recall': best_recall,
            'best_val_loss': best_val_loss,
            'best_scores': best_scores,
            'epochs_no_improve': epochs_no_improve,
            'history_records': history_records,
            # Run identity: cross-run/config resume fail-fast evidence.
            'run_id': os.path.abspath(OUTPUT_CKPT_DIR),
            'train_scenarios': _resolved_train,
            'val_scenarios': _resolved_val,
            'train_gt_version': TRAIN_GT_VERSION,
            'val_gt_version': VAL_GT_VERSION,
            'anchor_full_sha256': SPLIT_RECORD.get('anchor_full_sha256'),
            'batch_size': BATCH_SIZE,
            'grad_accum_steps': GRAD_ACCUM_STEPS,
            'use_temporal_memory': USE_TEMPORAL_MEMORY,
            'use_streaming_sampler': USE_STREAMING_SAMPLER,
            'temp_gnn_mode': TEMP_GNN_MODE,
            'sequence_length': SEQUENCE_LENGTH,
            'use_dense_depth': USE_DENSE_DEPTH,
            'use_amp': USE_AMP,
            'allow_tf32': ALLOW_TF32,
            'validate_every_epochs': VALIDATE_EVERY_EPOCHS,
            'fast_val_max_frames': FAST_VAL_MAX_FRAMES,
            'max_steps_per_epoch': MAX_STEPS_PER_EPOCH,
            # Phase 2: loss/optimizer/scheduler 계약 — resume 시 fail-fast 비교 대상.
            # NUM_EPOCHS 는 cosine 전체 길이를 결정하므로 반드시 일치해야 한다.
            'loss_contract': LOSS_CONTRACT,
            'optim_contract': OPTIM_CONTRACT,
            'num_epochs': NUM_EPOCHS,
            'warmup_steps': WARMUP_STEPS,
            'min_lr_ratio': MIN_LR_RATIO,
            'base_lrs': base_lrs,
            'weight_decays': [g['weight_decay'] for g in optimizer.param_groups],
            # 참고용(불일치해도 무해 — model_state 가 bias 를 덮어씀 / 스냅샷 주기만 변경)
            'cls_prior_prob': float(os.environ.get('CLS_PRIOR_PROB', '0.1')),
            'checkpoint_every_epochs': CHECKPOINT_EVERY_EPOCHS,
        }, LAST_CHECKPOINT_PATH)
        print(f"   💾 Resume 체크포인트 저장: {LAST_CHECKPOINT_PATH}"
              f" (epoch={epoch_idx+1}, completed={epoch_completed})")

    def run_validation_and_checkpoint(train_stats, lr, epoch_idx, epoch_completed,
                                      compute_metric):
        """검증→로깅→best 저장→resume checkpoint. early-stop 여부를 반환."""
        global best_val_loss, epochs_no_improve, eval_counter

        if device.type == "cuda":
            torch.cuda.synchronize()
        _val_started = time.perf_counter()
        val_loss, val_metrics = validate(
            model, val_loader, det_criterion, device,
            compute_metric=compute_metric, recall_thr=RECALL_THR,
            max_frames=FAST_VAL_MAX_FRAMES,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        _val_seconds = time.perf_counter() - _val_started
        # validate() stops only at batch boundaries, so count the same prefix of
        # batch indices here instead of assuming max_frames was hit exactly.
        _val_frames = 0
        for _indices in val_loader.batch_sampler:
            if FAST_VAL_MAX_FRAMES and _val_frames >= FAST_VAL_MAX_FRAMES:
                break
            _val_frames += len(_indices)
        append_throughput_record({
            'kind': 'validation',
            'epoch': epoch_idx + 1,
            'seconds': _val_seconds,
            'frames': _val_frames,
            'frames_per_second': (_val_frames / _val_seconds if _val_seconds > 0 else None),
            'compute_metric': bool(compute_metric),
            'max_frames': FAST_VAL_MAX_FRAMES,
        })
        print(f"[throughput] validation epoch={epoch_idx+1} frames={_val_frames} "
              f"seconds={_val_seconds:.3f} fps={_val_frames/max(_val_seconds, 1e-12):.3f}")

        train_loss         = train_stats['loss']
        train_cls_loss     = train_stats['cls']
        train_box_loss     = train_stats['box']
        train_quality_loss = train_stats['quality']
        train_depth_loss   = train_stats['depth']

        eval_counter += 1
        header = f"Epoch {epoch_idx+1}/{NUM_EPOCHS}"
        if not epoch_completed:
            header += f" · upd {global_update_step}"
        msg = (f"\n📊 {header} [eval {eval_counter}] | "
               f"Train: {train_loss:.4f} "
               f"(Cls {train_cls_loss:.4f}, Box {train_box_loss:.4f}, "
               f"Q {train_quality_loss:.4f}, Depth {train_depth_loss:.4f}) | "
               f"Val: {val_loss:.4f} | LR: {lr:.2e}")
        if val_metrics is not None:
            for mode in HISTORY_METRIC_MODES:
                msg += f"\n   └─ Val {mode} P/R/F1@{RECALL_THR}m:"
                for score_thr in HISTORY_SCORE_THRESHOLDS:
                    metric = val_metrics['by_mode'][mode][score_thr]
                    msg += (
                        f" score>={score_thr:.2f} "
                        f"{metric['precision']:.4f}/{metric['recall']:.4f}/{metric['f1']:.4f}"
                    )
            prim_cls = (
                val_metrics.get('by_mode_class', {})
                .get(PRIMARY_BEST_MODE, {})
                .get(PRIMARY_BEST_THR, {})
            )
            if prim_cls:
                msg += (
                    f"\n   └─ Val {PRIMARY_BEST_MODE} per-class P/R/F1@{RECALL_THR}m "
                    f"(score>={PRIMARY_BEST_THR:.2f}):"
                )
                for cls in sorted(CLASS_ID_NAMES):
                    cm = prim_cls.get(cls)
                    if cm is None:
                        continue
                    msg += (
                        f" {CLASS_ID_NAMES[cls]} "
                        f"{cm['precision']:.4f}/{cm['recall']:.4f}/{cm['f1']:.4f}"
                    )
            by_distance = val_metrics.get('by_distance', {})
            if by_distance:
                msg += "\n   └─ Val softcalibrated 거리구간별 P/R/F1@0.15 (matched center dist):"
                for (lo, hi), dm in by_distance.items():
                    cdist_str = (
                        f"{dm['mean_center_dist']:.3f}m" if dm['mean_center_dist'] is not None else "n/a"
                    )
                    msg += (
                        f" [{lo:.0f}-{hi:.0f}m) "
                        f"{dm['precision']:.4f}/{dm['recall']:.4f}/{dm['f1']:.4f} "
                        f"(cdist={cdist_str})"
                    )
        print(msg)

        history_records.append(make_history_record(
            eval_counter,
            train_loss,
            train_cls_loss,
            train_box_loss,
            train_quality_loss,
            train_depth_loss,
            val_loss,
            lr,
            val_metrics,
        ))
        save_training_history(history_records, HISTORY_CSV_PATH, HISTORY_PLOT_PATH)
        print(f"   📈 Loss 그래프 갱신: {HISTORY_PLOT_PATH} | 로그: {HISTORY_CSV_PATH}")

        wandb_log = {
            "epoch": epoch_idx + 1,
            "eval_point": eval_counter,
            "update_step": global_update_step,
            "train/loss": train_loss,                       # total = det + depth
            "train/cls_loss": train_cls_loss,
            "train/box_loss": train_box_loss,
            "train/quality_loss": train_quality_loss,
            "train/depth_loss": train_depth_loss,
            "train/det_loss": train_loss - train_depth_loss,  # val/loss 와 비교 가능한 detection-only
            "val/loss": val_loss,                           # detection-only (eval 은 depth 미실행)
            "lr": lr,
        }
        if val_metrics is not None:
            for mode in HISTORY_METRIC_MODES:
                for score_thr in HISTORY_SCORE_THRESHOLDS:
                    metric = val_metrics['by_mode'][mode][score_thr]
                    wandb_log[f"val/{mode}/precision@{score_thr:.2f}"] = metric['precision']
                    wandb_log[f"val/{mode}/recall@{score_thr:.2f}"] = metric['recall']
                    wandb_log[f"val/{mode}/f1@{score_thr:.2f}"] = metric['f1']
            by_mode_class = val_metrics.get('by_mode_class', {})
            for mode in HISTORY_METRIC_MODES:
                for score_thr in HISTORY_SCORE_THRESHOLDS:
                    cls_metrics = by_mode_class.get(mode, {}).get(score_thr, {})
                    for cls in sorted(CLASS_ID_NAMES):
                        cm = cls_metrics.get(cls)
                        if cm is None:
                            continue
                        cname = CLASS_ID_NAMES[cls]
                        wandb_log[f"val/{mode}/{cname}/precision@{score_thr:.2f}"] = cm['precision']
                        wandb_log[f"val/{mode}/{cname}/recall@{score_thr:.2f}"] = cm['recall']
                        wandb_log[f"val/{mode}/{cname}/f1@{score_thr:.2f}"] = cm['f1']
            for (lo, hi), dm in val_metrics.get('by_distance', {}).items():
                tag = f"dist_{lo:.0f}_{hi:.0f}m"
                wandb_log[f"val/softcalibrated/{tag}/precision@0.15"] = dm['precision']
                wandb_log[f"val/softcalibrated/{tag}/recall@0.15"] = dm['recall']
                wandb_log[f"val/softcalibrated/{tag}/f1@0.15"] = dm['f1']
                if dm['mean_center_dist'] is not None:
                    wandb_log[f"val/softcalibrated/{tag}/mean_center_dist@0.15"] = dm['mean_center_dist']
        wandb.log(wandb_log)

        # ─── Best save: 실사용/분석 기준을 분리 저장 ─────────
        primary_score = metric_value(
            val_metrics, PRIMARY_BEST_MODE, PRIMARY_BEST_THR, PRIMARY_BEST_KEY,
        )
        # Primary best(softcalibrated f1@0.15) 선택 + early-stop.
        #  - genuine_improve : F1 이 min_delta 이상 실제 상승 → best F1 갱신 + patience 리셋
        #  - tie_break       : F1 은 동률(±min_delta)이되 '현재 best 이상'이고 val_loss 가 더 낮음
        #                      → 체크포인트만 더 나은 것으로 교체. best F1 은 낮아지지 않고(max),
        #                        patience 도 리셋하지 않는다(실제 F1 개선이 아니므로 계속 카운트).
        #  ⚠️ 과거엔 F1 이 best 보다 '조금 낮아도'(1e-4 이내) val_loss 만 낮으면 저장+리셋해서
        #     best F1 이 계속 내려가고 early-stop 이 안 걸리는 결함이 있었다.
        genuine_improve = primary_score > best_scores['primary'] + EARLY_STOP_MIN_DELTA
        tie_break = (
            (not genuine_improve)
            and primary_score >= best_scores['primary']     # best 보다 낮은 F1 은 절대 저장 안 함
            and val_loss < best_val_loss
        )
        if genuine_improve or tie_break:
            best_scores['primary'] = max(best_scores['primary'], primary_score)
            best_val_loss = min(best_val_loss, val_loss)
            if genuine_improve:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1   # 동률 교체는 F1 개선이 아님 → early-stop 계속 카운트
            epoch_best_path = save_best_with_epoch(
                model, BEST_MODEL_PATH, epoch_idx + 1,
                f"{PRIMARY_BEST_MODE}_{PRIMARY_BEST_KEY}_{_score_suffix(PRIMARY_BEST_THR)}",
                primary_score,
            )
            _tag = "improve" if genuine_improve else "tie(val_loss↓)"
            print(
                f"   💾 Primary best 저장[{_tag}]: {BEST_MODEL_PATH} | "
                f"{PRIMARY_BEST_MODE} {PRIMARY_BEST_KEY}@{PRIMARY_BEST_THR:.2f}="
                f"{primary_score:.4f} (best={best_scores['primary']:.4f}) | Val Loss: {val_loss:.4f}"
                f"\n      ↳ epoch snapshot: {epoch_best_path}"
            )
        else:
            epochs_no_improve += 1
            print(
                f"   ⏳ Primary best 개선 없음 "
                f"({epochs_no_improve}/{EARLY_STOP_PATIENCE}) | "
                f"current={primary_score:.4f}, best={best_scores['primary']:.4f}"
            )

        raw_f1_025 = metric_value(val_metrics, 'raw', 0.25, 'f1')
        if raw_f1_025 > best_scores['raw_f1_025'] + EARLY_STOP_MIN_DELTA:
            best_scores['raw_f1_025'] = raw_f1_025
            epoch_best_path = save_best_with_epoch(
                model, BEST_RAW_F1_025_PATH, epoch_idx + 1, "raw_f1_025", raw_f1_025,
            )
            print(
                f"   💾 Raw F1@0.25 best 저장: {raw_f1_025:.4f} -> "
                f"{BEST_RAW_F1_025_PATH} ({epoch_best_path})"
            )

        if val_loss < best_scores['val_loss'] - EARLY_STOP_MIN_DELTA:
            best_scores['val_loss'] = val_loss
            # (best_val_loss 는 primary tie-break 전용 → 여기서 갱신하지 않아 서로 분리)
            epoch_best_path = save_best_with_epoch(
                model, BEST_VAL_LOSS_PATH, epoch_idx + 1, "val_loss", val_loss,
            )
            print(
                f"   💾 Val loss best 저장: {val_loss:.4f} -> "
                f"{BEST_VAL_LOSS_PATH} ({epoch_best_path})"
            )

        # Validation을 건너뛰는 epoch도 동일 helper로 저장해 복구 주기를 유지한다.
        save_resume_checkpoint(epoch_idx, epoch_completed)

        return epochs_no_improve >= EARLY_STOP_PATIENCE

    nan_skip_count = 0   # 비유한 loss 배치 스킵 누적(전체 run)
    _stop_after_requested = False
    for epoch in range(start_epoch, NUM_EPOCHS):
        # ─── Train ───────────────────────────────────────
        # 스트리밍 샘플러: epoch마다 시퀀스 순서 reshuffle (프레임 내부순서는 유지)
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        # 시퀀스-일관 photometric 증강 파라미터도 epoch마다 갱신
        if hasattr(train_ds, "set_epoch"):
            train_ds.set_epoch(epoch)
        model.train()
        if FREEZE_BACKBONE_BN:
            model.freeze_backbone_bn()
        if hasattr(model, "reset_temporal_memory"):
            model.reset_temporal_memory()
        print(f"\n========== [Epoch {epoch+1}/{NUM_EPOCHS}] ==========")
        if device.type == "cuda":
            torch.cuda.synchronize()
        _train_started = time.perf_counter()
        _epoch_train_frames = 0
        _epoch_train_steps = 0
        # interval 기준 running 누적 (full-epoch 모드에선 epoch 전체 = 1 interval).
        run = {'loss': 0.0, 'cls': 0.0, 'box': 0.0, 'quality': 0.0, 'depth': 0.0, 'n': 0}
        early_stop = False
        optimizer.zero_grad(set_to_none=True)
        accum_count = 0   # 이번 accumulation 창에 쌓인 '유효(backward 성공)' 배치 수

        for step, batch in enumerate(train_loader):
            if MAX_STEPS_PER_EPOCH and step >= MAX_STEPS_PER_EPOCH:
                break
            images     = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            ego_poses  = batch['ego_pose'].to(device)
            focal      = batch['focal'].to(device) if (USE_DENSE_DEPTH and 'focal' in batch) else None
            timestamps = batch.get('timestamp')   # [B] float64 (temporal dt 정밀도)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and USE_AMP)):
                model_out = model(
                    images,
                    intrinsics,
                    extrinsics,
                    stems=batch['stem'],
                    ego_poses=ego_poses,
                    focal=focal,
                    timestamps=timestamps,
                    return_intermediate=True,
                )
                batch_loss, batch_cls_loss, batch_box_loss, batch_quality_loss = (
                    _AUX_LOSS_FN(
                        model_out,
                        batch,
                        det_criterion,
                        device,
                        aux_weight=AUX_LOSS_WEIGHT,
                    )
                )
                # dense depth 보조 loss (weight 0.2는 DenseDepthNet.loss 내부, fp32 계산)
                batch_depth_loss = images.new_tensor(0.0)
                if model_out.get('depth_pred') is not None:
                    gt_depth_dev = [g.to(device) for g in batch['gt_depth']]
                    batch_depth_loss = model.depth_net.loss(
                        model_out['depth_pred'], gt_depth_dev)
                    batch_loss = batch_loss + batch_depth_loss

            # NaN/Inf 방어: loss 가 비유한이면 backward 하지 않고 이 배치를 스킵한다.
            # accumulation 은 '성공한 backward 수(accum_count)'로 세므로, NaN 배치를 건너뛰어도
            # 이미 쌓인 유효 grad 는 지우지 않고, optimizer step 은 항상 GRAD_ACCUM_STEPS 개의
            # 유효 배치가 모였을 때만 실행된다(부분 누적/스케일 왜곡 방지).
            if not torch.isfinite(batch_loss):
                nan_skip_count += 1
                # 근본원인 진단: 처음 몇 번만 상세(오버헤드는 NaN 배치에서만).
                if nan_skip_count <= 5:
                    diagnose_nonfinite(
                        f"epoch{epoch+1} step{step} | det={batch_loss.item():.3g} "
                        f"depth={float(batch_depth_loss.item()):.3g} stems={batch.get('stem')}",
                        [
                            ("in.images", images), ("in.intrinsics", intrinsics),
                            ("in.extrinsics", extrinsics), ("in.ego_poses", ego_poses),
                            ("in.focal", focal), ("in.gt_depth", batch.get('gt_depth')),
                            ("in.gt_boxes", batch.get('dynamic_gt_boxes')),
                            ("out.det_cls", model_out.get('det_cls')),
                            ("out.det_box", model_out.get('det_box')),
                            ("out.det_quality", model_out.get('det_quality')),
                            ("out.depth_pred", model_out.get('depth_pred')),
                            ("out.all_det_cls", model_out.get('all_det_cls')),
                            ("out.all_det_box", model_out.get('all_det_box')),
                        ],
                        per_layer=("all_det_cls", model_out.get('all_det_cls')),
                    )
                if nan_skip_count <= 20 or nan_skip_count % 100 == 0:
                    print(f"[WARN] 비유한 loss 배치 스킵 (epoch {epoch+1} step {step}, "
                          f"누적 {nan_skip_count})")
                continue

            loss_for_backward = batch_loss / GRAD_ACCUM_STEPS
            scaler.scale(loss_for_backward).backward()
            accum_count += 1

            # 유효 backward 가 GRAD_ACCUM_STEPS 개 모였거나(정확히 그 배수) epoch 마지막 배치이고
            # 창에 쌓인 grad 가 있으면 optimizer step.
            is_last_batch = (step + 1 == len(train_loader))
            should_step = (accum_count >= GRAD_ACCUM_STEPS) or (is_last_batch and accum_count > 0)
            if should_step:
                scaler.unscale_(optimizer)
                _total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 25.0)
                # ── preflight: 첫 PREFLIGHT_UPDATES update 의 grad/clip/loss 성분 기록.
                #    loss 계약 교체 시 clipping 이 비교를 왜곡하는지 검증하는 용도.
                if global_update_step < PREFLIGHT_UPDATES:
                    _pf_path = os.path.join(OUTPUT_CKPT_DIR, 'preflight.csv')
                    _pf_new = not os.path.isfile(_pf_path)
                    _ps = getattr(compute_parity_detection_loss, 'last_stats', {})
                    _np_s = str(_ps.get('num_pos', '')) if LOSS_CONTRACT == 'parity' else ''
                    _gr = _ps.get('gate_rate') if LOSS_CONTRACT == 'parity' else None
                    _gr_s = f"{_gr:.3f}" if isinstance(_gr, float) else ''
                    # temporal telemetry: temp_gate(layer별 학습 gate) / bank 활성 슬롯 /
                    # 직전 forward 의 temporal-active frame 수 / 누적 bank 무효화 횟수.
                    # cls>0.05 gate_rate(회귀 gate)와는 별개의 신호다.
                    _gates = [float(l.temp_gate.detach().float().mean())
                              for l in model.decoder_layers
                              if getattr(l, 'use_temp_gnn', False) and hasattr(l, 'temp_gate')]
                    _gate_s = f"{sum(abs(g) for g in _gates)/max(len(_gates),1):.5f}" if _gates else ''
                    _bank = getattr(model, '_bank', None)
                    _bank_s = str(sum(1 for e in _bank if e is not None)) if _bank else '0'
                    _ta = f"{getattr(model, '_last_temporal_active', 0)}/{getattr(model, '_last_batch_size', 0)}"
                    _rst = str(getattr(model, '_bank_reset_count', 0))
                    # backbone/head grad norm 분리 (pre-clip 값으로 환산해 기록).
                    # clip_grad_norm_ 반환값은 clip 전 total norm 이지만 grad 는 이미
                    # in-place 로 스케일됐으므로, 측정한 post-clip backbone norm 을
                    # clip 배율로 역보정한다. total² = backbone² + head².
                    _tn = float(_total_norm)
                    _bb_sq = 0.0
                    for _p in model.backbone.parameters():
                        if _p.grad is not None:
                            _bb_sq += float(_p.grad.detach().float().pow(2).sum())
                    _bb_post = _bb_sq ** 0.5
                    _bb_pre = _bb_post * (_tn / 25.0) if _tn > 25.0 else _bb_post
                    _hd_pre = max(_tn ** 2 - _bb_pre ** 2, 0.0) ** 0.5
                    with open(_pf_path, 'a', encoding='utf-8') as _pf:
                        if _pf_new:
                            _pf.write('update,grad_norm,clipped,loss,cls,box,quality,'
                                      'depth,num_pos,gate_rate,temp_gate_absmean,'
                                      'bank_slots,temporal_active,bank_resets,'
                                      'grad_backbone,grad_head\n')
                        _pf.write(
                            f"{global_update_step},{_tn:.4f},"
                            f"{int(_tn > 25.0)},"
                            f"{batch_loss.item():.4f},{batch_cls_loss.item():.4f},"
                            f"{batch_box_loss.item():.4f},{batch_quality_loss.item():.4f},"
                            f"{float(batch_depth_loss.item()):.4f},{_np_s},{_gr_s},"
                            f"{_gate_s},{_bank_s},{_ta},{_rst},"
                            f"{_bb_pre:.4f},{_hd_pre:.4f}\n")
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                if new_scale >= old_scale:
                    global_update_step += 1
                    set_optimizer_lr(global_update_step)
                if STOP_AFTER_UPDATES and global_update_step >= STOP_AFTER_UPDATES:
                    _stop_after_requested = True
                    break

            run['loss'] += batch_loss.item()
            run['cls'] += batch_cls_loss.item()
            run['box'] += batch_box_loss.item()
            run['quality'] += batch_quality_loss.item()
            run['depth'] += float(batch_depth_loss.item())
            run['n'] += 1
            _epoch_train_frames += int(images.shape[0])
            _epoch_train_steps += 1

            if step % 10 == 0:
                print(
                    f"  [train] Step {step:03d} | "
                    f"Det Loss: {batch_loss.item():.4f} | "
                    f"Cls: {batch_cls_loss.item():.4f} | "
                    f"Box: {batch_box_loss.item():.4f} | "
                    f"Quality: {batch_quality_loss.item():.4f} | "
                    f"Depth: {batch_depth_loss.item():.4f}"
                )

            # ─── step 기반 검증/체크포인트 (VAL_EVERY_STEPS>0) ───
            # optimizer update 경계에서만, 정확히 N update마다 1회 트리거.
            if (VAL_EVERY_STEPS > 0 and should_step and
                    global_update_step > 0 and
                    global_update_step % VAL_EVERY_STEPS == 0 and run['n'] > 0):
                train_stats = {k: run[k] / run['n'] for k in ('loss', 'cls', 'box', 'quality', 'depth')}
                early_stop = run_validation_and_checkpoint(
                    train_stats, optimizer.param_groups[-1]['lr'],
                    epoch, epoch_completed=False, compute_metric=True,
                )
                # interval 누적 리셋 + 학습 모드 복귀 (validate가 eval로 바꿔놓음)
                run = {k: (0.0 if k != 'n' else 0) for k in run}
                model.train()
                if FREEZE_BACKBONE_BN:
                    model.freeze_backbone_bn()
                if hasattr(model, "reset_temporal_memory"):
                    model.reset_temporal_memory()
                if early_stop:
                    break

        if _stop_after_requested:
            # 진단용 조기 종료: 정상 resume checkpoint + weight 스냅샷 저장 후 종료.
            # scheduler horizon 은 NUM_EPOCHS 그대로였으므로 이 시점까지의 preflight.csv 는
            # 본 학습과 동일한 LR 궤적을 관측한 것이다.
            save_resume_checkpoint(epoch, epoch_completed=False)
            _stop_w = os.path.join(OUTPUT_CKPT_DIR, f"stop_after_{global_update_step}_weights.pth")
            torch.save(model.state_dict(), _stop_w)
            print(f"\n[STOP_AFTER_UPDATES] {global_update_step} update 도달 → 종료. "
                  f"weights: {_stop_w}")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        _train_seconds = time.perf_counter() - _train_started
        append_throughput_record({
            'kind': 'train',
            'epoch': epoch + 1,
            'seconds': _train_seconds,
            'frames': _epoch_train_frames,
            'dataloader_steps': _epoch_train_steps,
            'optimizer_updates_total': global_update_step,
            'frames_per_second': (
                _epoch_train_frames / _train_seconds if _train_seconds > 0 else None
            ),
            'seconds_per_dataloader_step': (
                _train_seconds / _epoch_train_steps if _epoch_train_steps > 0 else None
            ),
            'max_steps_per_epoch': MAX_STEPS_PER_EPOCH,
        })
        print(
            f"[throughput] train epoch={epoch+1} frames={_epoch_train_frames} "
            f"steps={_epoch_train_steps} seconds={_train_seconds:.3f} "
            f"fps={_epoch_train_frames/max(_train_seconds, 1e-12):.3f}"
        )

        if early_stop:
            print(f"\n⚠️  Early Stopping! "
                  f"Primary metric이 {EARLY_STOP_PATIENCE}회 검증 동안 개선 없음.")
            break

        # ─── epoch 끝 검증 ────────────────────────────────
        # step 모드에서 직전 검증 이후 남은 step(run['n']>0)만 마무리 검증한다.
        # (VAL_EVERY_STEPS로 딱 떨어져 run['n']==0이면 중복 검증 생략.)
        if run['n'] > 0:
            train_stats = {k: run[k] / run['n'] for k in ('loss', 'cls', 'box', 'quality', 'depth')}
            # 매 epoch train loss 분해 텍스트 로그 (validation 여부 무관)
            append_epoch_loss_log(EPOCH_LOSS_LOG_PATH, epoch + 1,
                                  train_stats, optimizer.param_groups[-1]['lr'])
            _validation_due = (
                (epoch + 1) % VALIDATE_EVERY_EPOCHS == 0 or
                (epoch + 1) == NUM_EPOCHS
            )
            if _validation_due:
                early_stop = run_validation_and_checkpoint(
                    train_stats, optimizer.param_groups[-1]['lr'],
                    epoch, epoch_completed=True, compute_metric=True,
                )
            else:
                print(
                    f"[validation] epoch {epoch+1} skip "
                    f"(cadence={VALIDATE_EVERY_EPOCHS}); resumable checkpoint only"
                )
                save_resume_checkpoint(epoch, epoch_completed=True)

        # ─── temporal telemetry (epoch 단위) ──────────────
        if USE_TEMPORAL_MEMORY:
            _tg = [float(l.temp_gate.detach().float().mean())
                   for l in model.decoder_layers
                   if getattr(l, 'use_temp_gnn', False) and hasattr(l, 'temp_gate')]
            _bk = getattr(model, '_bank', None)
            print(f"   [temporal] temp_gate={[round(g, 5) for g in _tg]} "
                  f"bank_slots={sum(1 for e in _bk if e is not None) if _bk else 0} "
                  f"bank_resets(누적)={getattr(model, '_bank_reset_count', 0)}")

        # ─── 정기 체크포인트 ──────────────────────────────
        if (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0:
            ckpt_path = os.path.join(OUTPUT_CKPT_DIR, f"checkpoint_epoch{epoch+1}.pth") \
                if RUN_DIR else f"checkpoint_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"   📌 체크포인트 저장: {ckpt_path}")

        # ─── Early stop (primary metric 기준) ─────────────
        if early_stop:
            print(f"\n⚠️  Early Stopping! "
                  f"Primary metric이 {EARLY_STOP_PATIENCE}회 검증 동안 개선 없음.")
            break

    print("\n🎉 학습 완료!")
    torch.save(model.state_dict(), FINAL_WEIGHTS_PATH)
    print(f"💾 최종 모델 저장: {FINAL_WEIGHTS_PATH}")
    print(
        f"📊 Best primary {PRIMARY_BEST_MODE} {PRIMARY_BEST_KEY}@{PRIMARY_BEST_THR:.2f}: "
        
        f"{best_scores['primary']:.4f} | "
        f"Best raw F1@0.25: {best_scores['raw_f1_025']:.4f} | "
        f"Best Val Loss: {best_scores['val_loss']:.4f}"
    )
    wandb.finish()
