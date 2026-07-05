import wandb
import math
import os
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from morai_dataset import IMG_HEIGHT, IMG_WIDTH, MoraiDataset, morai_collate_fn
from resnet_fpn import ResNet50_FPN, Bottleneck
from anchor_generator import NUM_ANCHORS, generate_anchors_full
from decoder import FFNDecoder
from loss_calculator import CustomLoss
from make_kmeans import (
    DEFAULT_FULL_OUT,
    DEFAULT_K,
    DEFAULT_META_OUT,
    DEFAULT_XY_OUT,
    ensure_kmeans_files,
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


def align_anchor_prev_to_current(prev_anchor, prev_ego_pose, cur_ego_pose):
    """
    Convert anchors from previous ego frame into current ego frame.
    ego_pose: [timestamp, ego_x, ego_y, ego_z, ego_yaw_rad, valid]
    Anchor convention: x forward, y left, z up in ego/body frame.
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

    xy_global = (R_prev @ prev_anchor[:, 0:2].unsqueeze(-1)).squeeze(-1) + prev_t
    xy_cur = (R_cur_inv @ (xy_global - cur_t).unsqueeze(-1)).squeeze(-1)
    aligned[:, 0:2] = xy_cur
    aligned[:, 2] = prev_anchor[:, 2] + prev_z - cur_z

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
            instance_feature = instance_feature + self.dropout(temp_out.squeeze(0))

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
    ):
        super().__init__()
        anchors_full = generate_anchors_full()
        if anchors_full.shape != (NUM_ANCHORS, 11):
            raise ValueError(f"anchors_full shape 이상: {anchors_full.shape}")

        self.num_classes = num_classes
        self.use_temporal_memory = use_temporal_memory
        self.num_temp_instances = min(int(num_temp_instances), NUM_ANCHORS)
        self.temporal_confidence_decay = temporal_confidence_decay
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
        self._temporal_context = None
        self._cached_feature = None
        self._cached_anchor = None
        self._cached_confidence = None
        self._cached_ego_pose = None

    @staticmethod
    def _context_from_stem(stem):
        if stem is None:
            return None
        return str(stem).split('/')[0]

    @torch.no_grad()
    def _get_temporal_memory(self, context, cur_ego_pose, device, dtype):
        if (
            not self.use_temporal_memory or
            context is None or
            self._temporal_context != context or
            self._cached_feature is None or
            self._cached_anchor is None or
            self._cached_ego_pose is None
        ):
            if context is not None and self._temporal_context != context:
                self.reset_temporal_memory()
            return None, None

        if cur_ego_pose is None or cur_ego_pose.numel() < 6 or cur_ego_pose[-1] <= 0.5:
            return None, None

        temp_feature = self._cached_feature.to(device=device, dtype=dtype)
        prev_anchor = self._cached_anchor.to(device=device, dtype=dtype)
        prev_ego_pose = self._cached_ego_pose.to(device=device, dtype=dtype)
        aligned_anchor = align_anchor_prev_to_current(
            prev_anchor,
            prev_ego_pose,
            cur_ego_pose.to(device=device, dtype=dtype),
        )
        if aligned_anchor is None:
            return None, None
        return temp_feature, aligned_anchor

    @torch.no_grad()
    def _update_temporal_memory(
        self,
        instance_feature,
        anchor,
        det_cls,
        det_quality,
        context,
        cur_ego_pose,
    ):
        if (
            not self.use_temporal_memory or
            context is None or
            self.num_temp_instances <= 0 or
            cur_ego_pose is None or
            cur_ego_pose.numel() < 6 or
            cur_ego_pose[-1] <= 0.5
        ):
            return

        probs = det_cls.detach().float().sigmoid()
        confidence = probs.max(dim=-1).values
        if det_quality is not None:
            centerness = quality_centerness(det_quality.detach().float()).view(-1)
            confidence = confidence * torch.sigmoid(centerness)

        topk = min(self.num_temp_instances, confidence.shape[0])
        _, indices = torch.topk(confidence, k=topk, largest=True)
        self._temporal_context = context
        self._cached_feature = instance_feature.detach()[indices].cpu()
        self._cached_anchor = anchor.detach()[indices].cpu()
        self._cached_confidence = confidence.detach()[indices].cpu()
        self._cached_ego_pose = cur_ego_pose.detach().cpu()

    def forward(
        self,
        images,
        intrinsics,
        extrinsics,
        stems=None,
        ego_poses=None,
        return_intermediate=False,
    ):
        """
        images     : [B, 3, 3, H, W]
        intrinsics : [B, 3, 3, 3] resized-input coordinate system
        extrinsics : [B, 3, 4, 4]
        반환       : dict with final and per-decoder outputs
        """
        B = images.shape[0]
        image_h = images.shape[-2]
        image_w = images.shape[-1]

        # 배치별 출력 누적
        batch_det_classes = []
        batch_det_boxes   = []
        batch_det_quality = []
        batch_all_classes = []
        batch_all_boxes = []
        batch_all_quality = []

        for b in range(B):
            context = self._context_from_stem(stems[b]) if stems is not None else None
            cam_imgs = images[b]                            # [3, 3, H, W]
            all_features = self.backbone(cam_imgs)          # list of [3, C, H, W]

            cur_ego_pose = ego_poses[b] if ego_poses is not None else None
            det_feat = self.instance_feature
            det_box = self.det_anchors_full
            temp_feat, temp_box = self._get_temporal_memory(
                context,
                cur_ego_pose,
                det_feat.device,
                det_feat.dtype,
            )
            det_cls = None
            det_quality = None
            layer_classes = []
            layer_boxes = []
            layer_quality = []
            for layer in self.decoder_layers:
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

            batch_det_classes.append(det_cls)
            batch_det_boxes.append(det_box)
            batch_det_quality.append(det_quality)
            batch_all_classes.append(torch.stack(layer_classes))
            batch_all_boxes.append(torch.stack(layer_boxes))
            batch_all_quality.append(torch.stack(layer_quality))
            self._update_temporal_memory(
                det_feat,
                det_box,
                det_cls,
                det_quality,
                context,
                cur_ego_pose,
            )

        output = {
            'det_cls': torch.stack(batch_det_classes),
            'det_box': torch.stack(batch_det_boxes),
            'det_quality': torch.stack(batch_det_quality),
            'all_det_cls': torch.stack(batch_all_classes),
            'all_det_box': torch.stack(batch_all_boxes),
            'all_det_quality': torch.stack(batch_all_quality),
        }
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

    for b in range(B):
        gt_boxes = batch['dynamic_gt_boxes'][b].to(device)
        gt_classes = batch['dynamic_gt_labels'][b].to(device)
        for layer_idx in range(L):
            weight = layer_weights[layer_idx]
            det_loss, cls_loss, box_loss, quality_loss = criterion(
                all_cls[b, layer_idx],
                all_box[b, layer_idx],
                gt_classes,
                gt_boxes,
                all_quality[b, layer_idx],
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


def count_ego_pose_files(dataset):
    total = len(dataset.items)
    count = 0
    for scen_dir, stem in dataset.items:
        pose_path = os.path.join(scen_dir, 'ego_pose', f"{stem}.csv")
        if os.path.isfile(pose_path):
            count += 1
    return count, total


@torch.no_grad()
def validate(model, loader, criterion, device, compute_metric=False, recall_thr=2.0):
    """
    val set 전체에 대해 평균 detection loss를 계산하고,
    compute_metric=True면 score/class-aware Precision/Recall도 계산.
    반환: (val_loss, metrics_or_None)
    """
    model.eval()
    if hasattr(model, "reset_temporal_memory"):
        model.reset_temporal_memory()
    loss_sum = 0.0
    n_batches = 0
    main_score_threshold = 0.05
    metric_score_thresholds = (main_score_threshold, 0.10, 0.15, 0.25)
    metric_modes = {
        'calibrated': (True, 1.0),      # foreground * quality
        'softcalibrated': (True, 0.5),  # foreground * sqrt(quality)
        'raw': (False, 1.0),            # foreground only
    }
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
    score_count = 0

    # v10: softcalibrated@0.15 ego 방사거리 3구간 P/R/F1 + 매칭쌍 평균 center distance (로깅 전용)
    distance_bucket_counts = {
        i: {'tp': 0, 'fp': 0, 'fn': 0, 'dist_sum': 0.0, 'dist_n': 0}
        for i in range(len(DISTANCE_BUCKETS))
    }

    for batch in loader:
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
            return_intermediate=True,
        )
        det_classes_b = model_out['det_cls']
        det_boxes_b = model_out['det_box']
        det_quality_b = model_out['det_quality']
        batch_loss, _, _, _ = compute_auxiliary_detection_loss(
            model_out, batch, criterion, device
        )

        for i in range(n):
            gt_boxes = batch['dynamic_gt_boxes'][i].to(device)
            gt_classes = batch['dynamic_gt_labels'][i].to(device)

            if compute_metric:
                fg_probs = det_classes_b[i].sigmoid()
                raw_scores = fg_probs.max(dim=-1).values
                quality = torch.sigmoid(
                    quality_centerness(det_quality_b[i]).view(-1)
                ).clamp(min=1e-6)
                soft_scores = raw_scores * quality.sqrt()
                score_sums['quality'] += float(quality.sum().item())
                score_sums['raw'] += float(raw_scores.sum().item())
                score_sums['soft'] += float(soft_scores.sum().item())
                score_count += int(raw_scores.numel())

                for mode, (apply_quality, quality_power) in metric_modes.items():
                    for score_thr in metric_score_thresholds:
                        # 클래스별 카운트 (합산은 클래스별 합과 동일 → 기존 합산 로그 값 보존)
                        per_cls = compute_detection_counts_by_class(
                            det_classes_b[i],
                            det_boxes_b[i],
                            gt_classes,
                            gt_boxes,
                            det_quality=det_quality_b[i],
                            distance_thr=recall_thr,
                            score_thresh=score_thr,
                            apply_quality=apply_quality,
                            quality_power=quality_power,
                            class_ids=class_ids,
                        )
                        tp = fp = fn = 0
                        for cls in class_ids:
                            ctp, cfp, cfn = per_cls[cls]
                            metric_counts_cls[mode][score_thr][cls]['tp'] += ctp
                            metric_counts_cls[mode][score_thr][cls]['fp'] += cfp
                            metric_counts_cls[mode][score_thr][cls]['fn'] += cfn
                            tp += ctp
                            fp += cfp
                            fn += cfn
                        metric_counts[mode][score_thr]['tp'] += tp
                        metric_counts[mode][score_thr]['fp'] += fp
                        metric_counts[mode][score_thr]['fn'] += fn

                # v10: softcalibrated@0.15 거리구간별 집계 (기존 루프와 독립)
                bucket_result = compute_detection_counts_by_distance(
                    det_classes_b[i],
                    det_boxes_b[i],
                    gt_classes,
                    gt_boxes,
                    det_quality=det_quality_b[i],
                    distance_thr=recall_thr,
                    score_thresh=0.15,
                    apply_quality=True,
                    quality_power=0.5,
                )
                for b, counts in bucket_result.items():
                    for k in ('tp', 'fp', 'fn', 'dist_sum', 'dist_n'):
                        distance_bucket_counts[b][k] += counts[k]

        loss_sum += batch_loss.item()
        n_batches += 1

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


HISTORY_SCORE_THRESHOLDS = (0.05, 0.10, 0.15, 0.25)
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
    'val_loss',
    'lr',
    'val_quality_mean',
    'val_raw_score_mean',
    'val_soft_score_mean',
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
                        train_quality_loss, val_loss, lr, val_metrics):
    score_stats = (val_metrics or {}).get('score_stats', {})
    record = {
        'epoch': int(epoch),
        'train_loss': float(train_loss),
        'train_cls_loss': float(train_cls_loss),
        'train_box_loss': float(train_box_loss),
        'train_quality_loss': float(train_quality_loss),
        'val_loss': float(val_loss),
        'lr': float(lr),
        'val_quality_mean': float(score_stats.get('quality', float('nan'))),
        'val_raw_score_mean': float(score_stats.get('raw', float('nan'))),
        'val_soft_score_mean': float(score_stats.get('soft', float('nan'))),
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


def save_training_history(records, csv_path, plot_path):
    if not records:
        return

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, float('nan')) for field in HISTORY_FIELDS})

    epochs = [int(record['epoch']) for record in records]
    train_losses = [float(record['train_loss']) for record in records]
    val_losses = [float(record['val_loss']) for record in records]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, train_losses, marker='o', linewidth=1.8, label='Train loss')
    ax.plot(epochs, val_losses, marker='o', linewidth=2.2, label='Val loss')

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

    ax.set_title('Training / Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def metric_value(val_metrics, mode, score_thr, key):
    if val_metrics is None:
        return float('-inf')
    return float(val_metrics['by_mode'][mode][score_thr][key])


def save_best_with_epoch(model, fixed_path, epoch, tag, score):
    torch.save(model.state_dict(), fixed_path)
    root, ext = os.path.splitext(fixed_path)
    out_dir = os.path.dirname(root) or "."
    root_name = os.path.basename(root)
    for name in os.listdir(out_dir):
        if name.startswith(f"{root_name}_epoch") and name.endswith(ext):
            os.remove(os.path.join(out_dir, name))
    epoch_path = f"{root}_epoch{epoch:03d}_{tag}_{score:.4f}{ext}"
    torch.save(model.state_dict(), epoch_path)
    return epoch_path


# ===========================================================
# 학습 루프 (3전방 카메라 동적 객체 detection 전용)
# ===========================================================
if __name__ == "__main__":
    # ─── Config ────────────────────────────────────────
    DATASET_ROOT = './dataset'
    # val로 뺄 시나리오 이름. None이면 알파벳 정렬 마지막 5개를 자동 사용.
    # ⚠️ make_kmeans.py 의 --val-scenarios와 반드시 일치시킬 것.
    VAL_SCENARIOS = None

    NUM_EPOCHS           = 100
    BATCH_SIZE           = 4
    GRAD_ACCUM_STEPS     = 2      # effective batch size = 8
    EARLY_STOP_PATIENCE  = 10     # small MORAI split에서는 best 이후 drift가 빨라 20은 너무 오래 끈다.
    EARLY_STOP_MIN_DELTA = 1e-4
    METRIC_EVERY         = 1      # best 기준이 recall이므로 매 epoch P/R 계산
    RECALL_THR           = 2.0    # distance match threshold
    KMEANS_K             = DEFAULT_K
    FORCE_REMAKE_KMEANS  = True
    AUX_LOSS_WEIGHT      = 0.5
    USE_TEMPORAL_MEMORY  = False
    NUM_TEMP_INSTANCES   = 600
    PRIMARY_BEST_MODE    = "softcalibrated"
    PRIMARY_BEST_THR     = 0.15
    PRIMARY_BEST_KEY     = "f1"
    BEST_METRIC_MODE     = "raw"  # legacy/resume 호환용
    FREEZE_BACKBONE_BN   = True   # batch=sample별 3 cameras라 backbone BN은 고정
    USE_AMP              = False  # custom aggregation/quality 학습 안정성을 위해 FP32 사용
    BEST_MODEL_PATH      = "best_model.pth"
    BEST_RAW_F1_025_PATH = "best_model_raw_f1_025.pth"
    BEST_VAL_LOSS_PATH   = "best_model_val_loss.pth"
    LAST_CHECKPOINT_PATH = "last_checkpoint.pth"
    FINAL_WEIGHTS_PATH   = "morai_autonav_weights.pth"
    RESUME_FROM          = (
        LAST_CHECKPOINT_PATH if os.path.isfile(LAST_CHECKPOINT_PATH) else BEST_MODEL_PATH
    )  # None이면 처음부터 학습
    HISTORY_CSV_PATH     = "training_history.csv"
    HISTORY_PLOT_PATH    = "training_curves.png"
    WARMUP_STEPS         = 500
    MIN_LR_RATIO         = 1e-3
    if RESUME_FROM is not None and os.path.isfile(RESUME_FROM):
        FORCE_REMAKE_KMEANS = False
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
    print(f"   - metric      : Precision/Recall@{RECALL_THR}m, 매 {METRIC_EVERY} epoch")
    print(f"   - input size  : {IMG_WIDTH}x{IMG_HEIGHT}")
    print(f"   - decoder     : SparseDrive-style 6 refinement layers + auxiliary loss")
    print(f"   - aggregation : CUDA deformable op + learned camera embedding, grid_sample fallback")
    print(f"   - quality     : centerness/yawness head, score = foreground * centerness")
    print(f"   - best metric : primary={PRIMARY_BEST_MODE}/{PRIMARY_BEST_KEY}@{PRIMARY_BEST_THR:.2f}")
    print(f"   - temporal    : instance bank={USE_TEMPORAL_MEMORY}, temp={NUM_TEMP_INSTANCES}")
    print(f"   - backbone BN : freeze={FREEZE_BACKBONE_BN}")
    print(f"   - AMP         : {USE_AMP}")
    print(f"   - batch       : {BATCH_SIZE} x accum {GRAD_ACCUM_STEPS} = {BATCH_SIZE * GRAD_ACCUM_STEPS}\n")
    print(f"   - graph       : {HISTORY_PLOT_PATH} (csv: {HISTORY_CSV_PATH})\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[디바이스] {device}\n")

    ensure_kmeans_files(
        dataset_root=DATASET_ROOT,
        val_scenarios=VAL_SCENARIOS,
        k=KMEANS_K,
        xy_out=DEFAULT_XY_OUT,
        full_out=DEFAULT_FULL_OUT,
        meta_out=DEFAULT_META_OUT,
        force=FORCE_REMAKE_KMEANS,
    )

    model = AutoNavModel(
        num_decoder_layers=6,
        pretrained_backbone=True,
        use_temporal_memory=USE_TEMPORAL_MEMORY,
        num_temp_instances=NUM_TEMP_INSTANCES,
    ).to(device)
    if FREEZE_BACKBONE_BN:
        model.freeze_backbone_bn()

    train_ds = MoraiDataset(dataset_root=DATASET_ROOT, split='train', val_scenarios=VAL_SCENARIOS)
    val_ds   = MoraiDataset(dataset_root=DATASET_ROOT, split='val',   val_scenarios=VAL_SCENARIOS)

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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=not USE_TEMPORAL_MEMORY,
                              collate_fn=morai_collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=morai_collate_fn, num_workers=0)

    # num_classes=2: vehicle, pedestrian (sigmoid focal, 배경 채널 없음)
    # v9: quality_weight=0.2 명시 — centerness/confidence 학습 강화
    det_criterion = CustomLoss(num_classes=2, quality_weight=0.2).to(device)

    backbone_params = list(model.backbone.parameters())
    backbone_ids    = set(id(p) for p in backbone_params)
    other_params    = [p for p in model.parameters() if id(p) not in backbone_ids]

    wandb.init(
        project="morai-3d-detection",
        name="v10",
        tags=["v10"],
        config={
            "lr_backbone": 2e-5,
            "lr_head": 1e-4,
            "wd_backbone": 1e-3,
            "wd_head": 1e-2,
            "batch_size": 4,
            "grad_accum": 2,
        }
    )


    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5,  'weight_decay': 1e-3},
        {'params': other_params,    'lr': 5e-5,  'weight_decay': 1e-2},
    ])

    updates_per_epoch = max(math.ceil(len(train_loader) / GRAD_ACCUM_STEPS), 1)
    total_update_steps = max(updates_per_epoch * NUM_EPOCHS, 1)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return max(float(step + 1) / float(max(WARMUP_STEPS, 1)), MIN_LR_RATIO)
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
            if 'optimizer_state' in ckpt:
                pass  # optimizer state 스킵
            if 'scaler_state' in ckpt and ckpt['scaler_state'] is not None:
                scaler.load_state_dict(ckpt['scaler_state'])
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            global_update_step = int(ckpt.get('global_update_step', global_update_step))
            best_recall = float(ckpt.get('best_recall', best_recall))
            best_val_loss = float(ckpt.get('best_val_loss', best_val_loss))
            best_scores.update(ckpt.get('best_scores', {}))
            best_scores['val_loss'] = min(best_scores['val_loss'], best_val_loss)
            epochs_no_improve = int(ckpt.get('epochs_no_improve', epochs_no_improve))
            epochs_no_improve = 0  # v9 새 실험 시작 — 체크포인트 로드 직후 카운터 리셋
            history_records = ckpt.get('history_records', [])
            if not history_records:
                history_records = load_training_history(HISTORY_CSV_PATH, max_epoch=start_epoch)
            set_optimizer_lr(global_update_step)
            print(
                f"[학습] full checkpoint에서 재개: {RESUME_FROM} | "
                f"start_epoch={start_epoch + 1}, update_step={global_update_step}\n"
            )
        else:
            model.load_state_dict(ckpt)
            print(
                f"[학습] 모델 가중치에서 이어 학습: {RESUME_FROM} "
                f"(optimizer/LR/epoch은 새로 시작)\n"
            )
    else:
        if RESUME_FROM is not None:
            print(f"[학습] resume 파일 없음: {RESUME_FROM}")
        print("[학습] 처음부터 학습\n")

    for epoch in range(start_epoch, NUM_EPOCHS):
        # ─── Train ───────────────────────────────────────
        model.train()
        if FREEZE_BACKBONE_BN:
            model.freeze_backbone_bn()
        if hasattr(model, "reset_temporal_memory"):
            model.reset_temporal_memory()
        print(f"\n========== [Epoch {epoch+1}/{NUM_EPOCHS}] ==========")
        train_loss_sum = 0.0
        train_cls_loss_sum = 0.0
        train_box_loss_sum = 0.0
        train_quality_loss_sum = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            images     = batch['images'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            extrinsics = batch['extrinsics'].to(device)
            ego_poses  = batch['ego_pose'].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and USE_AMP)):
                model_out = model(
                    images,
                    intrinsics,
                    extrinsics,
                    stems=batch['stem'],
                    ego_poses=ego_poses,
                    return_intermediate=True,
                )
                batch_loss, batch_cls_loss, batch_box_loss, batch_quality_loss = (
                    compute_auxiliary_detection_loss(
                        model_out,
                        batch,
                        det_criterion,
                        device,
                        aux_weight=AUX_LOSS_WEIGHT,
                    )
                )

            loss_for_backward = batch_loss / GRAD_ACCUM_STEPS
            scaler.scale(loss_for_backward).backward()

            should_step = (
                ((step + 1) % GRAD_ACCUM_STEPS == 0) or
                (step + 1 == len(train_loader))
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 25.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()
                optimizer.zero_grad(set_to_none=True)
                if new_scale >= old_scale:
                    global_update_step += 1
                    set_optimizer_lr(global_update_step)

            train_loss_sum += batch_loss.item()
            train_cls_loss_sum += batch_cls_loss.item()
            train_box_loss_sum += batch_box_loss.item()
            train_quality_loss_sum += batch_quality_loss.item()

            if step % 10 == 0:
                print(
                    f"  [train] Step {step:03d} | "
                    f"Det Loss: {batch_loss.item():.4f} | "
                    f"Cls: {batch_cls_loss.item():.4f} | "
                    f"Box: {batch_box_loss.item():.4f} | "
                    f"Quality: {batch_quality_loss.item():.4f}"
                )

        train_loss = train_loss_sum / len(train_loader)
        train_cls_loss = train_cls_loss_sum / len(train_loader)
        train_box_loss = train_box_loss_sum / len(train_loader)
        train_quality_loss = train_quality_loss_sum / len(train_loader)

        # ─── Val ─────────────────────────────────────────
        compute_metric = ((epoch + 1) % METRIC_EVERY == 0)
        val_loss, val_metrics = validate(
            model, val_loader, det_criterion, device,
            compute_metric=compute_metric, recall_thr=RECALL_THR,
        )

        lr = optimizer.param_groups[-1]['lr']
        msg = (f"\n📊 Epoch {epoch+1} | "
               f"Train: {train_loss:.4f} "
               f"(Cls {train_cls_loss:.4f}, Box {train_box_loss:.4f}, Q {train_quality_loss:.4f}) | "
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
            # v9: primary mode 기준 클래스별(vehicle/pedestrian) 분리 수치
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
            # v10: softcalibrated@0.15 ego 방사거리 3구간 P/R/F1 + 매칭쌍 평균 center distance
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
            epoch + 1,
            train_loss,
            train_cls_loss,
            train_box_loss,
            train_quality_loss,
            val_loss,
            lr,
            val_metrics,
        ))
        save_training_history(history_records, HISTORY_CSV_PATH, HISTORY_PLOT_PATH)
        print(f"   📈 Loss 그래프 갱신: {HISTORY_PLOT_PATH} | 로그: {HISTORY_CSV_PATH}")

        wandb_log = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/cls_loss": train_cls_loss,
            "train/box_loss": train_box_loss,
            "train/quality_loss": train_quality_loss,
            "val/loss": val_loss,
            "lr": lr,
        }
        if val_metrics is not None:
            for mode in HISTORY_METRIC_MODES:
                for score_thr in HISTORY_SCORE_THRESHOLDS:
                    metric = val_metrics['by_mode'][mode][score_thr]
                    wandb_log[f"val/{mode}/precision@{score_thr:.2f}"] = metric['precision']
                    wandb_log[f"val/{mode}/recall@{score_thr:.2f}"] = metric['recall']
                    wandb_log[f"val/{mode}/f1@{score_thr:.2f}"] = metric['f1']
            # v9: 클래스별 P/R/F1 (예: val/softcalibrated/vehicle/f1@0.15)
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
            # v10: softcalibrated@0.15 거리구간별 P/R/F1 + 평균 matched center distance
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
            val_metrics,
            PRIMARY_BEST_MODE,
            PRIMARY_BEST_THR,
            PRIMARY_BEST_KEY,
        )
        primary_improved = (
            primary_score > best_scores['primary'] + EARLY_STOP_MIN_DELTA or
            (
                abs(primary_score - best_scores['primary']) <= EARLY_STOP_MIN_DELTA and
                val_loss < best_val_loss
            )
        )
        if primary_improved:
            best_scores['primary'] = primary_score
            best_val_loss = min(best_val_loss, val_loss)
            epochs_no_improve = 0
            epoch_best_path = save_best_with_epoch(
                model,
                BEST_MODEL_PATH,
                epoch + 1,
                f"{PRIMARY_BEST_MODE}_{PRIMARY_BEST_KEY}_{_score_suffix(PRIMARY_BEST_THR)}",
                primary_score,
            )
            print(
                f"   💾 Primary best 저장: {BEST_MODEL_PATH} | "
                f"{PRIMARY_BEST_MODE} {PRIMARY_BEST_KEY}@{PRIMARY_BEST_THR:.2f}="
                f"{primary_score:.4f} | Val Loss: {val_loss:.4f}"
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
                model,
                BEST_RAW_F1_025_PATH,
                epoch + 1,
                "raw_f1_025",
                raw_f1_025,
            )
            print(
                f"   💾 Raw F1@0.25 best 저장: {raw_f1_025:.4f} -> "
                f"{BEST_RAW_F1_025_PATH} ({epoch_best_path})"
            )

        if val_loss < best_scores['val_loss'] - EARLY_STOP_MIN_DELTA:
            best_scores['val_loss'] = val_loss
            best_val_loss = val_loss
            epoch_best_path = save_best_with_epoch(
                model,
                BEST_VAL_LOSS_PATH,
                epoch + 1,
                "val_loss",
                val_loss,
            )
            print(
                f"   💾 Val loss best 저장: {val_loss:.4f} -> "
                f"{BEST_VAL_LOSS_PATH} ({epoch_best_path})"
            )

        # ─── Resume용 full checkpoint ─────────────────────
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scaler_state': scaler.state_dict() if scaler is not None else None,
            'global_update_step': global_update_step,
            'best_recall': best_recall,
            'best_val_loss': best_val_loss,
            'best_scores': best_scores,
            'epochs_no_improve': epochs_no_improve,
            'history_records': history_records,
        }, LAST_CHECKPOINT_PATH)
        print(f"   💽 Resume 체크포인트 저장: {LAST_CHECKPOINT_PATH}")

        # ─── 정기 체크포인트 ──────────────────────────────
        if (epoch + 1) % 10 == 0:
            ckpt_path = f"checkpoint_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"   📌 체크포인트 저장: {ckpt_path}")

        # ─── Early stop (primary metric 기준) ─────────────
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early Stopping! "
                  f"Primary metric이 {EARLY_STOP_PATIENCE} epoch 동안 개선 없음.")
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
