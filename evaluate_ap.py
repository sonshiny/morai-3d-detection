#!/usr/bin/env python3
"""Phase 1 baseline harness: nuScenes-style front-ROI AP + 학습 진단 계측.

학습 코드는 변경하지 않는다. train.py / morai_dataset.py 를 import 해 기존 run 의
checkpoint 를 공식 SparseDrive decode 프로토콜로 재평가하고, 현재 학습 계약의
진단 지표(positive confidence, matching cost 분해, counterfactual yaw, quality
정보량)를 함께 기록한다.

평가 계약 (2026-07 합의 문서):
  decode  : query별 class argmax → raw cls 기준 top-300 membership (threshold 없음)
            → final = raw × sigmoid(centerness) 로 내부 re-ranking
            → 변형 3종 병행 보고:
                final_nonms (공식 parity, primary)
                raw_nonms   (quality reranking 효과 분리)
                final_nms   (배포용: raw>=0.05 prefilter 후 rotated NMS)
  ROI     : 3카메라 union-FOV 투영 유효 ∩ radial 50m (GT 제거 정책과 동일 계약).
            ROI 밖 prediction 은 FP 로 세지 않고 제거, 개수만 기록.
  AP      : nuScenes 규약 (101-pt 보간, min_recall=0.1, min_precision=0.1),
            center-distance {0.5,1,2,4}m × {vehicle, pedestrian}.
  TP err  : 2m 매칭(final_nonms) TP 의 ATE/ATE3D/ASE/AOE/AVE.
            nuScenes 의 recall-level 평균 대신 TP 단순 평균 (문서화된 간소화).
  진단    : 현재 학습 matcher 계약(cost_class=1·(-sigmoid), cost_bbox=5·L1/BOX_SCALE)
            그대로 사용 — 현재 학습 상태의 특성 측정이 목적이므로 의도적.

사용 예:
  DATASET_ROOT=/workspace/dataset python3 evaluate_ap.py \
      --run-dir runs/prod_v3_depth_20260726_1332 \
      --ckpt checkpoint_epoch20.pth --tag epoch20 [--max-frames 200]
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np


def _setup_env(run_cfg):
    """train.py/anchor_generator 가 env 를 읽기 전에 run 과 동일한 계약을 주입."""
    anchor_dir = run_cfg["anchor_dir"]
    os.environ.setdefault("ANCHOR_XY_FILE", os.path.join(anchor_dir, "anchor_kmeans_xy.npy"))
    os.environ.setdefault("ANCHOR_FULL_FILE", os.path.join(anchor_dir, "anchor_kmeans_full.npy"))
    # 원 run 과 동일: FOV 재필터 off(수집단계 적용), GT 제거 규칙 기본 on.
    os.environ.setdefault("FILTER_VISIBLE", "0")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True)
    p.add_argument("--ckpt", required=True, help="run-dir 내 checkpoint 파일명 또는 절대경로")
    p.add_argument("--tag", required=True, help="결과 파일 이름 (예: epoch20)")
    p.add_argument("--dataset-root", default=os.environ.get("DATASET_ROOT", "/workspace/dataset"))
    p.add_argument("--out-dir", default=None, help="기본: <run-dir>/ap_eval")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-frames", type=int, default=0, help="smoke test 용 프레임 제한 (0=전체)")
    p.add_argument("--no-diag", action="store_true", help="진단 계측 생략 (AP 만)")
    p.add_argument("--roi-max-dist", type=float, default=50.0)
    p.add_argument("--nms-prefilter", type=float, default=0.05,
                   help="final_nms 변형의 raw score prefilter (배포 threshold)")
    return p.parse_args()


ARGS = parse_args()

with open(os.path.join(ARGS.run_dir, "run_config.json"), encoding="utf-8") as f:
    RUN_CFG = json.load(f)
_setup_env(RUN_CFG)

import torch  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from morai_dataset import (  # noqa: E402
    IMG_HEIGHT,
    IMG_WIDTH,
    MoraiTemporalDataset,
    StreamingGroupSampler,
    morai_temporal_collate_fn,
)
from train import AutoNavModel, CLASS_ID_NAMES, bev_nms_rotated  # noqa: E402
from loss_calculator import BOX_SCALE  # noqa: E402

DIST_THRESHOLDS = (0.5, 1.0, 2.0, 4.0)
TP_ERR_DIST = 2.0
MIN_RECALL = 0.1
MIN_PRECISION = 0.1
BUCKETS = ((0.0, 20.0), (20.0, 40.0), (40.0, 50.0))
#   final_pre_nonms: raw>=prefilter 만 적용, NMS 없음 — final_nms 와의 차이가 순수
#   NMS 효과, final_nonms 와의 차이가 순수 prefilter 효과 (confound 분해용)
VARIANTS = ("final_nonms", "raw_nonms", "final_pre_nonms", "final_nms")
# 현재 matcher 계약 상수 (진단용 재현 — loss_calculator.HungarianMatcher 와 동일)
DIAG_COST_CLASS = 1.0
DIAG_COST_BBOX = 5.0


# ───────────────────────── decode (공식 parity) ─────────────────────────

def roi_mask_np(centers, intr, extr, max_dist):
    """centers [N,3] ego → 3카메라 union-FOV ∩ radial max_dist.
    투영 규약은 DeformableAggregation._build_projection_mats 와 동일:
      u = fx·(−Yc)/Xc + cx,  v = fy·(−Zc)/Xc + cy,  depth = Xc
    """
    n = centers.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    in_dist = np.linalg.norm(centers[:, :2], axis=1) <= max_dist
    homo = np.concatenate([centers, np.ones((n, 1), dtype=centers.dtype)], axis=1)  # [N,4]
    visible = np.zeros(n, dtype=bool)
    for c in range(intr.shape[0]):
        cam = (extr[c] @ homo.T).T  # [N,4] ego→cam
        x, y, z = cam[:, 0], cam[:, 1], cam[:, 2]
        valid = x > 0.1
        with np.errstate(divide="ignore", invalid="ignore"):
            u = intr[c, 0, 0] * (-y) / x + intr[c, 0, 2]
            v = intr[c, 1, 1] * (-z) / x + intr[c, 1, 2]
        visible |= valid & (u >= 0) & (u < IMG_WIDTH) & (v >= 0) & (v < IMG_HEIGHT)
    return in_dist & visible


def decode_official(det_cls, det_box, det_q, intr, extr, roi_max, nms_prefilter):
    """공식 SparseBox3DDecoder parity (det 경로: squeeze_cls=True 상당).

    반환: dict(boxes[K,11], labels[K], raw[K], final[K], keep masks per variant),
          roi_dropped(int)
    """
    probs = torch.sigmoid(det_cls)                      # [900,2]
    raw_all, labels_all = probs.max(dim=-1)             # [900]
    k = min(300, raw_all.shape[0])
    top = torch.topk(raw_all, k=k, largest=True).indices
    boxes = det_box[top].numpy()
    labels = labels_all[top].numpy()
    raw = raw_all[top].numpy()
    cns = torch.sigmoid(det_q[top, 0]).numpy()
    final = raw * cns

    roi = roi_mask_np(boxes[:, :3].astype(np.float64), intr, extr, roi_max)
    roi_dropped = int((~roi).sum())
    boxes, labels, raw, final, cns = boxes[roi], labels[roi], raw[roi], final[roi], cns[roi]

    out = {"boxes": boxes, "labels": labels, "raw": raw, "final": final, "cns": cns,
           "roi_dropped": roi_dropped}

    # 배포용: raw prefilter 후 final score 로 rotated NMS
    pre = raw >= nms_prefilter
    if pre.sum() > 0:
        b_t = torch.from_numpy(boxes[pre])
        s_t = torch.from_numpy(final[pre])
        l_t = torch.from_numpy(labels[pre])
        keep_local = bev_nms_rotated(b_t, s_t, l_t, iou_thresh=0.3).numpy()
        nms_idx = np.flatnonzero(pre)[keep_local]
    else:
        nms_idx = np.zeros(0, dtype=np.int64)
    out["nms_idx"] = nms_idx
    return out


# ───────────────────────── AP (nuScenes 규약) ─────────────────────────

def greedy_pr(scores, frames, xys, gt_by_frame, dist_th):
    """score 내림차순 greedy center-distance 매칭 → (tp flags, npos).
    gt_by_frame: frame → [M,2] xy 배열 (class 별 사전 분리).
    """
    order = np.argsort(-scores, kind="stable")
    tp = np.zeros(len(scores), dtype=bool)
    used = {}
    for i in order:
        f = frames[i]
        g = gt_by_frame.get(f)
        if g is None or g.shape[0] == 0:
            continue
        u = used.setdefault(f, np.zeros(g.shape[0], dtype=bool))
        d = np.linalg.norm(g - xys[i], axis=1)
        d[u] = np.inf
        j = int(np.argmin(d))
        if d[j] <= dist_th:
            u[j] = True
            tp[i] = True
    return tp[order], order


def dup_fp_stats(scores, frames, xys, gt_by_frame, dist_th=2.0):
    """2m greedy 매칭 후 FP 를 분해: dup(근처 GT 있으나 이미 매칭) vs iso(반경 내 GT 없음).
    추가로 GT 1개당 반경 dist_th 안의 예측 수 평균을 반환 (중복 예측 밀도)."""
    order = np.argsort(-scores, kind="stable")
    used = {}
    tp = dup_fp = iso_fp = 0
    for i in order:
        f = frames[i]
        g = gt_by_frame.get(f)
        if g is None or g.shape[0] == 0:
            iso_fp += 1
            continue
        dd = np.linalg.norm(g - xys[i], axis=1)
        u = used.setdefault(f, np.zeros(g.shape[0], dtype=bool))
        d2 = dd.copy()
        d2[u] = np.inf
        j = int(np.argmin(d2))
        if d2[j] <= dist_th:
            u[j] = True
            tp += 1
        elif dd.min() <= dist_th:
            dup_fp += 1
        else:
            iso_fp += 1
    cnt = n_gt = 0
    for f, g in gt_by_frame.items():
        m = frames == f
        n_gt += g.shape[0]
        if m.any():
            dmat = np.linalg.norm(g[:, None] - xys[m][None], axis=-1)
            cnt += int((dmat <= dist_th).sum())
    fp = dup_fp + iso_fp
    return {"tp": tp, "dup_fp": dup_fp, "iso_fp": iso_fp,
            "dup_fp_frac_of_fp": dup_fp / max(fp, 1),
            "preds_per_gt_2m": cnt / max(n_gt, 1)}


def nusc_ap(tp_sorted, npos):
    if npos == 0:
        return float("nan"), None, None
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(~tp_sorted)
    rec = tp_cum / npos
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    rec_interp = np.linspace(0, 1, 101)
    prec_i = np.interp(rec_interp, rec, prec, right=0) if len(rec) else np.zeros(101)
    prec_i = prec_i[round(100 * MIN_RECALL) + 1:]
    prec_i = np.clip(prec_i - MIN_PRECISION, 0, None)
    ap = float(prec_i.mean() / (1.0 - MIN_PRECISION))
    return ap, rec, prec


def aligned_size_error(box_p, box_g):
    """ASE: 중심·yaw 정렬 후 1 − IoU (dims 는 exp(ln))."""
    dp = np.exp(box_p[3:6])
    dg = np.exp(box_g[3:6])
    inter = np.prod(np.minimum(dp, dg))
    union = np.prod(dp) + np.prod(dg) - inter
    return float(1.0 - inter / max(union, 1e-9))


def yaw_of(box):
    return math.atan2(box[6], box[7])


def wrap_abs_angle(a):
    return abs((a + math.pi) % (2 * math.pi) - math.pi)


# ───────────────────────── 진단 계측 ─────────────────────────

def focal_elements(logits, target, alpha=0.25, gamma=2.0):
    p = 1.0 / (1.0 + np.exp(-logits))
    pt = p * target + (1 - p) * (1 - target)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    ce = -(target * np.log(np.clip(p, 1e-9, 1)) + (1 - target) * np.log(np.clip(1 - p, 1e-9, 1)))
    return a_t * (1 - pt) ** gamma * ce


def bce(t, q):
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return -(t * np.log(q) + (1 - t) * np.log(1 - q))


def entropy(t):
    t = np.clip(t, 1e-9, 1 - 1e-9)
    return -(t * np.log(t) + (1 - t) * np.log(1 - t))


def current_cost_components(cls_np, box_np, gt_cls, gt_box):
    """legacy(구 학습) matcher 계약의 cost 행렬을 성분별로 재현.
    반환: dict of [900, M] arrays (cls/xy/z/dim/yaw), total

    용도: 기준선/pilot 과의 연속 비교용 **고정 측정 계약**. parity 학습 모델에
    이 값을 적용한 결과는 실제 학습 assignment 가 아니라 '기하 모호성 proxy'다
    (yaw counterfactual 포함) — 실제 학습 assignment 진단은 parity 블록을 볼 것.
    """
    p = 1.0 / (1.0 + np.exp(-cls_np))                      # [900,2]
    c_cls = DIAG_COST_CLASS * (-p[:, gt_cls])              # [900,M]
    scale = np.asarray(BOX_SCALE, dtype=np.float64)
    d = np.abs(box_np[:, None, :8] / scale[:8] - gt_box[None, :, :8] / scale[:8])
    c_xy = DIAG_COST_BBOX * d[..., 0:2].sum(-1)
    c_z = DIAG_COST_BBOX * d[..., 2]
    c_dim = DIAG_COST_BBOX * d[..., 3:6].sum(-1)
    c_yaw = DIAG_COST_BBOX * d[..., 6:8].sum(-1)
    total = c_cls + c_xy + c_z + c_dim + c_yaw
    return {"cls": c_cls, "xy": c_xy, "z": c_z, "dim": c_dim, "yaw": c_yaw, "total": total}


def parity_cost_components(cls_np, box_np, gt_cls, gt_box, alpha=0.25, gamma=2.0, eps=1e-12):
    """parity(공식 Stage-1) matcher 계약 재현 — loss_calculator.ParityHungarianMatcher 동일.
    cls = focal(pos-neg)×2, box = |Δ|·[2,2,2,.5,.5,.5,0,0,0,0]×0.25 (yaw/vel 제외).
    yaw 성분은 정의상 0 → counterfactual 없음."""
    p = 1.0 / (1.0 + np.exp(-cls_np))
    neg = -np.log(1 - p + eps) * (1 - alpha) * p ** gamma
    pos = -np.log(p + eps) * alpha * (1 - p) ** gamma
    c_cls = (pos - neg)[:, gt_cls] * 2.0
    d = np.abs(box_np[:, None, :6] - gt_box[None, :, :6])
    c_xy = 0.25 * 2.0 * d[..., 0:2].sum(-1)
    c_z = 0.25 * 2.0 * d[..., 2]
    c_dim = 0.25 * 0.5 * d[..., 3:6].sum(-1)
    c_yaw = np.zeros_like(c_z)
    total = c_cls + c_xy + c_z + c_dim
    return {"cls": c_cls, "xy": c_xy, "z": c_z, "dim": c_dim, "yaw": c_yaw, "total": total}


class DiagAccum:
    """진단 지표 누적기 (final layer + per-layer 요약).

    contract='legacy'  : 구 학습 matcher 로 매칭 — 기준선/pilot 연속 비교용
                         고정 측정 계약. yaw-zero counterfactual 은 기하 모호성 proxy.
    contract='parity'  : 공식 Stage-1 matcher 로 매칭 — parity 학습 모델의
                         실제 training assignment 진단. yaw counterfactual 없음.
    """

    def __init__(self, num_layers, contract="legacy"):
        assert contract in ("legacy", "parity")
        self.contract = contract
        self.cost_fn = (current_cost_components if contract == "legacy"
                        else parity_cost_components)
        self.num_layers = num_layers
        self.p_pos = []                       # final layer matched confidence
        self.p_pos_layer = [[] for _ in range(num_layers)]
        self.agree_layer = np.zeros(num_layers)   # vs final layer, matched query 일치 수
        self.agree_total = np.zeros(num_layers)
        self.focal_pos_sum = 0.0
        self.focal_neg_sum = 0.0
        self.num_fg = 0
        self.cost_rows = {k: [] for k in ("cls", "xy", "z", "dim", "yaw")}
        self.margin = []
        self.yaw_gt_xy = 0
        self.cf_changed = 0                   # counterfactual(yaw=0) assignment 변경 수
        self.cf_total = 0
        self.cf_xy_with = []
        self.cf_xy_without = []
        self.err_xy = []
        self.err_3d = []
        self.err_yaw = []
        self.q_pos = []
        self.t_cur = []
        self.t_off = []
        self.bg_q_hist = np.zeros(101)        # background quality 히스토그램 (0..1, 0.01 bin)

    def frame(self, all_cls, all_box, all_q, gt_cls, gt_box):
        L = all_cls.shape[0]
        gt_m = gt_cls.shape[0]
        if gt_m == 0:
            # background quality 만 누적 (final layer)
            q_all = 1.0 / (1.0 + np.exp(-all_q[-1][:, 0]))
            np.add.at(self.bg_q_hist, np.clip((q_all * 100).astype(int), 0, 100), 1)
            return

        # 층별 matching (self.contract 계약) — layer0 은 temporal 교체로 slot 비교 불가 → 일치율 제외
        matches = []
        for li in range(L):
            comp = self.cost_fn(all_cls[li], all_box[li], gt_cls, gt_box)
            pi, gi = linear_sum_assignment(np.nan_to_num(comp["total"], nan=1e6, posinf=1e6))
            matches.append((pi, gi, comp))
            p = 1.0 / (1.0 + np.exp(-all_cls[li]))
            self.p_pos_layer[li].extend(p[pi, gt_cls[gi]].tolist())

        pi, gi, comp = matches[-1]
        # gt 순서로 정렬해 층간 비교 가능하게
        ord_f = np.argsort(gi)
        pi_f, gi_f = pi[ord_f], gi[ord_f]
        for li in range(1, L - 1):            # layer0 제외, final 제외
            pl, gl, _ = matches[li]
            ord_l = np.argsort(gl)
            self.agree_layer[li] += int((pl[ord_l] == pi_f).sum())
            self.agree_total[li] += gt_m

        p = 1.0 / (1.0 + np.exp(-all_cls[-1]))
        self.p_pos.extend(p[pi, gt_cls[gi]].tolist())

        # focal 분해 (final layer, 학습과 동일한 target 구성)
        target = np.zeros_like(all_cls[-1])
        target[pi, gt_cls[gi]] = 1.0
        el = focal_elements(all_cls[-1], target)
        self.focal_pos_sum += float(el[target == 1].sum())
        self.focal_neg_sum += float(el[target == 0].sum())
        self.num_fg += int(len(pi))

        # cost 성분 (선택된 match)
        for k in self.cost_rows:
            self.cost_rows[k].extend(comp[k][pi, gi].tolist())
        self.yaw_gt_xy += int((np.abs(comp["yaw"][pi, gi]) > np.abs(comp["xy"][pi, gi])).sum())
        # margin: 각 gt 열에서 (chosen 제외 최소) − chosen
        for a, j in zip(pi, gi):
            col = comp["total"][:, j].copy()
            chosen = col[a]
            col[a] = np.inf
            self.margin.append(float(col.min() - chosen))

        # counterfactual(yaw=0 재할당): legacy 계약에서만 의미 있음 —
        # parity 는 yaw cost 가 0 이라 정의상 변경률 0 (기하 모호성 proxy 로만 사용).
        if self.contract == "legacy":
            ct = comp["total"] - comp["yaw"]
            pc, gc = linear_sum_assignment(np.nan_to_num(ct, nan=1e6, posinf=1e6))
            ord_c = np.argsort(gc)
            self.cf_changed += int((pc[ord_c] != pi_f).sum())
            self.cf_total += gt_m
            xy_w = np.linalg.norm(all_box[-1][pi_f, :2] - gt_box[gi_f, :2], axis=1)
            xy_wo = np.linalg.norm(all_box[-1][pc[ord_c], :2] - gt_box[gc[ord_c], :2], axis=1)
            self.cf_xy_with.extend(xy_w.tolist())
            self.cf_xy_without.extend(xy_wo.tolist())

        # matched 오차 + quality
        d_xy = np.linalg.norm(all_box[-1][pi, :2] - gt_box[gi, :2], axis=1)
        d_3d = np.linalg.norm(all_box[-1][pi, :3] - gt_box[gi, :3], axis=1)
        self.err_xy.extend(d_xy.tolist())
        self.err_3d.extend(d_3d.tolist())
        for a, j in zip(pi, gi):
            self.err_yaw.append(wrap_abs_angle(yaw_of(all_box[-1][a]) - yaw_of(gt_box[j])))
        q_all = 1.0 / (1.0 + np.exp(-all_q[-1][:, 0]))
        self.q_pos.extend(q_all[pi].tolist())
        self.t_cur.extend(np.exp(-d_xy / 4.0).tolist())
        self.t_off.extend(np.exp(-d_3d).tolist())
        bg = np.ones(all_cls[-1].shape[0], dtype=bool)
        bg[pi] = False
        np.add.at(self.bg_q_hist, np.clip((q_all[bg] * 100).astype(int), 0, 100), 1)

    def summarize(self):
        def qtiles(x):
            if not len(x):
                return None
            a = np.asarray(x)
            return {"p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)),
                    "p50": float(np.percentile(a, 50)), "p75": float(np.percentile(a, 75)),
                    "p90": float(np.percentile(a, 90)), "mean": float(a.mean()), "n": len(a)}

        p = np.asarray(self.p_pos) if self.p_pos else np.zeros(0)
        q = np.asarray(self.q_pos) if self.q_pos else np.zeros(0)
        out = {
            "positive_confidence": qtiles(self.p_pos),
            "positive_conf_frac": {
                "lt_0.05": float((p < 0.05).mean()) if len(p) else None,
                "lt_0.10": float((p < 0.10).mean()) if len(p) else None,
                "lt_0.30": float((p < 0.30).mean()) if len(p) else None,
            },
            "positive_conf_by_layer": {
                f"layer{li}": qtiles(v) for li, v in enumerate(self.p_pos_layer)
            },
            "assignment_agreement_vs_final": {
                f"layer{li}": (float(self.agree_layer[li] / self.agree_total[li])
                               if self.agree_total[li] else None)
                for li in range(1, self.num_layers - 1)
            },
            "focal": {
                "pos_per_fg": self.focal_pos_sum / max(self.num_fg, 1),
                "neg_per_fg": self.focal_neg_sum / max(self.num_fg, 1),
                "num_fg": self.num_fg,
            },
            "matching_cost_components": {k: qtiles(v) for k, v in self.cost_rows.items()},
            "cost_margin_to_second": qtiles(self.margin),
            "matched_errors": {
                "xy_m": qtiles(self.err_xy),
                "d3d_m": qtiles(self.err_3d),
                "yaw_rad": qtiles(self.err_yaw),
            },
            "contract": self.contract,
        }
        if self.contract == "legacy":
            # 이하 두 지표는 legacy 측정 계약 전용 — parity 학습 모델에서는
            # '실제 assignment 의 yaw 의존'이 아니라 기하 모호성 proxy 로 읽을 것.
            out["frac_yaw_cost_gt_xy_cost"] = self.yaw_gt_xy / max(self.num_fg, 1)
            out["counterfactual_yaw_zero_proxy"] = {
                "assignment_changed_frac": self.cf_changed / max(self.cf_total, 1),
                "matched_xy_with_yaw": qtiles(self.cf_xy_with),
                "matched_xy_without_yaw": qtiles(self.cf_xy_without),
            }
        # quality 정보량 (현재 target / 공식 target 각각)
        for name, t_arr in (("current_exp(-dxy/4)", self.t_cur), ("official_exp(-d3d)", self.t_off)):
            if not len(t_arr):
                continue
            t = np.asarray(t_arr)
            ce = float(bce(t, q).mean())
            h_cond = float(entropy(t).mean())
            h_marg = float(entropy(np.asarray([t.mean()]))[0])
            rho_q_t = spearmanr(q, t).correlation if len(q) > 2 else None
            out.setdefault("quality_information", {})[name] = {
                "CE": ce, "H_cond": h_cond, "H_marg": h_marg,
                "IG": h_marg - ce, "regret": ce - h_cond,
                "budget": h_marg - h_cond,
                "spearman_q_vs_target": None if rho_q_t is None or np.isnan(rho_q_t) else float(rho_q_t),
                "target_mean": float(t.mean()),
            }
        if len(q) > 2 and len(self.err_3d):
            rho = spearmanr(q, np.asarray(self.err_3d)).correlation
            out["spearman_quality_vs_d3d"] = None if np.isnan(rho) else float(rho)
        hist = self.bg_q_hist
        if hist.sum() > 0:
            cdf = np.cumsum(hist) / hist.sum()
            centers = np.arange(101) / 100.0
            out["background_quality"] = {
                "mean": float((hist * centers).sum() / hist.sum()),
                "p50": float(centers[np.searchsorted(cdf, 0.5)]),
                "p90": float(centers[np.searchsorted(cdf, 0.9)]),
                "n": int(hist.sum()),
            }
        return out


# ───────────────────────── main ─────────────────────────

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ARGS.out_dir or os.path.join(ARGS.run_dir, "ap_eval")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = ARGS.ckpt if os.path.isabs(ARGS.ckpt) else os.path.join(ARGS.run_dir, ARGS.ckpt)
    print(f"[eval] ckpt={ckpt_path}")
    print(f"[eval] device={device}, roi_max={ARGS.roi_max_dist}m, "
          f"variants={VARIANTS}, diag={not ARGS.no_diag}")

    # ── 모델 (run 계약과 동일; depth head 는 평가에 불필요) ──
    model = AutoNavModel(
        hidden_dim=256, num_classes=2, num_decoder_layers=6,
        pretrained_backbone=False,
        use_temporal_memory=bool(RUN_CFG.get("use_temporal_memory", True)),
        num_temp_instances=int(RUN_CFG.get("num_temp_instances", 600)),
        use_grid_mask=False, use_dense_depth=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k: v for k, v in sd.items() if not k.startswith("depth_net.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    real_missing = [k for k in missing if not k.startswith("depth_net.")]
    if real_missing or unexpected:
        raise RuntimeError(f"state_dict 불일치: missing={real_missing}, unexpected={unexpected}")
    model.to(device).eval()
    model.reset_temporal_memory()

    # ── val loader (학습 val 과 동일 계약: streaming, shuffle=False) ──
    val_ds = MoraiTemporalDataset(
        dataset_root=ARGS.dataset_root, split="val",
        val_scenarios=RUN_CFG["val_scenarios"],
        load_depth=False, gt_version=RUN_CFG.get("val_gt_version", "v3"),
    )
    sampler = StreamingGroupSampler(val_ds, batch_size=ARGS.batch_size,
                                    shuffle=False, seed=0, drop_uneven_tail=False)
    loader = DataLoader(val_ds, batch_sampler=sampler,
                        collate_fn=morai_temporal_collate_fn, num_workers=ARGS.num_workers)
    print(f"[eval] val frames={len(val_ds):,}, batches={len(loader):,}")

    # ── 누적 컨테이너 ──
    class_ids = sorted(CLASS_ID_NAMES.keys())
    preds = {v: {c: {"score": [], "frame": [], "xy": [], "box": []} for c in class_ids}
             for v in VARIANTS}
    gts = {c: {} for c in class_ids}          # class → frame → [M,2] xy
    gt_full = {c: {} for c in class_ids}      # class → frame → [M,11]+vel_valid
    n_gt = {c: 0 for c in class_ids}
    roi_dropped_pred = 0
    roi_dropped_gt = 0
    # 이중 진단: legacy(기준선/pilot 연속 비교) + parity(실제 학습 assignment)
    diag = DiagAccum(num_layers=6, contract="legacy") if not ARGS.no_diag else None
    diag_parity = DiagAccum(num_layers=6, contract="parity") if not ARGS.no_diag else None
    # per-frame decode 덤프 (offline 변형 재계산용 — forward 재실행 방지)
    dump = {"frame": [], "box": [], "label": [], "raw": [], "cns": []}
    n_pre_total = 0      # prefilter 통과 예측 수 (NMS 전)
    n_post_total = 0     # NMS 후 예측 수

    frame_no = 0
    with torch.no_grad():
        for batch in loader:
            if ARGS.max_frames and frame_no >= ARGS.max_frames:
                break
            images = batch["images"].to(device)
            intr_b = batch["intrinsics"]
            extr_b = batch["extrinsics"]
            out = model(
                images, intr_b.to(device), extr_b.to(device),
                stems=batch["stem"], ego_poses=batch["ego_pose"].to(device),
                timestamps=batch.get("timestamp"), return_intermediate=True,
            )
            det_cls = out["det_cls"].detach().float().cpu()
            det_box = out["det_box"].detach().float().cpu()
            det_q = out["det_quality"].detach().float().cpu()
            if diag is not None:
                all_cls = out["all_det_cls"].detach().float().cpu().numpy()
                all_box = out["all_det_box"].detach().float().cpu().numpy()
                all_q = out["all_det_quality"].detach().float().cpu().numpy()

            B = det_cls.shape[0]
            for i in range(B):
                if ARGS.max_frames and frame_no >= ARGS.max_frames:
                    break
                intr = intr_b[i].numpy().astype(np.float64)
                extr = extr_b[i].numpy().astype(np.float64)
                gt_box = batch["dynamic_gt_boxes"][i].numpy().astype(np.float64)
                gt_cls = batch["dynamic_gt_labels"][i].numpy().astype(np.int64).reshape(-1)
                vel_valid = batch["velocity_valid"][i]
                vel_valid = (vel_valid.numpy().astype(np.float64).reshape(-1)
                             if torch.is_tensor(vel_valid) else np.asarray(vel_valid, dtype=np.float64).reshape(-1))

                # GT ROI (정책상 no-op 이어야 함 — 검증 겸 기록)
                if gt_box.shape[0]:
                    g_roi = roi_mask_np(gt_box[:, :3], intr, extr, ARGS.roi_max_dist)
                    roi_dropped_gt += int((~g_roi).sum())
                    gt_box, gt_cls, vel_valid = gt_box[g_roi], gt_cls[g_roi], vel_valid[g_roi]
                for c in class_ids:
                    m = gt_cls == c
                    if m.sum():
                        gts[c][frame_no] = gt_box[m][:, :2].copy()
                        gt_full[c][frame_no] = (gt_box[m].copy(), vel_valid[m].copy())
                        n_gt[c] += int(m.sum())

                dec = decode_official(det_cls[i], det_box[i], det_q[i], intr, extr,
                                      ARGS.roi_max_dist, ARGS.nms_prefilter)
                roi_dropped_pred += dec["roi_dropped"]
                dump["frame"].append(np.full(len(dec["raw"]), frame_no, dtype=np.int32))
                dump["box"].append(dec["boxes"].astype(np.float32))
                dump["label"].append(dec["labels"].astype(np.int8))
                dump["raw"].append(dec["raw"].astype(np.float32))
                dump["cns"].append(dec["cns"].astype(np.float32))
                pre_idx = np.flatnonzero(dec["raw"] >= ARGS.nms_prefilter)
                n_pre_total += len(pre_idx)
                n_post_total += len(dec["nms_idx"])
                sel = {"final_nonms": (np.arange(len(dec["final"])), dec["final"]),
                       "raw_nonms": (np.arange(len(dec["raw"])), dec["raw"]),
                       "final_pre_nonms": (pre_idx, dec["final"][pre_idx]),
                       "final_nms": (dec["nms_idx"], dec["final"][dec["nms_idx"]])}
                for vname, (idx, sc) in sel.items():
                    for c in class_ids:
                        m = dec["labels"][idx] == c
                        if not m.any():
                            continue
                        ii = idx[m]
                        preds[vname][c]["score"].append(sc[m])
                        preds[vname][c]["frame"].append(np.full(int(m.sum()), frame_no))
                        preds[vname][c]["xy"].append(dec["boxes"][ii][:, :2])
                        if vname == "final_nonms":   # box 는 TP error 계산에만 사용
                            preds[vname][c]["box"].append(dec["boxes"][ii])

                if diag is not None:
                    diag.frame(all_cls[i], all_box[i], all_q[i],
                               gt_cls.astype(np.int64), gt_box)
                    diag_parity.frame(all_cls[i], all_box[i], all_q[i],
                                      gt_cls.astype(np.int64), gt_box)
                frame_no += 1

            if frame_no % 2000 < B:
                el = time.time() - t0
                print(f"[eval] {frame_no:,} frames, {el:.0f}s ({frame_no / max(el, 1e-9):.1f} fps)",
                      flush=True)

    print(f"[eval] forward done: {frame_no:,} frames in {time.time() - t0:.0f}s")

    # ── AP 계산 ──
    results = {"meta": {
        "ckpt": ckpt_path, "tag": ARGS.tag, "frames": frame_no,
        "roi_max_dist_m": ARGS.roi_max_dist,
        "roi_dropped_pred": roi_dropped_pred, "roi_dropped_gt": roi_dropped_gt,
        "nms_prefilter": ARGS.nms_prefilter,
        "n_gt": {CLASS_ID_NAMES[c]: n_gt[c] for c in class_ids},
        "val_scenarios": RUN_CFG["val_scenarios"],
        "protocol": "nuScenes-style front-ROI AP (101pt, min_recall=0.1, min_precision=0.1); "
                    "TP errors = mean over TP@2m (simplified)",
    }, "ap": {}, "tp_errors": {}, "bucket_ap_2m": {}}

    for vname in VARIANTS:
        results["ap"][vname] = {}
        for c in class_ids:
            cname = CLASS_ID_NAMES[c]
            P = preds[vname][c]
            if not P["score"]:
                results["ap"][vname][cname] = {str(t): None for t in DIST_THRESHOLDS}
                continue
            score = np.concatenate(P["score"])
            frame = np.concatenate(P["frame"])
            xy = np.concatenate(P["xy"])
            ap_c = {}
            for th in DIST_THRESHOLDS:
                tp_sorted, _ = greedy_pr(score, frame, xy, gts[c], th)
                ap, _, _ = nusc_ap(tp_sorted, n_gt[c])
                ap_c[str(th)] = ap
            ap_c["mean"] = float(np.nanmean([v for v in ap_c.values() if v is not None]))
            results["ap"][vname][cname] = ap_c
        vals = [results["ap"][vname][CLASS_ID_NAMES[c]].get("mean") for c in class_ids]
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        results["ap"][vname]["mAP"] = float(np.mean(vals)) if vals else None

    # ── TP errors @2m (final_nonms) + bucket AP ──
    for c in class_ids:
        cname = CLASS_ID_NAMES[c]
        P = preds["final_nonms"][c]
        if not P["score"]:
            continue
        score = np.concatenate(P["score"])
        frame = np.concatenate(P["frame"])
        xy = np.concatenate(P["xy"])
        box = np.concatenate(P["box"])
        order = np.argsort(-score, kind="stable")
        used = {}
        errs = {"ate_xy": [], "ate_3d": [], "ase": [], "aoe_rad": [], "ave": []}
        for i in order:
            f = frame[i]
            if f not in gt_full[c]:
                continue
            g_box, g_vv = gt_full[c][f]
            u = used.setdefault(f, np.zeros(g_box.shape[0], dtype=bool))
            d = np.linalg.norm(g_box[:, :2] - xy[i], axis=1)
            d[u] = np.inf
            j = int(np.argmin(d))
            if d[j] <= TP_ERR_DIST:
                u[j] = True
                errs["ate_xy"].append(float(d[j]))
                errs["ate_3d"].append(float(np.linalg.norm(g_box[j, :3] - box[i, :3])))
                errs["ase"].append(aligned_size_error(box[i], g_box[j]))
                errs["aoe_rad"].append(wrap_abs_angle(yaw_of(box[i]) - yaw_of(g_box[j])))
                if g_vv[j] > 0.5:
                    errs["ave"].append(float(np.linalg.norm(box[i, 8:10] - g_box[j, 8:10])))
        results["tp_errors"][cname] = {
            k: (float(np.mean(v)) if v else None) for k, v in errs.items()
        }
        results["tp_errors"][cname]["n_tp"] = len(errs["ate_xy"])
        results["tp_errors"][cname]["ave_coverage"] = (
            len(errs["ave"]) / max(len(errs["ate_xy"]), 1))

        # bucket AP@2m: GT/pred 를 radial bucket 으로 각각 제한
        results["bucket_ap_2m"][cname] = {}
        for lo, hi in BUCKETS:
            g_b = {}
            npos_b = 0
            for f, g in gts[c].items():
                r = np.linalg.norm(g, axis=1)
                m = (r >= lo) & (r < hi)
                if m.sum():
                    g_b[f] = g[m]
                    npos_b += int(m.sum())
            r_p = np.linalg.norm(xy, axis=1)
            m_p = (r_p >= lo) & (r_p < hi)
            if npos_b == 0:
                results["bucket_ap_2m"][cname][f"{lo:.0f}-{hi:.0f}m"] = None
                continue
            tp_sorted, _ = greedy_pr(score[m_p], frame[m_p], xy[m_p], g_b, 2.0)
            ap, _, _ = nusc_ap(tp_sorted, npos_b)
            results["bucket_ap_2m"][cname][f"{lo:.0f}-{hi:.0f}m"] = ap

    # ── NMS 구조 지표 + duplicate FP 분해 (Phase 2 판정표 추가분) ──
    # 무인 체인에서 신규 지표 오류로 전체 평가를 잃지 않도록 방어적으로 계산.
    try:
        pre_map = results["ap"]["final_pre_nonms"]["mAP"]
        nms_map = results["ap"]["final_nms"]["mAP"]
        results["nms_effect"] = {
            "pred_per_frame_pre": n_pre_total / max(frame_no, 1),
            "pred_per_frame_post": n_post_total / max(frame_no, 1),
            "nms_removed_frac": 1.0 - n_post_total / max(n_pre_total, 1),
            "gain_abs": (nms_map - pre_map) if (nms_map is not None and pre_map is not None) else None,
            "gain_ratio": (nms_map / pre_map) if (nms_map and pre_map) else None,
        }
        # pre/post NMS 각각의 dup/iso 분해 — iso_pre/iso_post 비가 hallucination
        # 군집 크기의 proxy (NMS 는 GT 근처가 아닌 군집도 제거하므로).
        for key, vname in (("duplicates_pre_nms_2m", "final_pre_nonms"),
                           ("duplicates_post_nms_2m", "final_nms")):
            results[key] = {}
            for c in class_ids:
                P = preds[vname][c]
                if not P["score"]:
                    continue
                results[key][CLASS_ID_NAMES[c]] = dup_fp_stats(
                    np.concatenate(P["score"]), np.concatenate(P["frame"]),
                    np.concatenate(P["xy"]), gts[c], dist_th=2.0)
    except Exception as e:  # noqa: BLE001
        print(f"[eval] WARN: nms/duplicate 지표 계산 실패 (본 평가에는 영향 없음): {e}")

    if diag is not None:
        results["diagnostics"] = diag.summarize()            # legacy 측정 계약 (연속 비교)
        results["diagnostics_parity"] = diag_parity.summarize()  # 실제 학습 계약

    out_path = os.path.join(out_dir, f"{ARGS.tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[eval] saved → {out_path}")

    # GT xy 도 함께 덤프 → 이후 어떤 score 변형이든 forward 없이 재계산 가능
    gt_dump_frame, gt_dump_xy, gt_dump_cls = [], [], []
    for c in class_ids:
        for f_no, g in gts[c].items():
            gt_dump_frame.append(np.full(g.shape[0], f_no, dtype=np.int32))
            gt_dump_xy.append(g.astype(np.float32))
            gt_dump_cls.append(np.full(g.shape[0], c, dtype=np.int8))
    npz_path = os.path.join(out_dir, f"{ARGS.tag}_preds.npz")
    np.savez_compressed(
        npz_path,
        frame=np.concatenate(dump["frame"]), box=np.concatenate(dump["box"]),
        label=np.concatenate(dump["label"]), raw=np.concatenate(dump["raw"]),
        cns=np.concatenate(dump["cns"]),
        gt_frame=np.concatenate(gt_dump_frame) if gt_dump_frame else np.zeros(0, np.int32),
        gt_xy=np.concatenate(gt_dump_xy) if gt_dump_xy else np.zeros((0, 2), np.float32),
        gt_cls=np.concatenate(gt_dump_cls) if gt_dump_cls else np.zeros(0, np.int8),
    )
    print(f"[eval] preds dump → {npz_path}")

    # ── 요약 출력 ──
    print("\n══════════ AP (nuScenes-style front-ROI) ══════════")
    for vname in VARIANTS:
        r = results["ap"][vname]
        print(f"[{vname}] mAP={r['mAP']:.4f}" if r["mAP"] is not None else f"[{vname}] mAP=n/a")
        for c in class_ids:
            cname = CLASS_ID_NAMES[c]
            a = r[cname]
            row = "  ".join(f"AP@{t}={a[str(t)]:.4f}" if a[str(t)] is not None else f"AP@{t}=n/a"
                            for t in DIST_THRESHOLDS)
            print(f"  {cname:10s} {row}")
    print("\n══════════ TP errors @2m (final_nonms) ══════════")
    for cname, e in results["tp_errors"].items():
        print(f"  {cname:10s} " + "  ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()))
    if diag is not None:
        d = results["diagnostics"]
        print("\n══════════ 진단 요약 ══════════")
        pc = d["positive_confidence"]
        if pc:
            print(f"  positive conf: p50={pc['p50']:.3f} p10={pc['p10']:.3f} "
                  f"(<0.1: {d['positive_conf_frac']['lt_0.10']:.1%}, "
                  f"<0.3: {d['positive_conf_frac']['lt_0.30']:.1%})")
        if "counterfactual_yaw_zero_proxy" in d:
            print(f"  [legacy proxy] yaw>xy cost 비율: {d['frac_yaw_cost_gt_xy_cost']:.1%}, "
                  f"yaw-zero 재할당 변경: "
                  f"{d['counterfactual_yaw_zero_proxy']['assignment_changed_frac']:.1%}")
        dp = results.get("diagnostics_parity")
        if dp:
            pcp = dp["positive_confidence"]
            agp = {k: round(v, 2) for k, v in dp["assignment_agreement_vs_final"].items()
                   if v is not None}
            print(f"  [parity 계약] posconf p50={pcp['p50']:.3f} | layer agree={agp} | "
                  f"matched xy p50={dp['matched_errors']['xy_m']['p50']:.2f}")
        for name, qi in d.get("quality_information", {}).items():
            print(f"  quality[{name}]: IG={qi['IG']:.4f} regret={qi['regret']:.4f} "
                  f"budget={qi['budget']:.4f}")
        if d.get("spearman_quality_vs_d3d") is not None:
            print(f"  spearman(quality, d3d) = {d['spearman_quality_vs_d3d']:.3f}")
    print(f"\n[eval] total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
