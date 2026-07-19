#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MORAI source-header 기반 epoch/glitch-safe 프레임 동기화기.

배경 (geom_verify로 검증됨):
  * MORAI의 모든 토픽 header.stamp는 하나의 공통 source clock에서 나온다.
    이 source clock은 WSL wall clock보다 ~9% 빠르지만(slope a≈1.0899) 모든
    토픽이 같은 slope를 공유하므로 header 값끼리는 직접 비교 가능하다.
    => 절대 age나 공통 slope로 stale 판정하지 않는다. source header끼리만 비교.
  * receive-time 최근접(기존 production)은 source clock glitch 구간에서 서로 다른
    타임라인(정상 vs +1.25s)을 결합해 ~3% 프레임을 최대 1.3s 오배치했다.
  * source-header 최근접은 글리치를 같은 타임라인끼리 자동으로 짝짓는다(교차결합 0).
    다만 글리치 구간에서 두 타임라인이 교대로 배출되므로, 순수 header-nearest는
    낮은 타임라인 프레임을 높은 타임라인 뒤에 저장 -> timestamp 역행이 생긴다.
    => monotonic ceiling + epoch 감지/hold로 역행 0 & 오배치 0을 동시에 보장한다.

동일 코드가 라이브 노드(morai_3d_live.py)와 오프라인 회귀(geom_verify/replay_ab_c.py)
양쪽에서 쓰인다. mode로 A/B/C를 전환한다:
  mode="receive": 기존 production (receive-time nearest, lidar=latest)  -- A
  mode="header" : 순수 source-header nearest (monotonic/epoch 없음)      -- B
  mode="epoch"  : source-header nearest + monotonic + epoch/glitch-safe  -- C (기본)
"""

from collections import deque


# =========================
# 토픽/허용오차 규약 (라이브·오프라인 공통)
# =========================

REF_TOPIC = "/cam_front"

EGO_TOPIC = "/Ego_topic"
OBJ_TOPIC = "/Object_topic"
LEFT_TOPIC = "/cam_front_left"
RIGHT_TOPIC = "/cam_front_right"
LIDAR_TOPIC = "/lidar3D"

# 선택된 메시지의 source-header가 ref(cam_front) header에서 이 이상 벌어지면 폐기.
#   Ego/Object 50ms, 측면 카메라 100ms, LiDAR 100ms.
#   (검증: source-header nearest에서 ego/obj max=11ms, left max=54ms,
#    right max=95ms, lidar max=71ms -> 모두 게이트 내. glitch 1.25s는 항상 배제.)
TOPIC_TOL = {
    EGO_TOPIC: 0.050,
    OBJ_TOPIC: 0.050,
    LEFT_TOPIC: 0.100,
    RIGHT_TOPIC: 0.100,
    LIDAR_TOPIC: 0.100,
}

# receive-time 모드(A)의 기존 production 허용오차.
RECEIVE_GAP = 0.10

# epoch jump 임계. 근거(동일 bag 분포):
#   정상 카메라 forward header dt max ≈ 151~178ms (p99 ≈ 146~171ms).
#   글리치 |step| min ≈ 1.17s.  0.5s는 정상(≤0.18s)과 글리치(≥1.17s) 사이 넓은 마진.
EPOCH_JUMP_THRESH = 0.5

# epoch jump 후 저장 재개까지 요구하는 연속 monotonic ref 개수.
CONFIRM_N = 3

# ref(cam_front) 결정을 지연시키는 receive-time settle 창(초).
#   근거: ego/obj(46Hz)는 항상 조밀하지만 측면 카메라/LiDAR(~10Hz)는 같은 sim tick의
#   프레임이 cam_front보다 '조금 늦게' 도착할 수 있다(인과적 스트리밍). 즉시 결정하면
#   직전(≈100ms 이전) 프레임만 보여 게이트를 넘겨 불필요하게 drop된다.
#   한 측면-카메라 주기(≈100ms) + 지터 마진. 오프라인 GT 수집이라 이 지연은 무해하다.
#   (검증: settle 적용 시 측면/LiDAR sync_gap drop이 정상주행 구간에서 거의 사라짐.)
SETTLE_SEC = 0.15

# 버퍼 길이(선택 탐색 범위 제한). 기존 production과 동일 규모.
DEFAULT_MAXLEN = {
    EGO_TOPIC: 200,
    OBJ_TOPIC: 200,
    LEFT_TOPIC: 100,
    RIGHT_TOPIC: 100,
    LIDAR_TOPIC: 30,
    REF_TOPIC: 100,
}


def _nearest(buf, target):
    """buf=[(src, recv, msg)]에서 |src-target| 최소 항목과 gap 반환."""
    best = None
    best_gap = None
    for item in buf:
        gap = abs(item[0] - target)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = item
    return best, best_gap


def _nearest_recv(buf, target):
    """receive-time 최근접 (A 모드용). buf=[(src, recv, msg)]."""
    best = None
    best_gap = None
    for item in buf:
        gap = abs(item[1] - target)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = item
    return best, best_gap


def _latest_recv(buf):
    """receive 시각 기준 최신 항목 (A 모드 lidar용)."""
    if not buf:
        return None
    return max(buf, key=lambda it: it[1])


class Decision(object):
    """push(ref)의 결정. action='save'면 selected/frame 정보 포함, 'drop'이면 reason."""

    __slots__ = ("action", "reason", "frame_id", "ref_src", "ref_recv",
                 "ref_msg", "epoch_id", "buffer_reset", "selected", "sidecar")

    def __init__(self, action, reason=None, frame_id=None, ref_src=None,
                 ref_recv=None, ref_msg=None, epoch_id=None, buffer_reset=False,
                 selected=None, sidecar=None):
        self.action = action
        self.reason = reason
        self.frame_id = frame_id
        self.ref_src = ref_src
        self.ref_recv = ref_recv
        self.ref_msg = ref_msg
        self.epoch_id = epoch_id
        self.buffer_reset = buffer_reset
        self.selected = selected or {}      # topic -> (src, recv, msg)
        self.sidecar = sidecar or {}


class EpochGlitchSynchronizer(object):
    def __init__(self, mode="epoch", ref_topic=REF_TOPIC, topic_tol=None,
                 epoch_jump_thresh=EPOCH_JUMP_THRESH, confirm_n=CONFIRM_N,
                 settle_sec=SETTLE_SEC, maxlen=None):
        assert mode in ("epoch", "header", "receive")
        self.mode = mode
        self.ref_topic = ref_topic
        self.topic_tol = dict(topic_tol or TOPIC_TOL)
        self.epoch_jump_thresh = epoch_jump_thresh
        self.confirm_n = confirm_n
        self.settle = settle_sec
        maxlen = maxlen or DEFAULT_MAXLEN

        self.sel_topics = [t for t in self.topic_tol.keys()]
        self.buffers = {}
        for t in list(self.topic_tol.keys()) + [ref_topic]:
            self.buffers[t] = deque(maxlen=maxlen.get(t, 200))

        # 상태
        self.frame_idx = 0
        self.last_saved_ref = None      # 저장된 ref의 source ts (monotonic ceiling)
        self.last_saved_ref_recv = None  # A 모드 receive ceiling
        self.last_ref_seen = None       # 직전에 '본' ref source ts (jump 감지용)
        self.epoch_id = 0
        self.epoch_floor = None
        self.holding = False
        self.confirm = []               # 재개 확인용 연속 monotonic streak
        self.pending = deque()          # settle 대기 중인 ref [(src, recv, msg)]
        self.max_recv = None            # 지금까지 본 최대 receive ts (settle 기준)

    # ---------- 공통 push ----------
    def push(self, topic, source_ts, recv_ts, msg):
        """메시지 도착. 결정 가능한 Decision들의 list 반환(0개 이상)."""
        if topic not in self.buffers:
            return []
        self.buffers[topic].append((source_ts, recv_ts, msg))
        if recv_ts is not None and (self.max_recv is None or recv_ts > self.max_recv):
            self.max_recv = recv_ts

        if self.mode == "receive":
            # 기존 production: ref 도착 즉시 결정(지연 없음).
            if topic != self.ref_topic:
                return []
            d = self._emit_receive(source_ts, recv_ts, msg)
            return [d] if d is not None else []

        # header/epoch: ref는 settle 창만큼 지연 후 결정(측면/LiDAR가 도착할 시간 확보).
        if topic == self.ref_topic:
            self.pending.append((source_ts, recv_ts, msg))
        return self._drain(flush=False)

    def _drain(self, flush=False):
        out = []
        while self.pending:
            s, r, m = self.pending[0]
            ready = flush or (r is not None and self.max_recv is not None and
                              (r + self.settle) <= self.max_recv)
            if not ready:
                break
            self.pending.popleft()
            if self.mode == "header":
                out.append(self._decide_header(s, r, m))
            else:
                out.append(self._decide_epoch(s, r, m))
        return out

    def flush(self):
        """스트림 종료(bag 끝/노드 종료) 시 남은 pending ref를 모두 결정."""
        if self.mode == "receive":
            return []
        return self._drain(flush=True)

    def _valid_src(self, s):
        return s is not None and s > 0.0

    # ---------- 선택 (source-header nearest + 게이트) ----------
    def _select_by_header(self, ref_src):
        """모든 필수 토픽을 source-header 최근접으로 선택. 게이트 위반 시 (None, 실패토픽, dt맵)."""
        selected = {}
        dts = {}
        for t in self.sel_topics:
            best, gap = _nearest(self.buffers[t], ref_src)
            if best is None:
                return None, t, dts
            dts[t] = best[0] - ref_src
            if gap > self.topic_tol[t]:
                return None, t, dts
            selected[t] = best
        return selected, None, dts

    def _sidecar(self, ref_src, ref_recv, selected, epoch_id, buffer_reset,
                 drop_reason=None):
        sc = {
            "frame_id": self.frame_idx,
            "epoch_id": epoch_id,
            "ref_topic": self.ref_topic,
            "ref_src_ts": ref_src,
            "ref_recv_ts": ref_recv,
            "buffer_reset": bool(buffer_reset),
            "drop_reason": drop_reason,
            "topics": {},
        }
        for t in self.sel_topics:
            if selected and t in selected:
                s, r, _ = selected[t]
                sc["topics"][t] = {
                    "src_ts": s, "recv_ts": r,
                    "src_dt": s - ref_src, "recv_dt": r - ref_recv,
                }
            else:
                sc["topics"][t] = None
        return sc

    # ---------- C: epoch/glitch-safe ----------
    def _decide_epoch(self, s, r, msg):
        # 0. source header 없음/0 -> receive-time fallback 금지, drop
        if not self._valid_src(s):
            return Decision("drop", reason="ref_no_source_header",
                            ref_src=s, ref_recv=r,
                            sidecar=self._sidecar(s, r, None, self.epoch_id,
                                                  False, "ref_no_source_header"))

        # 1. epoch jump 감지 (직전 본 ref 대비 |Δ| > threshold). 앞/뒤 점프 모두 이벤트.
        jumped = (self.last_ref_seen is not None and
                  abs(s - self.last_ref_seen) > self.epoch_jump_thresh)
        self.last_ref_seen = s

        if jumped:
            self.epoch_id += 1
            self.epoch_floor = s
            self.holding = True
            self.confirm = [s]
            # 이전 타임라인 late arrival 폐기: 새 floor에서 epoch_jump_thresh 넘게
            # 벌어진 버퍼 항목 제거 (버퍼 리셋).
            self._prune_far_from_floor(s)
            return Decision("drop", reason="epoch_jump", ref_src=s, ref_recv=r,
                            ref_msg=msg, epoch_id=self.epoch_id, buffer_reset=True,
                            sidecar=self._sidecar(s, r, None, self.epoch_id, True,
                                                  "epoch_jump"))

        # 2. hold 중이면 연속 monotonic ref가 confirm_n개 쌓일 때까지 저장 보류
        if self.holding:
            if s > self.confirm[-1]:
                self.confirm.append(s)
                if len(self.confirm) >= self.confirm_n:
                    self.holding = False
                    # 통과 -> 아래 정상 저장 경로로 진행
                else:
                    return Decision("drop", reason="epoch_confirm_hold",
                                    ref_src=s, ref_recv=r, ref_msg=msg,
                                    epoch_id=self.epoch_id,
                                    sidecar=self._sidecar(s, r, None, self.epoch_id,
                                                          False, "epoch_confirm_hold"))
            else:
                # 단조가 깨짐(오실레이션 하강) -> streak 재시작, 계속 보류
                self.confirm = [s]
                return Decision("drop", reason="epoch_confirm_hold",
                                ref_src=s, ref_recv=r, ref_msg=msg,
                                epoch_id=self.epoch_id,
                                sidecar=self._sidecar(s, r, None, self.epoch_id,
                                                      False, "epoch_confirm_hold"))

        # 3. monotonic ceiling: 이전 저장 ts 이하의 ref(늦게 온 이전 타임라인)는 폐기
        if self.last_saved_ref is not None and s <= self.last_saved_ref:
            return Decision("drop", reason="ref_not_monotonic", ref_src=s,
                            ref_recv=r, ref_msg=msg, epoch_id=self.epoch_id,
                            sidecar=self._sidecar(s, r, None, self.epoch_id, False,
                                                  "ref_not_monotonic"))

        # 4. 필수 토픽 source-header 최근접 선택 + 게이트
        selected, fail_topic, _ = self._select_by_header(s)
        if selected is None:
            reason = "sync_gap:%s" % fail_topic
            return Decision("drop", reason=reason, ref_src=s, ref_recv=r,
                            ref_msg=msg, epoch_id=self.epoch_id,
                            sidecar=self._sidecar(s, r, None, self.epoch_id, False,
                                                  reason))

        # 5. 저장
        fid = self.frame_idx
        dec = Decision("save", frame_id=fid, ref_src=s, ref_recv=r, ref_msg=msg,
                       epoch_id=self.epoch_id, selected=selected,
                       sidecar=self._sidecar(s, r, selected, self.epoch_id, False))
        self.last_saved_ref = s
        self.frame_idx += 1
        return dec

    def _prune_far_from_floor(self, floor):
        for t in self.buffers:
            buf = self.buffers[t]
            kept = [it for it in buf if abs(it[0] - floor) <= self.epoch_jump_thresh]
            buf.clear()
            buf.extend(kept)

    # ---------- B: 순수 source-header nearest ----------
    def _decide_header(self, s, r, msg):
        if not self._valid_src(s):
            return Decision("drop", reason="ref_no_source_header", ref_src=s,
                            ref_recv=r,
                            sidecar=self._sidecar(s, r, None, 0, False,
                                                  "ref_no_source_header"))
        selected, fail_topic, _ = self._select_by_header(s)
        if selected is None:
            reason = "sync_gap:%s" % fail_topic
            return Decision("drop", reason=reason, ref_src=s, ref_recv=r,
                            ref_msg=msg, epoch_id=0,
                            sidecar=self._sidecar(s, r, None, 0, False, reason))
        fid = self.frame_idx
        dec = Decision("save", frame_id=fid, ref_src=s, ref_recv=r, ref_msg=msg,
                       epoch_id=0, selected=selected,
                       sidecar=self._sidecar(s, r, selected, 0, False))
        self.frame_idx += 1
        return dec

    # ---------- A: 기존 production (receive-time nearest, lidar=latest) ----------
    def _emit_receive(self, s, r, msg):
        # ref 중복 방지(기존 production last_processed_ref_ts와 동일 취지)
        if self.last_saved_ref_recv is not None and abs(r - self.last_saved_ref_recv) < 1e-6:
            return Decision("drop", reason="dup_ref", ref_src=s, ref_recv=r)
        selected = {}
        for t in self.sel_topics:
            if t == LIDAR_TOPIC:
                best = _latest_recv(self.buffers[t])
                if best is None:
                    return self._recv_drop("sync_gap:%s" % t, s, r, msg)
                selected[t] = best
                continue
            best, gap = _nearest_recv(self.buffers[t], r)
            if best is None or gap > RECEIVE_GAP:
                return self._recv_drop("sync_gap:%s" % t, s, r, msg)
            selected[t] = best
        fid = self.frame_idx
        dec = Decision("save", frame_id=fid, ref_src=s, ref_recv=r, ref_msg=msg,
                       epoch_id=0, selected=selected,
                       sidecar=self._sidecar_recv(s, r, selected))
        self.last_saved_ref_recv = r
        self.frame_idx += 1
        return dec

    def _recv_drop(self, reason, s, r, msg):
        return Decision("drop", reason=reason, ref_src=s, ref_recv=r, ref_msg=msg,
                        epoch_id=0, sidecar=self._sidecar_recv(s, r, None, reason))

    def _sidecar_recv(self, ref_src, ref_recv, selected, drop_reason=None):
        sc = {
            "frame_id": self.frame_idx, "epoch_id": 0, "ref_topic": self.ref_topic,
            "ref_src_ts": ref_src, "ref_recv_ts": ref_recv, "buffer_reset": False,
            "drop_reason": drop_reason, "topics": {},
        }
        for t in self.sel_topics:
            if selected and t in selected:
                s, rr, _ = selected[t]
                sc["topics"][t] = {"src_ts": s, "recv_ts": rr,
                                   "src_dt": (s - ref_src) if (ref_src and s) else None,
                                   "recv_dt": rr - ref_recv}
            else:
                sc["topics"][t] = None
        return sc


# 사이드카 CSV 컬럼 (라이브·오프라인 공통)
SIDECAR_FIELDS = [
    "stem", "frame_id", "action", "epoch_id", "buffer_reset", "drop_reason",
    "ref_src_ts", "ref_recv_ts",
    "ego_src_ts", "ego_recv_ts", "ego_src_dt_ms",
    "obj_src_ts", "obj_recv_ts", "obj_src_dt_ms",
    "left_src_ts", "left_recv_ts", "left_src_dt_ms",
    "right_src_ts", "right_recv_ts", "right_src_dt_ms",
    "lidar_src_ts", "lidar_recv_ts", "lidar_src_dt_ms",
]

_SC_KEY = {"ego": EGO_TOPIC, "obj": OBJ_TOPIC, "left": LEFT_TOPIC,
           "right": RIGHT_TOPIC, "lidar": LIDAR_TOPIC}


def sidecar_row(stem, action, sc):
    """sidecar dict -> SIDECAR_FIELDS 순서의 리스트."""
    def f(x, nd=6):
        return "" if x is None else ("%.*f" % (nd, x))
    row = [stem, sc.get("frame_id"), action, sc.get("epoch_id"),
           int(bool(sc.get("buffer_reset"))), sc.get("drop_reason") or "",
           f(sc.get("ref_src_ts")), f(sc.get("ref_recv_ts"))]
    for k in ("ego", "obj", "left", "right", "lidar"):
        td = sc.get("topics", {}).get(_SC_KEY[k])
        if td:
            row += [f(td.get("src_ts")), f(td.get("recv_ts")),
                    f((td.get("src_dt") * 1000.0) if td.get("src_dt") is not None else None, 1)]
        else:
            row += ["", "", ""]
    return row
