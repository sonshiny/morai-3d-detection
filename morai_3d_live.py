#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
import threading
from collections import deque

import cv2
import numpy as np
import rospy

from sensor_msgs.msg import CompressedImage, PointCloud2
from sensor_msgs import point_cloud2
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList

from morai_dataset import box_visible_in_any_camera, _box_corners_ego
from camera_configs import INTRINSICS
from verify_lidar_camera_overlay import project_body, _BOX_EDGES

import morai_sync
from morai_sync import (
    EpochGlitchSynchronizer,
    EGO_TOPIC, OBJ_TOPIC, LEFT_TOPIC, RIGHT_TOPIC, LIDAR_TOPIC as SYNC_LIDAR_TOPIC,
    SIDECAR_FIELDS, sidecar_row,
)


# =========================
# 3 Camera Topics
# =========================

CAMERA_TOPICS = {
    "/cam_front": "cam_front",
    "/cam_front_left": "cam_front_left",
    "/cam_front_right": "cam_front_right",
}

REFERENCE_CAMERA_TOPIC = "/cam_front"

LIDAR_TOPIC = "/lidar3D"


# =========================
# Ego 기준 수집 범위 설정
# =========================
# x: Ego 전방 방향
# y: Ego 좌우 방향
# z: 높이 방향

FRONT_RANGE_MIN = 0.0       # Ego 뒤쪽 객체 제거
FRONT_RANGE_MAX = 60.0      # Ego 전방 60m

LATERAL_RANGE_LEFT = 30.0   # Ego 좌측 30m
LATERAL_RANGE_RIGHT = 30.0  # Ego 우측 30m

MAX_RANGE_Z = 3.0           # z 방향 ±3m


# =========================
# class_id 정의
# =========================

VEHICLE_CLASS_ID = 0
PEDESTRIAN_CLASS_ID = 1

CLASS_NAMES = {
    VEHICLE_CLASS_ID: "vehicle",
    PEDESTRIAN_CLASS_ID: "pedestrian",
}


# 차량 NPC type 필터
VALID_NPC_TYPES = {1}

# 보행자는 MORAI 설정에 따라 type이 다를 수 있으므로 전체 허용
VALID_PEDESTRIAN_TYPES = None


# 차량은 MORAI 기준점이 rear axle이라 박스 중심까지 고정 거리를 heading 방향으로 밀어준다.
# 실측(라이다-카메라 오버레이 대조)으로 확정한 rear axle -> box center 거리(m).
VEHICLE_REAR_AXLE_TO_CENTER = 1.28

# 보행자는 기준점이 이미 중심이므로 보정하지 않는다.
PEDESTRIAN_OFFSET_DIST = 0.0


# =========================
# 기하 보정 디버그 (임시 검증용)
# =========================
# True로 바꾸면 length_axis_err/offset_axis_err가 임계값을 넘을 때 예외를 던진다.
# 지금은 로그로 여러 프레임 값을 눈으로 확인하는 단계라 False로 둔다.
STRICT_ASSERTS = False

LENGTH_AXIS_ERR_THRESH_DEG = 5.0
OFFSET_AXIS_ERR_THRESH_DEG = 5.0

# cam_front에 보정된 3D 박스를 그려서 저장할 디버그 오버레이 최대 프레임 수 (0이면 끔)
GEOMETRY_OVERLAY_MAX_FRAMES = 30


CSV_HEADER = [
    "frame_id",
    "timestamp",
    "object_source",
    "object_index",
    "object_type",
    "class_id",
    "class_name",
    "x",
    "y",
    "z",
    "w",
    "l",
    "h",
    "ln_w",
    "ln_l",
    "ln_h",
    "sin_yaw",
    "cos_yaw",
    "vx",
    "vy",
    "vz"
]

EGO_POSE_HEADER = [
    "frame_id",
    "timestamp",
    "ego_x",
    "ego_y",
    "ego_z",
    "ego_heading_deg",
    "ego_yaw_rad",
]


def create_next_scenario_dir(dataset_root, prefix="scen", digits=2):
    """
    dataset_root 아래에 scen01, scen02, scen03 ... 형태의 폴더를 자동 생성.
    이미 있는 마지막 번호를 찾아 다음 번호로 새 폴더를 만든다.
    """

    os.makedirs(dataset_root, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_idx = 0

    for name in os.listdir(dataset_root):
        path = os.path.join(dataset_root, name)

        if not os.path.isdir(path):
            continue

        match = pattern.match(name)

        if match:
            idx = int(match.group(1))
            max_idx = max(max_idx, idx)

    next_idx = max_idx + 1
    scenario_name = f"{prefix}{next_idx:0{digits}d}"
    scenario_dir = os.path.join(dataset_root, scenario_name)

    os.makedirs(scenario_dir, exist_ok=False)

    return scenario_dir, scenario_name


def world_to_ego(obj_pos, obj_heading_deg, obj_velocity,
                 ego_pos, ego_heading_deg, offset_dist):
    """
    MORAI world 좌표의 객체를 Ego 차량 기준 좌표로 변환.

    변환 후:
      x_e > 0 : Ego 전방
      x_e < 0 : Ego 후방
      y_e     : Ego 좌우 방향

    offset_dist: 기준점(MORAI가 주는 위치)에서 박스 중심까지 heading 방향으로
                 밀어줄 고정 거리(m). 차량은 rear axle -> center 거리, 보행자는 0.
    """

    obj_yaw_rad = np.radians(obj_heading_deg)

    obj_pos_corrected = np.array([
        obj_pos[0] + offset_dist * np.cos(obj_yaw_rad),
        obj_pos[1] + offset_dist * np.sin(obj_yaw_rad),
        obj_pos[2]
    ], dtype=np.float32)

    dp = obj_pos_corrected - np.array(ego_pos, dtype=np.float32)

    yaw = np.radians(ego_heading_deg)
    c, s = np.cos(yaw), np.sin(yaw)

    x_e = c * dp[0] + s * dp[1]
    y_e = -s * dp[0] + c * dp[1]
    z_e = dp[2]

    rel_yaw_raw = np.radians(obj_heading_deg - ego_heading_deg)
    rel_yaw = np.arctan2(np.sin(rel_yaw_raw), np.cos(rel_yaw_raw))

    vx_w, vy_w, vz_w = obj_velocity
    vx_e = c * vx_w + s * vy_w
    vy_e = -s * vx_w + c * vy_w
    vz_e = vz_w

    return (
        np.array([x_e, y_e, z_e], dtype=np.float32),
        rel_yaw,
        np.array([vx_e, vy_e, vz_e], dtype=np.float32)
    )


def decode_compressed_image(img_msg):
    np_arr = np.frombuffer(img_msg.data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def msg_time_or_now(msg):
    """
    ROS msg header.stamp가 있으면 그 시간을 사용.
    없으면 현재 ROS 시간을 사용.
    """

    try:
        if hasattr(msg, "header"):
            stamp = msg.header.stamp
            if stamp.to_sec() > 0:
                return stamp.to_sec()
    except Exception:
        pass

    return rospy.Time.now().to_sec()


def find_closest(buffer, target_sec, max_gap):
    """
    target_sec 기준으로 가장 가까운 메시지를 찾는다.
    max_gap보다 시간 차이가 크면 None 반환.
    """

    if not buffer:
        return None

    best_ts, best_msg = min(buffer, key=lambda item: abs(item[0] - target_sec))
    gap = abs(best_ts - target_sec)

    if gap > max_gap:
        return None

    return best_msg


def find_closest_with_ts(buffer, target_sec, max_gap):
    """
    target_sec 기준으로 가장 가까운 메시지와 timestamp를 함께 반환.
    """

    if not buffer:
        return None, None

    best_ts, best_msg = min(buffer, key=lambda item: abs(item[0] - target_sec))
    gap = abs(best_ts - target_sec)

    if gap > max_gap:
        return None, None

    return best_ts, best_msg


class MoraiLive3Cam3DLabelerCSV:
    def __init__(self, dataset_dir, scenario_name, max_sync_gap=0.10, save_images=True, save_lidar=True):
        self.dataset_dir = dataset_dir
        self.scenario_name = scenario_name
        self.max_sync_gap = max_sync_gap
        self.save_images = save_images
        self.save_lidar = save_lidar

        self.img_root_dir = os.path.join(dataset_dir, "images")
        self.csv_dir = os.path.join(dataset_dir, "labels_3d")
        self.ego_pose_dir = os.path.join(dataset_dir, "ego_pose")
        self.lidar_dir = os.path.join(dataset_dir, "lidar")

        os.makedirs(self.img_root_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.ego_pose_dir, exist_ok=True)
        os.makedirs(self.lidar_dir, exist_ok=True)

        self.camera_image_dirs = {}
        for topic, cam_name in CAMERA_TOPICS.items():
            cam_dir = os.path.join(self.img_root_dir, cam_name)
            os.makedirs(cam_dir, exist_ok=True)
            self.camera_image_dirs[topic] = cam_dir

        # ------------------------------------------------------------------
        # source-header 기반 epoch/glitch-safe 동기화기 (morai_sync).
        # 기존 receive-time 버퍼/최근접 매칭을 대체한다. 좌표변환/라벨/오프셋은 불변.
        #   - Ego/Object/측면카메라/LiDAR를 cam_front source-header 최근접으로 선택
        #   - Ego/Object 50ms, 측면 100ms, LiDAR 100ms 게이트
        #   - monotonic ref + epoch jump(0.5s) 감지 + buffer reset + hold-3
        #   - source header 없음/0 -> receive fallback 금지, drop (bad label보다 drop 우선)
        # ------------------------------------------------------------------
        # mode="receive"(전 토픽 recv축 최근접)는 scen17 오염 사고로 금지:
        # 카메라 구독 백로그(+66~76s, sync_log 실측)까지 recv 매칭이 무시해
        # 이미지·GT가 서로 다른 심시각으로 짝지어졌다.
        # [2026-07-19 갱신] 26.R1 header 시간축 3분화(카메라=클럭 A / Ego·Object=
        # 클럭 B / LiDAR=A 계열이나 별도 오프셋·slope, scen18·19 라이브 실측)
        # → epoch 모드는 morai_sync의 혼합 매칭 사용: 카메라=src(클럭 A) 게이트,
        # Ego/Object/LiDAR=recv축 게이트(RECV_TOL). 전체 receive 전환 절대 금지.
        self.sync = EpochGlitchSynchronizer()  # mode="epoch" (혼합 매칭)
        # 콜백 스레드 간 synchronizer/sync_log 접근 직렬화용 락 + 종료 플래그.
        self._lock = threading.RLock()
        self._shutdown = False
        self._lidar_gaps = []
        self._lidar_miss = 0
        self._drop_count = 0
        self._drop_reasons = {}

        self.overlay_dir = os.path.join(dataset_dir, "_geom_debug_overlay")
        os.makedirs(self.overlay_dir, exist_ok=True)
        self._overlay_saved_count = 0

        # 프레임별 sync sidecar 로그 (scenario 단위 append).
        self.sync_log_path = os.path.join(dataset_dir, "sync_log.csv")
        self._sync_log_f = open(self.sync_log_path, "w", newline="", encoding="utf-8")
        self._sync_log_w = csv.writer(self._sync_log_f)
        self._sync_log_w.writerow(SIDECAR_FIELDS)
        self._sync_log_f.flush()

        # 종료 시 안전하게 unregister하기 위해 subscriber 핸들을 보관.
        self._subs = []

        # buff_size 명시: rospy 기본 buff_size(64KiB)는 고레이트/대형 메시지에서
        # 소켓 백로그를 만들어 콜백이 과거 메시지를 받게 한다(scen17 카메라 실측
        # +66~76s; 적용 후 scen18 시작 lag 0.239s로 개선). queue_size×메시지
        # 크기보다 크게 잡는다. Ego/Object는 recv축 매칭이므로 백로그가 곧 GT
        # 오염이 된다 — 특히 중요.
        self._subs.append(rospy.Subscriber(
            "/Ego_topic",
            EgoVehicleStatus,
            self.ego_callback,
            queue_size=50,
            buff_size=2**22
        ))

        self._subs.append(rospy.Subscriber(
            "/Object_topic",
            ObjectStatusList,
            self.object_callback,
            queue_size=50,
            buff_size=2**24
        ))

        for topic in CAMERA_TOPICS.keys():
            self._subs.append(rospy.Subscriber(
                topic,
                CompressedImage,
                self.camera_callback,
                callback_args=topic,
                queue_size=10,
                buff_size=2**24
            ))

        # PointCloud2 ~460KB/msg: 기본 buff_size로는 상시 지연(scen17 실측 카메라 대비
        # -1.9s). 투영/기하 경로는 불변 — 구독 버퍼만 확대.
        self._subs.append(rospy.Subscriber(
            LIDAR_TOPIC,
            PointCloud2,
            self.lidar_callback,
            queue_size=2,
            buff_size=2**26
        ))

        rospy.on_shutdown(self._on_shutdown)

        rospy.loginfo("MORAI live 3-camera 3D CSV labeler started")
        rospy.loginfo("scenario       = %s", self.scenario_name)
        rospy.loginfo("target classes = vehicle + pedestrian")
        rospy.loginfo("dataset_dir    = %s", self.dataset_dir)
        rospy.loginfo("max_sync_gap   = %.3f sec", self.max_sync_gap)
        rospy.loginfo("save_images    = %s", self.save_images)

        rospy.loginfo("camera topics:")
        for topic, cam_name in CAMERA_TOPICS.items():
            rospy.loginfo("  %s -> %s", topic, cam_name)

        rospy.loginfo(
            "ROI: x %.1f~%.1f m, y -%.1f~%.1f m, z ±%.1f m",
            FRONT_RANGE_MIN,
            FRONT_RANGE_MAX,
            LATERAL_RANGE_RIGHT,
            LATERAL_RANGE_LEFT,
            MAX_RANGE_Z
        )

    @staticmethod
    def _src_ts(msg):
        """source-header timestamp(초). 없거나 0이면 None (receive fallback 금지)."""
        try:
            if hasattr(msg, "header"):
                s = msg.header.stamp.to_sec()
                if s > 0.0:
                    return s
        except Exception:
            pass
        return None

    # rospy는 각 subscriber 콜백을 별도 스레드로 디스패치한다. synchronizer 상태
    # (frame_idx/pending/last_saved_ref/epoch/버퍼)와 sync_log 기록은 공유 자원이므로
    # push+저장을 콜백당 락으로 직렬화한다. 직렬화하지 않으면 라이브에서 결정 순서가
    # 뒤섞여 sync_log 역행/frame_id 비연속이 발생한다(오프라인 단일스레드에선 안 보임).
    def ego_callback(self, msg):
        recv = rospy.Time.now().to_sec()
        src = self._src_ts(msg)
        with self._lock:
            if self._shutdown:
                return
            self._handle(self.sync.push(EGO_TOPIC, src, recv, msg))

    def object_callback(self, msg):
        recv = rospy.Time.now().to_sec()
        src = self._src_ts(msg)
        with self._lock:
            if self._shutdown:
                return
            self._handle(self.sync.push(OBJ_TOPIC, src, recv, msg))

    def camera_callback(self, img_msg, cam_topic):
        recv = rospy.Time.now().to_sec()
        src = self._src_ts(img_msg)
        if cam_topic == REFERENCE_CAMERA_TOPIC:
            rospy.loginfo_throttle(
                5.0,
                "cam_front recv_ts=%.3f src_header_ts=%s (header lag %s)",
                recv,
                "%.3f" % src if src is not None else "NONE",
                "%.3f" % (recv - src) if src is not None else "n/a",
            )
            # 구독 백로그 조기경보: recv-src가 초 단위로 벌어지면 배달 지연이
            # 쌓이는 중(scen17 사고 서명: 시작부터 +66s). 즉시 수집을 중단하고
            # 노드 재시작 + 네트워크/부하 점검할 것.
            if src is not None and (recv - src) > 2.0:
                rospy.logerr_throttle(
                    5.0,
                    "[BACKLOG] cam_front 배달지연 %.1fs > 2s — 구독 백로그 의심. "
                    "이 상태의 수집물은 GT 오염(scen17 사고). 수집 중단 권장.",
                    recv - src,
                )
        # 모든 카메라를 push. 결정은 cam_front(ref)가 settle된 뒤 나온다.
        with self._lock:
            if self._shutdown:
                return
            self._handle(self.sync.push(cam_topic, src, recv, img_msg))

    def lidar_callback(self, msg):
        # LiDAR도 source-header nearest로 선택된다(latest scan 아님). 버퍼에만 push.
        recv = rospy.Time.now().to_sec()
        src = self._src_ts(msg)
        with self._lock:
            if self._shutdown:
                return
            self._handle(self.sync.push(SYNC_LIDAR_TOPIC, src, recv, msg))

    def _handle(self, decisions):
        """synchronizer가 반환한 Decision list 처리: save는 저장, drop은 로깅."""
        for dec in decisions:
            if dec.action == "save":
                self._save_frame(dec)
            elif dec.action == "drop":
                if dec.reason == "dup_ref":
                    continue
                self._log_drop(dec)

    def _log_drop(self, dec):
        self._drop_count += 1
        key = (dec.reason or "").split(":")[0]
        self._drop_reasons[key] = self._drop_reasons.get(key, 0) + 1
        self._sync_log_w.writerow(sidecar_row("", "drop", dec.sidecar))
        self._sync_log_f.flush()
        rospy.logwarn_throttle(
            1.0,
            "frame drop: reason=%s epoch=%s buffer_reset=%s (total drops=%d)",
            dec.reason, dec.epoch_id, dec.buffer_reset, self._drop_count,
        )

    def _save_frame(self, dec):
        stem = f"live_{dec.frame_id:06d}"
        # epoch 모드: ref_src(=cam_front source header)가 촬영시각. recv는 배달시각이라
        # 백로그 시 수십 초 늦다(scen17 실측 +74s). save 결정은 항상 ref_src를 가진다.
        timestamp = dec.ref_src if dec.ref_src is not None else dec.ref_recv

        sel = dec.selected
        ego_msg = sel[EGO_TOPIC][2]
        obj_msg = sel[OBJ_TOPIC][2]
        lidar_msg = sel[SYNC_LIDAR_TOPIC][2]
        synced_camera_msgs = {
            REFERENCE_CAMERA_TOPIC: dec.ref_msg,
            LEFT_TOPIC: sel[LEFT_TOPIC][2],
            RIGHT_TOPIC: sel[RIGHT_TOPIC][2],
        }

        if self.save_images:
            for topic, img_msg in synced_camera_msgs.items():
                self.save_compressed_image(img_msg=img_msg, topic=topic, stem=stem)

        # LiDAR는 recv축 nearest로 게이트 통과한 스캔(클럭 A 계열이나 별도 오프셋/
        # slope라 src 비교 불가). 매칭 축과 동일한 recv gap을 통계로 기록.
        if self.save_lidar:
            self._lidar_gaps.append(abs(sel[SYNC_LIDAR_TOPIC][1] - dec.ref_recv))
            self.save_lidar_pointcloud(stem=stem, msg=lidar_msg)

        rows = self.make_label_rows(
            frame_id=dec.frame_id,
            timestamp=timestamp,
            ego_msg=ego_msg,
            obj_msg=obj_msg,
        )

        if (
            GEOMETRY_OVERLAY_MAX_FRAMES > 0
            and self._overlay_saved_count < GEOMETRY_OVERLAY_MAX_FRAMES
            and rows
        ):
            self.save_geometry_overlay(
                stem=stem,
                cam_front_msg=dec.ref_msg,
                rows=rows,
            )

        self.save_ego_pose(
            stem=stem,
            frame_id=dec.frame_id,
            timestamp=timestamp,
            ego_msg=ego_msg,
        )

        frame_csv_path = os.path.join(self.csv_dir, f"{stem}.csv")
        with open(frame_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)

        # 프레임별 sync sidecar 기록.
        self._sync_log_w.writerow(sidecar_row(stem, "save", dec.sidecar))
        self._sync_log_f.flush()

        if dec.frame_id % 30 == 0:
            num_vehicle = sum(1 for r in rows if r[6] == "vehicle")
            num_ped = sum(1 for r in rows if r[6] == "pedestrian")
            rospy.loginfo(
                "scenario %s | frame %06d | epoch %s | total: %d | vehicle: %d | pedestrian: %d | drops: %d",
                self.scenario_name, dec.frame_id, dec.epoch_id,
                len(rows), num_vehicle, num_ped, self._drop_count,
            )

    def save_compressed_image(self, img_msg, topic, stem):
        img = decode_compressed_image(img_msg)

        if img is None:
            rospy.logwarn("failed to decode compressed image: %s", topic)
            return

        img_path = os.path.join(
            self.camera_image_dirs[topic],
            f"{stem}.jpg"
        )

        cv2.imwrite(img_path, img)

    def save_geometry_overlay(self, stem, cam_front_msg, rows):
        """
        보정된 3D 박스 코너를 cam_front 원본(1600x900) 이미지에 그려서
        _geom_debug_overlay/에 저장한다. 육안으로 박스가 차체에 맞는지,
        길이축이 진행방향과 나란한지 확인하기 위한 임시 검증용.
        """
        img = decode_compressed_image(cam_front_msg)
        if img is None:
            return

        k_raw = INTRINSICS["cam_front"].astype(np.float64)
        drawn = 0

        for row in rows:
            class_name = row[6]
            box_vals = [
                float(row[7]), float(row[8]), float(row[9]),
                float(row[13]), float(row[14]), float(row[15]),
                float(row[16]), float(row[17]),
                float(row[18]), float(row[19]), float(row[20]),
            ]
            corners = _box_corners_ego(box_vals)
            corners_h = np.concatenate(
                [corners.astype(np.float64), np.ones((corners.shape[0], 1))], axis=1
            )
            u, v, depth, valid = project_body(corners_h, "cam_front", k_raw)
            inframe = valid & (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
            if not inframe.any():
                continue

            color = (0, 255, 0) if class_name == "vehicle" else (0, 165, 255)
            for a, b in _BOX_EDGES:
                if not (valid[a] and valid[b]):
                    continue
                pa = (int(round(u[a])), int(round(v[a])))
                pb = (int(round(u[b])), int(round(v[b])))
                cv2.line(img, pa, pb, color, 2, lineType=cv2.LINE_AA)
            drawn += 1

        if drawn == 0:
            return

        out_path = os.path.join(self.overlay_dir, f"{stem}.jpg")
        cv2.imwrite(out_path, img)
        self._overlay_saved_count += 1
        rospy.loginfo(
            "geom_overlay saved (%d/%d): %s (%d boxes drawn)",
            self._overlay_saved_count, GEOMETRY_OVERLAY_MAX_FRAMES, out_path, drawn
        )

    def _on_shutdown(self):
        # 1) 새 콜백 유입 차단(subscriber unregister). 2) 락 안에서 shutdown 플래그 세팅 →
        #    이 시점 이후 in-flight 콜백은 즉시 반환. 락 획득 자체가 진행 중이던 _save_frame
        #    (파일+sidecar row+frame_idx 증가)이 끝날 때까지 대기하므로 마지막 프레임의
        #    sidecar 누락(파일 306 vs row 305 같은 경계 불일치)을 방지한다.
        for s in getattr(self, "_subs", []):
            try:
                s.unregister()
            except Exception:
                pass

        with self._lock:
            self._shutdown = True
            # 종료 시 settle 대기 중이던 ref를 모두 결정(flush).
            try:
                self._handle(self.sync.flush())
            except Exception as e:
                rospy.logwarn("sync flush on shutdown failed: %s", e)
            self._finalize_locked()

    def _finalize_locked(self):
        if self._lidar_gaps:
            avg_gap = sum(self._lidar_gaps) / len(self._lidar_gaps)
            max_gap = max(self._lidar_gaps)
            rospy.loginfo(
                "lidar recv-gap stats: n=%d avg=%.3f s max=%.3f s (recv축 nearest, gated<=RECV_TOL)",
                len(self._lidar_gaps),
                avg_gap,
                max_gap,
            )

        rospy.loginfo(
            "sync summary: saved=%d drops=%d drop_reasons=%s | sidecar=%s",
            self.sync.frame_idx, self._drop_count, self._drop_reasons, self.sync_log_path,
        )

        try:
            self._sync_log_f.close()
        except Exception:
            pass

    def save_lidar_pointcloud(self, stem, msg):
        field_names = [f.name for f in msg.fields]
        fields = ("x", "y", "z", "intensity") if "intensity" in field_names else ("x", "y", "z")
        pts = list(point_cloud2.read_points(msg, field_names=fields, skip_nans=True))
        if len(pts) == 0:
            rospy.logwarn_throttle(1.0, "lidar scan empty: stem=%s", stem)
            return
        arr = np.asarray(pts, dtype=np.float32)
        np.save(os.path.join(self.lidar_dir, f"{stem}.npy"), arr)

    def save_ego_pose(self, stem, frame_id, timestamp, ego_msg):
        ego_heading_deg = float(ego_msg.heading)
        ego_yaw_rad = float(np.radians(ego_heading_deg))
        pose_path = os.path.join(self.ego_pose_dir, f"{stem}.csv")
        with open(pose_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(EGO_POSE_HEADER)
            writer.writerow([
                frame_id,
                f"{timestamp:.6f}",
                f"{float(ego_msg.position.x):.6f}",
                f"{float(ego_msg.position.y):.6f}",
                f"{float(ego_msg.position.z):.6f}",
                f"{ego_heading_deg:.6f}",
                f"{ego_yaw_rad:.9f}",
            ])

    def make_label_rows(self, frame_id, timestamp, ego_msg, obj_msg):
        ego_pos = [
            ego_msg.position.x,
            ego_msg.position.y,
            ego_msg.position.z
        ]

        ego_heading = ego_msg.heading

        rows = []

        # =========================
        # 1. 차량 NPC 처리
        # =========================
        if hasattr(obj_msg, "npc_list"):
            for object_index, obj in enumerate(obj_msg.npc_list):
                if obj.type not in VALID_NPC_TYPES:
                    continue

                row = self.object_to_csv_row(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    object_source="npc_list",
                    object_index=object_index,
                    obj=obj,
                    ego_pos=ego_pos,
                    ego_heading=ego_heading,
                    class_id=VEHICLE_CLASS_ID,
                    offset_dist=VEHICLE_REAR_AXLE_TO_CENTER
                )

                if row is not None:
                    rows.append(row)

        # =========================
        # 2. 보행자 처리
        # =========================
        if hasattr(obj_msg, "pedestrian_list"):
            for object_index, obj in enumerate(obj_msg.pedestrian_list):
                if VALID_PEDESTRIAN_TYPES is not None:
                    if obj.type not in VALID_PEDESTRIAN_TYPES:
                        continue

                row = self.object_to_csv_row(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    object_source="pedestrian_list",
                    object_index=object_index,
                    obj=obj,
                    ego_pos=ego_pos,
                    ego_heading=ego_heading,
                    class_id=PEDESTRIAN_CLASS_ID,
                    offset_dist=PEDESTRIAN_OFFSET_DIST
                )

                if row is not None:
                    rows.append(row)

        return rows

    def object_to_csv_row(self, frame_id, timestamp, object_source,
                          object_index, obj, ego_pos, ego_heading,
                          class_id, offset_dist):
        obj_pos = [
            obj.position.x,
            obj.position.y,
            obj.position.z
        ]

        obj_vel = [
            obj.velocity.x,
            obj.velocity.y,
            obj.velocity.z
        ]

        obj_size = [
            obj.size.x,
            obj.size.y,
            obj.size.z
        ]

        # MORAI obj.size: x=길이(진행방향), y=폭, z=높이
        l = float(obj_size[0])
        w = float(obj_size[1])
        h = float(obj_size[2])

        pos_ego, rel_yaw, vel_ego = world_to_ego(
            obj_pos=obj_pos,
            obj_heading_deg=obj.heading,
            obj_velocity=obj_vel,
            ego_pos=ego_pos,
            ego_heading_deg=ego_heading,
            offset_dist=offset_dist
        )

        # =========================
        # Ego 기준 관심 영역 필터
        # =========================
        if (
            pos_ego[0] < FRONT_RANGE_MIN or
            pos_ego[0] > FRONT_RANGE_MAX or
            pos_ego[1] < -LATERAL_RANGE_RIGHT or
            pos_ego[1] > LATERAL_RANGE_LEFT or
            abs(pos_ego[2]) > MAX_RANGE_Z
        ):
            return None

        sin_yaw = float(np.sin(rel_yaw))
        cos_yaw = float(np.cos(rel_yaw))

        ln_w = float(np.log(max(w, 0.01)))
        ln_l = float(np.log(max(l, 0.01)))
        ln_h = float(np.log(max(h, 0.01)))

        box_for_visibility = [
            float(pos_ego[0]), float(pos_ego[1]), float(pos_ego[2]),
            ln_w, ln_l, ln_h,
            sin_yaw, cos_yaw,
            float(vel_ego[0]), float(vel_ego[1]), float(vel_ego[2]),
        ]

        self.log_vehicle_geometry_debug(
            object_source=object_source,
            object_index=object_index,
            class_name=CLASS_NAMES[class_id],
            obj_pos=obj_pos,
            obj_heading_deg=obj.heading,
            ego_pos=ego_pos,
            ego_heading_deg=ego_heading,
            offset_dist=offset_dist,
            pos_ego=pos_ego,
            rel_yaw=rel_yaw,
            h=h,
            box_vals=box_for_visibility,
        )

        if not box_visible_in_any_camera(box_for_visibility):
            return None

        class_name = CLASS_NAMES[class_id]

        row = [
            frame_id,
            f"{timestamp:.6f}",
            object_source,
            object_index,
            obj.type,
            class_id,
            class_name,
            f"{float(pos_ego[0]):.4f}",
            f"{float(pos_ego[1]):.4f}",
            f"{float(pos_ego[2]):.4f}",
            f"{w:.4f}",
            f"{l:.4f}",
            f"{h:.4f}",
            f"{ln_w:.4f}",
            f"{ln_l:.4f}",
            f"{ln_h:.4f}",
            f"{sin_yaw:.4f}",
            f"{cos_yaw:.4f}",
            f"{float(vel_ego[0]):.4f}",
            f"{float(vel_ego[1]):.4f}",
            f"{float(vel_ego[2]):.4f}",
        ]

        return row

    def log_vehicle_geometry_debug(self, object_source, object_index, class_name,
                                    obj_pos, obj_heading_deg, ego_pos, ego_heading_deg,
                                    offset_dist, pos_ego, rel_yaw, h, box_vals):
        """
        기준점 보정이 의도대로 됐는지 확인하는 디버그 로그 (STRICT_ASSERTS=False면 로그만).

        - offset_axis_err : 기준점 -> 박스중심 이동 벡터가 차량 heading 전방과
                            나란한지(부호 반전 등 회귀를 잡아내는 용도).
        - length_axis_err : _box_corners_ego로 실제 생성한 박스 코너의 앞/뒤면
                            중심을 잇는 축이 객체 heading 방향과 나란한지.
        - cam_front_proj  : 보정 전/후 박스 중심을 cam_front에 투영한 픽셀 좌표.
        """
        tag = f"{object_source}#{object_index}({class_name})"

        if offset_dist == 0.0:
            return

        heading_ego = np.radians(obj_heading_deg - ego_heading_deg)

        pos_ego_raw, _, _ = world_to_ego(
            obj_pos=obj_pos,
            obj_heading_deg=obj_heading_deg,
            obj_velocity=[0.0, 0.0, 0.0],
            ego_pos=ego_pos,
            ego_heading_deg=ego_heading_deg,
            offset_dist=0.0,
        )

        offset_vec = pos_ego[:2] - pos_ego_raw[:2]
        offset_dir = float(np.arctan2(offset_vec[1], offset_vec[0]))
        offset_axis_err_deg = float(np.degrees(np.arctan2(
            np.sin(offset_dir - heading_ego), np.cos(offset_dir - heading_ego)
        )))

        corners = _box_corners_ego(box_vals)  # (9,3): 0~7 코너, 8 center
        front_mid = corners[0:4].mean(axis=0)
        back_mid = corners[4:8].mean(axis=0)
        length_axis = front_mid[:2] - back_mid[:2]
        length_axis_angle = float(np.arctan2(length_axis[1], length_axis[0]))
        length_axis_err_deg = float(np.degrees(np.arctan2(
            np.sin(length_axis_angle - heading_ego), np.cos(length_axis_angle - heading_ego)
        )))

        def _project_center_to_cam_front(pos_ego_pt):
            center = np.array(
                [pos_ego_pt[0], pos_ego_pt[1], pos_ego_pt[2] + h * 0.5, 1.0],
                dtype=np.float64
            )
            u, v, depth, valid = project_body(center[np.newaxis, :], "cam_front", INTRINSICS["cam_front"])
            if not valid[0]:
                return "behind_cam"
            return f"({u[0]:.0f},{v[0]:.0f},d={depth[0]:.1f}m)"

        proj_pre = _project_center_to_cam_front(pos_ego_raw)
        proj_post = _project_center_to_cam_front(pos_ego)

        is_bad = (
            abs(offset_axis_err_deg) > OFFSET_AXIS_ERR_THRESH_DEG or
            abs(length_axis_err_deg) > LENGTH_AXIS_ERR_THRESH_DEG
        )
        log_fn = rospy.logwarn_throttle if is_bad else rospy.loginfo_throttle

        log_fn(
            2.0,
            "geom_debug %s offset_dist=%.3f heading_ego=%+.2fdeg rel_yaw=%+.2fdeg "
            "offset_axis_err=%+.2fdeg length_axis_err=%+.2fdeg "
            "cam_front_proj pre=%s post=%s",
            tag, offset_dist, np.degrees(heading_ego), np.degrees(rel_yaw),
            offset_axis_err_deg, length_axis_err_deg, proj_pre, proj_post
        )

        if STRICT_ASSERTS:
            assert abs(offset_axis_err_deg) <= OFFSET_AXIS_ERR_THRESH_DEG, (
                f"{tag} offset_axis_err={offset_axis_err_deg:.2f}deg exceeds threshold"
            )
            assert abs(length_axis_err_deg) <= LENGTH_AXIS_ERR_THRESH_DEG, (
                f"{tag} length_axis_err={length_axis_err_deg:.2f}deg exceeds threshold"
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_root",
        default="/data/dataset",
        help="scen01, scen02가 생성될 상위 dataset 폴더"
    )

    parser.add_argument(
        "--scenario_prefix",
        default="scen",
        help="시나리오 폴더 prefix"
    )

    parser.add_argument(
        "--scenario_digits",
        type=int,
        default=2,
        help="scen 번호 자릿수. 2이면 scen01, 3이면 scen001"
    )

    parser.add_argument(
        "--max_sync_gap",
        type=float,
        default=0.10,
        help="카메라-Ego/Object 동기화 허용 시간 차이"
    )

    parser.add_argument(
        "--no_save_images",
        action="store_true",
        help="이미지 저장을 끄고 CSV만 저장"
    )

    parser.add_argument(
        "--no_save_lidar",
        action="store_true",
        help="라이다 포인트클라우드 저장을 끈다 (기본은 저장 on)"
    )

    args, _ = parser.parse_known_args()

    rospy.init_node("morai_live_3cam_3d_labeler_csv", anonymous=False)

    scenario_dir, scenario_name = create_next_scenario_dir(
        dataset_root=args.dataset_root,
        prefix=args.scenario_prefix,
        digits=args.scenario_digits
    )

    rospy.loginfo("new scenario folder = %s", scenario_dir)

    MoraiLive3Cam3DLabelerCSV(
        dataset_dir=scenario_dir,
        scenario_name=scenario_name,
        max_sync_gap=args.max_sync_gap,
        save_images=not args.no_save_images,
        save_lidar=not args.no_save_lidar
    )

    rospy.spin()


if __name__ == "__main__":
    main()
