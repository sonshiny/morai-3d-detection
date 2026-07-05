#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import argparse
from collections import deque

import cv2
import numpy as np
import rospy

from sensor_msgs.msg import CompressedImage, PointCloud2
from sensor_msgs import point_cloud2
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList

from morai_dataset import box_visible_in_any_camera


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


# 차량은 MORAI 기준점이 중심이 아닐 수 있어서 기존처럼 보정
VEHICLE_OFFSET_RATIO = 0.2

# 보행자는 객체 중심 기준으로 사용
PEDESTRIAN_OFFSET_RATIO = 0.0


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
                 ego_pos, ego_heading_deg, obj_length, offset_ratio):
    """
    MORAI world 좌표의 객체를 Ego 차량 기준 좌표로 변환.

    변환 후:
      x_e > 0 : Ego 전방
      x_e < 0 : Ego 후방
      y_e     : Ego 좌우 방향
    """

    offset_dist = obj_length * offset_ratio
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

    rel_yaw_raw = np.radians(obj_heading_deg - ego_heading_deg) - np.pi / 2.0
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

        self.ego_buffer = deque(maxlen=200)
        self.obj_buffer = deque(maxlen=200)

        self.camera_buffers = {
            topic: deque(maxlen=100) for topic in CAMERA_TOPICS.keys()
        }

        self.lidar_buffer = deque(maxlen=20)
        self._lidar_gaps = []
        self._lidar_miss = 0

        self.frame_idx = 0
        self.last_processed_ref_ts = None

        rospy.Subscriber(
            "/Ego_topic",
            EgoVehicleStatus,
            self.ego_callback,
            queue_size=50
        )

        rospy.Subscriber(
            "/Object_topic",
            ObjectStatusList,
            self.object_callback,
            queue_size=50
        )

        for topic in CAMERA_TOPICS.keys():
            rospy.Subscriber(
                topic,
                CompressedImage,
                self.camera_callback,
                callback_args=topic,
                queue_size=10,
                buff_size=2**24
            )

        rospy.Subscriber(
            LIDAR_TOPIC,
            PointCloud2,
            self.lidar_callback,
            queue_size=2
        )

        rospy.on_shutdown(self._dump_lidar_stats)

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

    def ego_callback(self, msg):
        ts = rospy.Time.now().to_sec()   # header.stamp는 카메라와 다른 시계라 신뢰 불가
        self.ego_buffer.append((ts, msg))

    def object_callback(self, msg):
        ts = rospy.Time.now().to_sec()   # header.stamp는 카메라와 다른 시계라 신뢰 불가
        self.obj_buffer.append((ts, msg))

    def camera_callback(self, img_msg, cam_topic):
        ts = msg_time_or_now(img_msg)
        self.camera_buffers[cam_topic].append((ts, img_msg))

        # 진단(임시): 카메라 header.stamp가 rospy.Time.now()와 같은 시계 계열인지 확인용.
        # diff가 작으면(<0.1s) ego/object의 now() 전환과 정합됨. 검증 후 제거 가능.
        rospy.loginfo_throttle(
            5.0,
            "cam ref_ts=%.3f vs now=%.3f (diff %.3f)",
            ts,
            rospy.Time.now().to_sec(),
            rospy.Time.now().to_sec() - ts
        )

        # 카메라 메시지가 들어올 때마다 현재 3개 카메라가 모두 모였는지 확인
        self.try_process_synced_frame()

    def lidar_callback(self, msg):
        # 라이다는 프레임 트리거가 아니다. 버퍼에만 쌓고, 저장은
        # /cam_front가 트리거한 try_process_synced_frame에서 비차단으로 매칭한다.
        ts = msg_time_or_now(msg)
        self.lidar_buffer.append((ts, msg))

    def try_process_synced_frame(self):
        ref_buffer = self.camera_buffers[REFERENCE_CAMERA_TOPIC]

        if not ref_buffer:
            return

        ref_ts, ref_msg = ref_buffer[-1]

        if self.last_processed_ref_ts is not None:
            if abs(ref_ts - self.last_processed_ref_ts) < 1e-6:
                return

        synced_camera_msgs = {}

        for topic in CAMERA_TOPICS.keys():
            cam_ts, cam_msg = find_closest_with_ts(
                self.camera_buffers[topic],
                ref_ts,
                self.max_sync_gap
            )

            if cam_msg is None:
                rospy.logwarn_throttle(
                    1.0,
                    "camera sync fail: topic=%s ref_ts=%.6f",
                    topic,
                    ref_ts
                )
                return

            synced_camera_msgs[topic] = cam_msg

        ego_msg = find_closest(
            self.ego_buffer,
            ref_ts,
            self.max_sync_gap
        )

        obj_msg = find_closest(
            self.obj_buffer,
            ref_ts,
            self.max_sync_gap
        )

        if ego_msg is None or obj_msg is None:
            rospy.logwarn_throttle(
                1.0,
                "sync fail: ego=%s object=%s",
                ego_msg is not None,
                obj_msg is not None
            )
            return

        stem = f"live_{self.frame_idx:06d}"

        if self.save_images:
            for topic, img_msg in synced_camera_msgs.items():
                self.save_compressed_image(
                    img_msg=img_msg,
                    topic=topic,
                    stem=stem
                )

        # 비차단 라이다 매칭+저장. 라이다 실패는 프레임을 버리지 않는다
        # (카메라/ego/object all-or-nothing 게이트는 위에서 이미 통과함).
        # /lidar3D의 header.stamp는 /cam_front ref_ts와 클록 기준이 달라
        # 스탬프 비교(find_closest)로는 항상 max_sync_gap을 초과해 실패한다.
        # 따라서 스탬프 비교 없이 버퍼의 가장 최신 스캔을 그대로 사용한다.
        if self.save_lidar:
            if len(self.lidar_buffer) > 0:
                lidar_ts, lidar_msg = self.lidar_buffer[-1]
                offset = lidar_ts - ref_ts
                self._lidar_gaps.append(abs(offset))
                self.save_lidar_pointcloud(stem=stem, msg=lidar_msg)
                rospy.loginfo_throttle(
                    2.0,
                    "lidar saved (stamp offset %+.3f s, not used for matching)",
                    offset
                )
            else:
                self._lidar_miss += 1
                rospy.logwarn_throttle(
                    1.0,
                    "lidar buffer empty: frame=%06d",
                    self.frame_idx
                )

        rows = self.make_label_rows(
            frame_id=self.frame_idx,
            timestamp=ref_ts,
            ego_msg=ego_msg,
            obj_msg=obj_msg
        )

        self.save_ego_pose(
            stem=stem,
            frame_id=self.frame_idx,
            timestamp=ref_ts,
            ego_msg=ego_msg,
        )

        frame_csv_path = os.path.join(self.csv_dir, f"{stem}.csv")

        with open(frame_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            writer.writerows(rows)

        if self.frame_idx % 30 == 0:
            num_vehicle = sum(1 for r in rows if r[6] == "vehicle")
            num_ped = sum(1 for r in rows if r[6] == "pedestrian")

            rospy.loginfo(
                "scenario %s | frame %06d | images: 3 | total: %d | vehicle: %d | pedestrian: %d",
                self.scenario_name,
                self.frame_idx,
                len(rows),
                num_vehicle,
                num_ped
            )

        self.last_processed_ref_ts = ref_ts
        self.frame_idx += 1

    def save_compressed_image(self, img_msg, topic, stem):
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            rospy.logwarn("failed to decode compressed image: %s", topic)
            return

        img_path = os.path.join(
            self.camera_image_dirs[topic],
            f"{stem}.jpg"
        )

        cv2.imwrite(img_path, img)

    def _dump_lidar_stats(self):
        if not self._lidar_gaps and not self._lidar_miss:
            return

        if self._lidar_gaps:
            avg_gap = sum(self._lidar_gaps) / len(self._lidar_gaps)
            max_gap = max(self._lidar_gaps)
            rospy.loginfo(
                "lidar stamp offset stats: n=%d avg=%.3f s max=%.3f s (clock mismatch, informational only)",
                len(self._lidar_gaps),
                avg_gap,
                max_gap
            )

        if self._lidar_miss:
            rospy.logwarn(
                "lidar buffer was empty for %d frame(s)",
                self._lidar_miss
            )

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
                    offset_ratio=VEHICLE_OFFSET_RATIO
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
                    offset_ratio=PEDESTRIAN_OFFSET_RATIO
                )

                if row is not None:
                    rows.append(row)

        return rows

    def object_to_csv_row(self, frame_id, timestamp, object_source,
                          object_index, obj, ego_pos, ego_heading,
                          class_id, offset_ratio):
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

        w = float(obj_size[0])
        l = float(obj_size[1])
        h = float(obj_size[2])

        obj_length = l

        pos_ego, rel_yaw, vel_ego = world_to_ego(
            obj_pos=obj_pos,
            obj_heading_deg=obj.heading,
            obj_velocity=obj_vel,
            ego_pos=ego_pos,
            ego_heading_deg=ego_heading,
            obj_length=obj_length,
            offset_ratio=offset_ratio
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
