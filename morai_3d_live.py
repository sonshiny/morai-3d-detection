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

from sensor_msgs.msg import CompressedImage
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList


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
    이미지 timestamp 기준으로 가장 가까운 Ego/Object 메시지를 찾는다.
    max_gap보다 시간 차이가 크면 동기화 실패로 처리한다.
    """

    if not buffer:
        return None

    best_ts, best_msg = min(buffer, key=lambda item: abs(item[0] - target_sec))
    gap = abs(best_ts - target_sec)

    if gap > max_gap:
        return None

    return best_msg


class MoraiLive3DLabelerCSV:
    def __init__(self, dataset_dir, scenario_name, max_sync_gap=0.10, save_images=False):
        self.dataset_dir = dataset_dir
        self.scenario_name = scenario_name
        self.max_sync_gap = max_sync_gap
        self.save_images = save_images

        self.img_dir = os.path.join(dataset_dir, "images")
        self.csv_dir = os.path.join(dataset_dir, "labels_3d")

        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)

        self.ego_buffer = deque(maxlen=200)
        self.obj_buffer = deque(maxlen=200)

        self.frame_idx = 0

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

        rospy.Subscriber(
            "/image_jpeg/compressed",
            CompressedImage,
            self.image_callback,
            queue_size=10
        )

        rospy.loginfo("MORAI live 3D CSV labeler started")
        rospy.loginfo("scenario       = %s", self.scenario_name)
        rospy.loginfo("target classes = vehicle + pedestrian")
        rospy.loginfo("dataset_dir    = %s", self.dataset_dir)
        rospy.loginfo("max_sync_gap   = %.3f sec", self.max_sync_gap)
        rospy.loginfo("save_images    = %s", self.save_images)
        rospy.loginfo(
            "ROI: x %.1f~%.1f m, y -%.1f~%.1f m, z ±%.1f m",
            FRONT_RANGE_MIN,
            FRONT_RANGE_MAX,
            LATERAL_RANGE_RIGHT,
            LATERAL_RANGE_LEFT,
            MAX_RANGE_Z
        )

    def ego_callback(self, msg):
        ts = msg_time_or_now(msg)
        self.ego_buffer.append((ts, msg))

    def object_callback(self, msg):
        ts = msg_time_or_now(msg)
        self.obj_buffer.append((ts, msg))

    def image_callback(self, img_msg):
        img_ts = msg_time_or_now(img_msg)

        ego_msg = find_closest(
            self.ego_buffer,
            img_ts,
            self.max_sync_gap
        )

        obj_msg = find_closest(
            self.obj_buffer,
            img_ts,
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
            self.save_compressed_image(img_msg, stem)

        rows = self.make_label_rows(
            frame_id=self.frame_idx,
            timestamp=img_ts,
            ego_msg=ego_msg,
            obj_msg=obj_msg
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
                "scenario %s | frame %06d | total: %d | vehicle: %d | pedestrian: %d",
                self.scenario_name,
                self.frame_idx,
                len(rows),
                num_vehicle,
                num_ped
            )

        self.frame_idx += 1

    def save_compressed_image(self, img_msg, stem):
        np_arr = np.frombuffer(img_msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            rospy.logwarn("failed to decode compressed image")
            return

        img_path = os.path.join(self.img_dir, f"{stem}.jpg")
        cv2.imwrite(img_path, img)

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
        # x < 0이면 Ego 뒤쪽 객체이므로 저장하지 않음
        # x: 0~60m 전방
        # y: 좌우 -30~30m
        # z: ±3m
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
        default="./dataset",
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
        help="이미지-Ego/Object 동기화 허용 시간 차이"
    )

    parser.add_argument(
        "--save_images",
        action="store_true",
        help="compressed image도 jpg로 저장"
    )

    args, _ = parser.parse_known_args()

    rospy.init_node("morai_live_3d_labeler_csv", anonymous=False)

    scenario_dir, scenario_name = create_next_scenario_dir(
        dataset_root=args.dataset_root,
        prefix=args.scenario_prefix,
        digits=args.scenario_digits
    )

    rospy.loginfo("new scenario folder = %s", scenario_dir)

    MoraiLive3DLabelerCSV(
        dataset_dir=scenario_dir,
        scenario_name=scenario_name,
        max_sync_gap=args.max_sync_gap,
        save_images=args.save_images
    )

    rospy.spin()


if __name__ == "__main__":
    main()