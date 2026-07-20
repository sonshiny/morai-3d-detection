#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time

import cv2
import numpy as np
import rospy

from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList
from sensor_msgs.msg import Image, CompressedImage


FRONT_RANGE_MIN = 0.0
FRONT_RANGE_MAX = 60.0
LATERAL_RANGE_LEFT = 30.0
LATERAL_RANGE_RIGHT = 30.0
MAX_RANGE_Z = 3.0

VEHICLE_CLASS_ID = 0
PEDESTRIAN_CLASS_ID = 1

CLASS_NAMES = {
    VEHICLE_CLASS_ID: "vehicle",
    PEDESTRIAN_CLASS_ID: "pedestrian",
}

VALID_NPC_TYPES = {1}
VALID_PEDESTRIAN_TYPES = None

VEHICLE_REAR_AXLE_TO_CENTER = 1.28
PEDESTRIAN_OFFSET_DIST = 0.0


def wrap_pi(rad):
    return float(math.atan2(math.sin(rad), math.cos(rad)))


def world_to_ego_point(point_w, ego_pos, ego_heading_deg):
    dx = float(point_w[0]) - float(ego_pos[0])
    dy = float(point_w[1]) - float(ego_pos[1])
    dz = float(point_w[2]) - float(ego_pos[2])
    yaw = math.radians(float(ego_heading_deg))
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([
        c * dx + s * dy,
        -s * dx + c * dy,
        dz,
    ], dtype=np.float32)


def object_to_gt_box(obj, ego_pos, ego_heading_deg, class_id):
    obj_pos = np.array([obj.position.x, obj.position.y, obj.position.z], dtype=np.float32)
    obj_heading_deg = float(obj.heading)
    obj_yaw_w = math.radians(obj_heading_deg)

    if class_id == VEHICLE_CLASS_ID:
        # MORAI vehicle size.x behaves as physical length, size.y as width.
        l = float(obj.size.x)
        w = float(obj.size.y)
        h = float(obj.size.z)
        offset_dist = VEHICLE_REAR_AXLE_TO_CENTER
    else:
        w = float(obj.size.x)
        l = float(obj.size.y)
        h = float(obj.size.z)
        offset_dist = PEDESTRIAN_OFFSET_DIST

    center_w = np.array([
        obj_pos[0] + offset_dist * math.cos(obj_yaw_w),
        obj_pos[1] + offset_dist * math.sin(obj_yaw_w),
        obj_pos[2],
    ], dtype=np.float32)

    raw_ego = world_to_ego_point(obj_pos, ego_pos, ego_heading_deg)
    center_ego = world_to_ego_point(center_w, ego_pos, ego_heading_deg)
    yaw_ego = wrap_pi(math.radians(obj_heading_deg - float(ego_heading_deg)))

    return {
        "class_id": int(class_id),
        "class_name": CLASS_NAMES[int(class_id)],
        "type": int(obj.type),
        "raw_ego": raw_ego,
        "center_ego": center_ego,
        "yaw_ego": yaw_ego,
        "w": w,
        "l": l,
        "h": h,
        "offset_dist": offset_dist,
    }


def box_corners_bev(box):
    x, y = float(box["center_ego"][0]), float(box["center_ego"][1])
    l, w = float(box["l"]), float(box["w"])
    yaw = float(box["yaw_ego"])
    c, s = math.cos(yaw), math.sin(yaw)

    local = np.array([
        [ l * 0.5,  w * 0.5],
        [ l * 0.5, -w * 0.5],
        [-l * 0.5, -w * 0.5],
        [-l * 0.5,  w * 0.5],
    ], dtype=np.float32)

    out = []
    for lx, ly in local:
        gx = c * lx - s * ly + x
        gy = s * lx + c * ly + y
        out.append([gx, gy])
    return np.asarray(out, dtype=np.float32)


def in_roi(box):
    x, y, z = box["center_ego"]
    return (
        FRONT_RANGE_MIN <= x <= FRONT_RANGE_MAX and
        -LATERAL_RANGE_RIGHT <= y <= LATERAL_RANGE_LEFT and
        abs(float(z)) <= MAX_RANGE_Z
    )


class MoraiGTBEVViewer:
    def __init__(self):
        self.width = int(rospy.get_param("~width", 900))
        self.height = int(rospy.get_param("~height", 900))
        self.margin = int(rospy.get_param("~margin", 45))
        self.show_out_of_roi = bool(rospy.get_param("~show_out_of_roi", False))
        self.max_hz = float(rospy.get_param("~max_hz", 15.0))

        self.ego_msg = None
        self.last_pub_time = 0.0

        self.pub_img = rospy.Publisher("/morai_gt/bev", Image, queue_size=1)
        self.pub_comp = rospy.Publisher("/morai_gt/bev/compressed", CompressedImage, queue_size=1)

        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, self.ego_callback, queue_size=20)
        rospy.Subscriber("/Object_topic", ObjectStatusList, self.object_callback, queue_size=20)

        rospy.loginfo("MORAI GT BEV viewer started")
        rospy.loginfo("publish: /morai_gt/bev and /morai_gt/bev/compressed")
        rospy.loginfo(
            "GT convention: x=forward, y=left, yaw=obj_heading-ego_heading, "
            "vehicle l=size.x w=size.y offset=%.2fm",
            VEHICLE_REAR_AXLE_TO_CENTER,
        )

    def ego_callback(self, msg):
        self.ego_msg = msg

    def object_callback(self, msg):
        now = time.time()
        if self.max_hz > 0.0 and now - self.last_pub_time < 1.0 / self.max_hz:
            return
        if self.ego_msg is None:
            rospy.logwarn_throttle(1.0, "waiting for /Ego_topic")
            return

        boxes = self.make_boxes(msg)
        if not self.show_out_of_roi:
            boxes = [b for b in boxes if in_roi(b)]

        img = self.render_bev(boxes)
        self.publish_image(img)
        self.last_pub_time = now

    def make_boxes(self, obj_msg):
        ego_pos = [
            self.ego_msg.position.x,
            self.ego_msg.position.y,
            self.ego_msg.position.z,
        ]
        ego_heading = float(self.ego_msg.heading)
        boxes = []

        if hasattr(obj_msg, "npc_list"):
            for obj in obj_msg.npc_list:
                if obj.type not in VALID_NPC_TYPES:
                    continue
                boxes.append(object_to_gt_box(obj, ego_pos, ego_heading, VEHICLE_CLASS_ID))

        if hasattr(obj_msg, "pedestrian_list"):
            for obj in obj_msg.pedestrian_list:
                if VALID_PEDESTRIAN_TYPES is not None and obj.type not in VALID_PEDESTRIAN_TYPES:
                    continue
                boxes.append(object_to_gt_box(obj, ego_pos, ego_heading, PEDESTRIAN_CLASS_ID))

        return boxes

    def meter_to_px(self, x, y):
        usable_w = max(1, self.width - 2 * self.margin)
        usable_h = max(1, self.height - 2 * self.margin)
        scale = min(
            usable_w / float(LATERAL_RANGE_LEFT + LATERAL_RANGE_RIGHT),
            usable_h / float(FRONT_RANGE_MAX - FRONT_RANGE_MIN),
        )
        cx = self.width * 0.5
        bottom = self.height - self.margin
        px = int(round(cx - float(y) * scale))
        py = int(round(bottom - float(x) * scale))
        return px, py

    def render_bev(self, boxes):
        img = np.full((self.height, self.width, 3), (28, 30, 34), dtype=np.uint8)
        self.draw_grid(img)
        self.draw_ego(img)

        for box in boxes:
            color = (80, 220, 120) if box["class_id"] == VEHICLE_CLASS_ID else (80, 180, 255)
            corners = box_corners_bev(box)
            pts = np.array([self.meter_to_px(x, y) for x, y in corners], dtype=np.int32)
            cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

            cx, cy = self.meter_to_px(box["center_ego"][0], box["center_ego"][1])
            rx, ry = self.meter_to_px(box["raw_ego"][0], box["raw_ego"][1])
            cv2.circle(img, (cx, cy), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            if box["offset_dist"] > 0.0:
                cv2.circle(img, (rx, ry), 3, (0, 80, 255), -1, lineType=cv2.LINE_AA)
                cv2.line(img, (rx, ry), (cx, cy), (0, 80, 255), 1, lineType=cv2.LINE_AA)

            front_mid = (pts[0] + pts[1]) // 2
            cv2.line(img, (cx, cy), tuple(front_mid), (255, 255, 255), 1, lineType=cv2.LINE_AA)

            label = (
                f"{box['class_name']} x={box['center_ego'][0]:.1f} y={box['center_ego'][1]:.1f} "
                f"yaw={math.degrees(box['yaw_ego']):.0f}"
            )
            cv2.putText(
                img,
                label,
                (min(self.width - 260, max(4, cx + 6)), max(14, cy - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            img,
            f"GT BEV  boxes={len(boxes)}  green=vehicle orange=ped  white=center red=raw MORAI pos",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        return img

    def draw_grid(self, img):
        grid_color = (62, 66, 72)
        axis_color = (115, 120, 128)
        for x in range(0, int(FRONT_RANGE_MAX) + 1, 10):
            p1 = self.meter_to_px(x, -LATERAL_RANGE_RIGHT)
            p2 = self.meter_to_px(x, LATERAL_RANGE_LEFT)
            cv2.line(img, p1, p2, grid_color, 1, lineType=cv2.LINE_AA)
            cv2.putText(img, f"{x}m", (p1[0] + 4, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, axis_color, 1)

        for y in range(-int(LATERAL_RANGE_RIGHT), int(LATERAL_RANGE_LEFT) + 1, 10):
            p1 = self.meter_to_px(FRONT_RANGE_MIN, y)
            p2 = self.meter_to_px(FRONT_RANGE_MAX, y)
            color = axis_color if y == 0 else grid_color
            cv2.line(img, p1, p2, color, 1, lineType=cv2.LINE_AA)

    def draw_ego(self, img):
        p0 = np.array(self.meter_to_px(0.0, 0.0), dtype=np.int32)
        p1 = np.array(self.meter_to_px(4.2, 0.0), dtype=np.int32)
        left = np.array(self.meter_to_px(0.0, 1.0), dtype=np.int32)
        right = np.array(self.meter_to_px(0.0, -1.0), dtype=np.int32)
        pts = np.array([p1, left, right], dtype=np.int32)
        cv2.fillConvexPoly(img, pts, (95, 145, 255), lineType=cv2.LINE_AA)
        cv2.circle(img, tuple(p0), 3, (255, 255, 255), -1, lineType=cv2.LINE_AA)

    def publish_image(self, img):
        stamp = rospy.Time.now()

        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "ego"
        msg.height = img.shape[0]
        msg.width = img.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = img.shape[1] * 3
        msg.data = img.tobytes()
        self.pub_img.publish(msg)

        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if ok:
            comp = CompressedImage()
            comp.header.stamp = stamp
            comp.header.frame_id = "ego"
            comp.format = "jpeg"
            comp.data = enc.tobytes()
            self.pub_comp.publish(comp)


def main():
    rospy.init_node("morai_gt_bev_viewer", anonymous=False)
    MoraiGTBEVViewer()
    rospy.spin()


if __name__ == "__main__":
    main()
