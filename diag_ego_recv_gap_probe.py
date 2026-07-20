#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_ego_recv_gap_probe.py — [사람 실행 필요, MORAI 라이브]

혼합 매칭의 Ego/Object recv축 게이트(RECV_TOL) 확정용 실측.
production(morai_3d_live)과 동일한 구독 설정으로 일정 시간 수신하며,
각 cam_front 도착시각에 대한 Ego/Object recv 최근접 gap 분포를 잰다.
부가로 클럭 A/B 상태(헤더 vs wall 오프셋·드리프트, 카메라 백로그)도 기록.

사용:
  source /opt/ros/noetic/setup.bash && source ~/morai_ws/devel/setup.bash
  python3 diag_ego_recv_gap_probe.py --sec 60
"""

import argparse
import threading

import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage, PointCloud2
from morai_msgs.msg import EgoVehicleStatus, ObjectStatusList


class Probe(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.cam = []    # (recv, src)
        self.ego = []    # (recv, src)
        self.obj = []    # (recv, src)
        self.lidar = []  # (recv, src)

    def _src(self, msg):
        try:
            s = msg.header.stamp.to_sec()
            return s if s > 0 else None
        except Exception:
            return None

    def cb(self, store):
        def f(msg):
            now = rospy.Time.now().to_sec()
            with self.lock:
                store.append((now, self._src(msg)))
        return f


def stats(name, cam_recv, other_recv):
    if not other_recv or not cam_recv:
        print(f"  {name}: 데이터 없음")
        return None
    o = np.asarray(other_recv)
    gaps = []
    for t in cam_recv:
        gaps.append(np.abs(o - t).min())
    g = np.asarray(gaps) * 1000.0  # ms
    p50, p90, p99, mx = (np.percentile(g, 50), np.percentile(g, 90),
                         np.percentile(g, 99), g.max())
    print(f"  {name}: n_cam={len(cam_recv)} n_{name}={len(o)}  "
          f"gap p50={p50:.1f} p90={p90:.1f} p99={p99:.1f} max={mx:.1f} ms")
    return mx


def clockline(name, arr):
    a = [(r, s) for (r, s) in arr if s is not None]
    if len(a) < 2:
        print(f"  {name}: header 없음/부족")
        return
    (r0, s0), (r1, s1) = a[0], a[-1]
    off0, off1 = s0 - r0, s1 - r1
    slope = (s1 - s0) / max(r1 - r0, 1e-9)
    print(f"  {name}: header-wall 오프셋 {off0:+.3f}s -> {off1:+.3f}s "
          f"(드리프트 {off1 - off0:+.3f}s / {r1 - r0:.1f}s, slope={slope:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", type=float, default=60.0)
    args = ap.parse_args()

    rospy.init_node("diag_ego_recv_gap_probe", anonymous=True)
    p = Probe()
    subs = [
        rospy.Subscriber("/cam_front", CompressedImage, p.cb(p.cam),
                         queue_size=10, buff_size=2**24),
        rospy.Subscriber("/Ego_topic", EgoVehicleStatus, p.cb(p.ego),
                         queue_size=50, buff_size=2**22),
        rospy.Subscriber("/Object_topic", ObjectStatusList, p.cb(p.obj),
                         queue_size=50, buff_size=2**24),
        rospy.Subscriber("/lidar3D", PointCloud2, p.cb(p.lidar),
                         queue_size=2, buff_size=2**26),
    ]
    print(f"[probe] {args.sec:.0f}초 수신 중... (MORAI 시나리오 주행 상태 권장)")
    rospy.sleep(args.sec)
    for s in subs:
        s.unregister()

    with p.lock:
        cam_recv = [r for (r, _) in p.cam]
        ego_recv = [r for (r, _) in p.ego]
        obj_recv = [r for (r, _) in p.obj]
        lidar_recv = [r for (r, _) in p.lidar]

    print("\n== 수신율 ==")
    dur = (cam_recv[-1] - cam_recv[0]) if len(cam_recv) > 1 else 0
    for nm, arr in (("cam_front", cam_recv), ("Ego", ego_recv), ("Object", obj_recv),
                    ("lidar", [r for (r, _) in p.lidar])):
        hz = (len(arr) - 1) / dur if dur > 0 and len(arr) > 1 else 0
        print(f"  {nm}: {len(arr)}개 (~{hz:.1f} Hz)")

    print("\n== cam_front 도착시각 대비 recv 최근접 gap (RECV_TOL 결정용) ==")
    mx_e = stats("ego", cam_recv, ego_recv)
    mx_o = stats("obj", cam_recv, obj_recv)
    mx_l = stats("lidar", cam_recv, lidar_recv)
    if lidar_recv and len(lidar_recv) > 1:
        per = (lidar_recv[-1] - lidar_recv[0]) / (len(lidar_recv) - 1) * 1000.0
        print(f"  lidar 주기 실측 ≈ {per:.1f} ms (반주기 {per/2:.1f} ms가 nearest 이상치 기준)")

    print("\n== 클럭 상태 (header - wall) ==")
    clockline("cam_front(클럭A)", p.cam)
    clockline("lidar   (클럭A)", p.lidar)
    clockline("Ego     (클럭B)", p.ego)
    clockline("Object  (클럭B)", p.obj)

    cam_lag = [(r - s) for (r, s) in p.cam if s is not None]
    if cam_lag:
        print(f"\n== 카메라 배달지연(recv-src, 백로그 체크) ==\n"
              f"  시작 {cam_lag[0]:+.3f}s -> 끝 {cam_lag[-1]:+.3f}s")

    if mx_e is not None and mx_o is not None:
        rec = max(mx_e, mx_o) * 2.0
        rec = np.ceil(rec / 10.0) * 10.0
        print(f"\n[권장] ego/obj RECV_TOL = {rec / 1000.0:.3f}s (실측 max의 2배, 10ms 올림).")
    if mx_l is not None and lidar_recv and len(lidar_recv) > 1:
        per = (lidar_recv[-1] - lidar_recv[0]) / (len(lidar_recv) - 1) * 1000.0
        # 인접 스캔 경계 = 1주기. 1주기 초과 gap은 인접 스캔 부재(스톨)라 drop이 맞다.
        rec_l = np.ceil(per / 10.0) * 10.0
        print(f"[권장] lidar RECV_TOL = {rec_l / 1000.0:.3f}s (실측 1주기={per:.1f}ms 올림; "
              f"nearest max={mx_l:.1f}ms가 이보다 크면 스톨 존재 — 그 프레임은 drop이 정답).")


if __name__ == "__main__":
    main()
