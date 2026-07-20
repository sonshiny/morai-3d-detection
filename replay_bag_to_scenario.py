#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
replay_bag_to_scenario.py  (visibility 작업용 clean scene 생성기 — 신규 파일)
============================================================================
유효 원본 bag을 "기존 epoch-safe sync + 기존 GT 생성 로직 그대로" 오프라인에서
replay 해서 새 데이터 루트에 clean scenario 하나를 만든다.

핵심 원칙(보호 대상 무수정):
  * 동기화는 morai_sync.EpochGlitchSynchronizer(mode="epoch") — 라이브 노드가 쓰는 코드.
  * 라벨/이미지/라이다/ego_pose 저장은 morai_3d_live.MoraiLive3Cam3DLabelerCSV의
    _save_frame 경로를 그대로 재사용한다(world_to_ego, 1.28m offset, ROI, visibility
    필터, sync_log 전부 라이브와 동일). 여기서 좌표변환/offset/GT 로직을 새로 만들지 않는다.
  * 유일한 차이: ROS 콜백/마스터 대신, bag receipt 순서로 sync.push를 직접 구동한다
    (geom_verify/replay_ab_c.py가 검증에 쓰는 바로 그 오프라인 에뮬레이션 패턴).
    이를 위해 rospy만 no-op으로 스텁한다(마스터 불필요). sensor_msgs/morai_msgs는 실제 사용.

이후 파이프라인(별도 실행):
  preprocess_dataset.process_scenario(scen_dir)  -> labels_3d_v2 + scene_info.json
  generate_occlusion_gt.process_scen(scen_dir)   -> occlusion/ (num_lidar_pts 보조신호)

사용:
  source /opt/ros/noetic/setup.bash && source ~/morai_ws/devel/setup.bash
  python3 replay_bag_to_scenario.py \
      --bag /home/autonav/geom_drive_all_20260710_180341.bag \
      --dataset_root /home/autonav/visibility_test_dataset
"""

import os
import sys
import types
import argparse

# 데이터셋은 프로젝트 안 dataset/ (스크립트와 동일 위치) 에 있다.
_HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 실제 rospy를 쓰되(= rosbag의 Time 언팩 등 정상 동작) 마스터가 필요한 부분만
# 무해한 no-op으로 monkeypatch 한다. morai_3d_live의 __init__이 실행되기 전에 패치.
#   - Subscriber / on_shutdown / init_node / spin : 마스터/노드 필요 -> no-op
#   - log* : 노드 미초기화 상태에서 throttle 계산 시 시간 접근 -> no-op
#   - Time.now : 노드 미초기화면 예외 -> 안전한 Time(0). (저장경로에선 미사용)
# ---------------------------------------------------------------------------
def _patch_rospy_for_offline():
    import rospy  # 실제 모듈

    class _Sub(object):
        def unregister(self):
            pass

    def _noop(*a, **k):
        return None

    rospy.Subscriber = lambda *a, **k: _Sub()
    rospy.on_shutdown = _noop
    rospy.init_node = _noop
    rospy.spin = _noop
    rospy.loginfo = _noop
    rospy.logwarn = _noop
    rospy.logerr = _noop
    rospy.loginfo_throttle = _noop
    rospy.logwarn_throttle = _noop
    rospy.logerr_throttle = _noop
    _RealTime = rospy.Time
    rospy.Time.now = staticmethod(lambda: _RealTime(0))
    return rospy


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


def replay(bag_path, scen_dir, scenario_name):
    import rosbag  # 실제 ROS (sourced env 필요)
    import morai_sync as MS
    import morai_3d_live as L

    topics = list(MS.TOPIC_TOL.keys()) + [MS.REF_TOPIC]  # ego,obj,left,right,lidar,+cam_front

    labeler = L.MoraiLive3Cam3DLabelerCSV(
        dataset_dir=scen_dir,
        scenario_name=scenario_name,
        max_sync_gap=0.10,
        save_images=True,
        save_lidar=True,
    )

    # bag을 receipt(record) 순서 그대로 스트림으로. (read_messages는 이미 시간순)
    stream = []
    with rosbag.Bag(bag_path, "r") as bag:
        for topic, msg, t in bag.read_messages(topics=topics):
            stream.append((t.to_sec(), topic, msg))
    stream.sort(key=lambda x: x[0])
    print("[replay] bag messages: %d (topics=%d)" % (len(stream), len(topics)))

    # 라이브 콜백과 동일하게 push -> Decision -> _handle(저장/드롭). 단일스레드.
    for (recv, topic, msg) in stream:
        src = _src_ts(msg)
        labeler._handle(labeler.sync.push(topic, src, recv, msg))
    # bag 종료: settle 대기 중이던 ref flush 후 finalize(통계/파일 close).
    labeler._handle(labeler.sync.flush())
    labeler._finalize_locked()

    saved = labeler.sync.frame_idx
    print("[replay] saved frames=%d drops=%d reasons=%s"
          % (saved, labeler._drop_count, labeler._drop_reasons))
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="/home/autonav/geom_drive_all_20260710_180341.bag")
    ap.add_argument("--dataset_root", default=os.path.join(_HERE, "dataset"))
    ap.add_argument("--scenario_prefix", default="scen")
    ap.add_argument("--scenario_digits", type=int, default=2)
    args = ap.parse_args()

    if not os.path.isfile(args.bag):
        raise SystemExit("[ERROR] bag 없음: %s" % args.bag)

    _patch_rospy_for_offline()
    # 패치 이후 import (morai_3d_live가 top-level에서 rospy를 import함 -> 같은 패치된 객체)
    import morai_3d_live as L

    scen_dir, scenario_name = L.create_next_scenario_dir(
        dataset_root=args.dataset_root,
        prefix=args.scenario_prefix,
        digits=args.scenario_digits,
    )
    print("[replay] new scenario: %s (%s)" % (scen_dir, scenario_name))

    n = replay(args.bag, scen_dir, scenario_name)
    if n == 0:
        raise SystemExit("[ERROR] 저장된 프레임이 0개입니다. sync/토픽 확인 필요.")
    print("[replay] DONE. scenario_dir=%s" % scen_dir)


if __name__ == "__main__":
    main()
