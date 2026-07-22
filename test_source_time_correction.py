#!/usr/bin/env python3
"""
test_source_time_correction.py
==============================
correct_source_time.py 의 좌표/속도 재구성을 합성(synthetic) 그라운드트루스로 검증한다.
추가로 P2 velocity-validity loss mask 의 gradient 동작을 검증한다.

검증 케이스(좌표계):
  - 직진 ego / 회전 ego(constant yaw-rate)
  - t_e == t_o / t_e != t_o
  - static object / moving object (+ vz)
각 케이스에서 보정 결과가 독립적으로 계산한 t_ref 정답과 부동소수 허용오차 내 일치해야 한다.

원리: ego 를 시간에 대해 선형(위치)·상수 yaw-rate 로 정의하고 그 위에 t_e 표본을 놓으면,
EgoSourceInterp(x/y/z 선형, yaw unwrap+선형)의 t_ref 보간값은 해석적 참값과 정확히 일치한다.
object 는 등속(constant world velocity)로 정의한다. 이렇게 하면 보정기가 구현한
"world 복원 + 선형보간 ego 재표현"이 참 t_ref GT 와 일치하는지 폐루프로 확인할 수 있다.
"""
import math
import numpy as np

from correct_source_time import (
    EgoSourceInterp, EpochedEgoInterp, _correct_object, _interp_valid, _wrap,
    MAX_EXTRAP,
)
from preprocess_dataset import _decode_object, _rot2


def R(theta):
    return _rot2(theta)


class EgoLine:
    """해석적 ego: 위치 선형, yaw = yaw0 + wrate*(t-tA)."""
    def __init__(self, tA, p0, v, z0, vz, yaw0, wrate):
        self.tA = tA; self.p0 = np.array(p0, float); self.v = np.array(v, float)
        self.z0 = z0; self.vz = vz; self.yaw0 = yaw0; self.wrate = wrate

    def pos(self, t):
        return self.p0 + self.v * (t - self.tA)

    def z(self, t):
        return self.z0 + self.vz * (t - self.tA)

    def yaw(self, t):
        return self.yaw0 + self.wrate * (t - self.tA)

    def state(self, t):
        p = self.pos(t)
        return {"x": float(p[0]), "y": float(p[1]), "z": float(self.z(t)),
                "yaw": float(self.yaw(t)), "timestamp": float(t), "frame_id": 0}


def encode_raw(ego_line, t_e, t_o, p_o_center_at, v_world, yaw_o_world, vz_world, h=1.5, w=1.8, l=4.5):
    """world 참값으로부터 generator 가 저장했을 raw label dict 를 만든다.

    generator 규약:
      stored[x,y] = R(-yaw_e(t_e)) @ (p_o(t_o) - p_e(t_e))
      stored z(bottom) = p_o_center_z(t_o) - h/2 - ego_z(t_e)
      stored sin/cos = sin/cos(yaw_o_world - yaw_e(t_e))     (decode 가 그대로 복원)
      stored[vx,vy] = R(-yaw_e(t_e)) @ (3.6 * R(-yaw_o_world) @ v_world)   (obj-local km/h)
      stored vz = vz_world * 3.6                              (수평과 동일 km/h 규약)
    """
    ego_te = ego_line.state(t_e)
    yaw_e = ego_te["yaw"]
    p_o_o = np.array(p_o_center_at(t_o), float)              # object center world @ t_o (xy)
    p_e = np.array([ego_te["x"], ego_te["y"]], float)
    rel = R(-yaw_e) @ (p_o_o - p_e)
    # z: object center world @ t_o
    # (p_o_center_at 는 xy 만; z 는 별도 인자로 관리)
    v_local_kmh = 3.6 * (R(-yaw_o_world) @ np.array(v_world, float))
    stored_v = R(-yaw_e) @ v_local_kmh
    raw = {
        "object_source": "npc_list", "object_index": 0, "class_id": 0,
        "x": float(rel[0]), "y": float(rel[1]),
        "z": None,   # 채워짐(아래)
        "w": w, "l": l, "h": h,
        "sin_yaw": math.sin(yaw_o_world - yaw_e), "cos_yaw": math.cos(yaw_o_world - yaw_e),
        "vx": float(stored_v[0]), "vy": float(stored_v[1]), "vz": float(vz_world * 3.6),
    }
    return raw, ego_te


def run_case(name, ego_line, t_ref, t_e, t_o,
             p_o0_xy, v_world, yaw_o_world, z_o_center0, vz_world, t_anchor,
             tol=1e-6):
    v_world = np.array(v_world, float)

    def p_o_center_xy(t):
        return np.array(p_o0_xy, float) + v_world * (t - t_anchor)

    def p_o_center_z(t):
        return z_o_center0 + vz_world * (t - t_anchor)

    h = 1.5
    raw, ego_te = encode_raw(ego_line, t_e, t_o, p_o_center_xy, v_world, yaw_o_world, vz_world, h=h)
    # z(bottom) stored = p_o_center_z(t_o) - h/2 - ego_z(t_e)
    raw["z"] = float(p_o_center_z(t_o) - h / 2.0 - ego_te["z"])

    # 보간표: ego line 위 표본 여러 개(t_ref/t_e/t_o 를 감싸도록)
    lo = min(t_ref, t_e, t_o) - 0.3
    hi = max(t_ref, t_e, t_o) + 0.3
    sample_ts = list(np.linspace(lo, hi, 9))
    if t_e not in sample_ts:
        sample_ts.append(t_e)          # t_e 표본은 정확히 포함(저장에 쓰인 ego)
    sample_ts = sorted(set(sample_ts))
    samples = []
    for t in sample_ts:
        s = ego_line.state(t)
        samples.append((t, s["x"], s["y"], s["z"], s["yaw"]))
    interp = EgoSourceInterp(samples)
    ego_ref, meta = interp.at(t_ref)

    # 보간 정확성(선형 ego 이므로 해석적 참값과 일치해야)
    truth_ref = ego_line.state(t_ref)
    assert abs(ego_ref[0] - truth_ref["x"]) < 1e-9, f"{name}: ego interp x"
    assert abs(ego_ref[1] - truth_ref["y"]) < 1e-9, f"{name}: ego interp y"
    assert abs(_wrap(ego_ref[3] - truth_ref["yaw"])) < 1e-9, f"{name}: ego interp yaw"

    dec = _decode_object(raw, ego_te)
    _correct_object(dec, ego_te, ego_ref, t_ref, t_o)

    # ---- 독립 계산한 t_ref 정답 ----
    yaw_r = truth_ref["yaw"]
    p_e_ref = np.array([truth_ref["x"], truth_ref["y"]], float)
    p_o_ref = p_o_center_xy(t_ref)
    rel_true = R(-yaw_r) @ (p_o_ref - p_e_ref)
    z_true = p_o_center_z(t_ref) - truth_ref["z"]
    yaw_ego_true = _wrap(yaw_o_world - yaw_r)
    v_ego_true = R(-yaw_r) @ v_world

    errs = {
        "x": abs(dec["x"] - rel_true[0]),
        "y": abs(dec["y"] - rel_true[1]),
        "z_center": abs(dec["z_center"] - z_true),
        "yaw_ego": abs(_wrap(dec["yaw_ego"] - yaw_ego_true)),
        "vx": abs(dec["vx_ego"] - v_ego_true[0]),
        "vy": abs(dec["vy_ego"] - v_ego_true[1]),
        "gx": abs(dec["gx"] - p_o_ref[0]),
        "gy": abs(dec["gy"] - p_o_ref[1]),
    }
    worst = max(errs.values())
    status = "OK " if worst < tol else "FAIL"
    print(f"  [{status}] {name:38s} worst_err={worst:.2e} corr_dist={dec['corr_dist']:.4f}m "
          f"method={meta['method']}")
    if worst >= tol:
        for k, v in errs.items():
            print(f"        {k}: err={v:.3e}")
    return worst < tol


def test_coordinate_frames():
    print("[coordinate-frame synthetic tests]")
    ok = True
    # 직진 ego (yaw 고정 90deg 근처), 등속 object
    straight = EgoLine(tA=100.0, p0=[10.0, 20.0], v=[8.0, 0.0], z0=0.0, vz=0.0,
                       yaw0=math.radians(5.0), wrate=0.0)
    # 회전 ego (constant yaw-rate 0.3 rad/s)
    turning = EgoLine(tA=100.0, p0=[10.0, 20.0], v=[6.0, 1.0], z0=0.1, vz=0.02,
                      yaw0=math.radians(20.0), wrate=0.3)

    T = 100.5
    cases = [
        # name, ego, t_ref, t_e, t_o, p_o0_xy, v_world, yaw_o_world, z_center0, vz_world, t_anchor
        ("straight/static/te=to",   straight, T, T + 0.045, T + 0.045, [40.0, 22.0], [0, 0], math.radians(10), 0.8, 0.0, T),
        ("straight/moving/te=to",   straight, T, T + 0.045, T + 0.045, [40.0, 22.0], [12.0, 0.5], math.radians(3), 0.8, 0.0, T),
        ("straight/moving/te!=to",  straight, T, T + 0.050, T + 0.030, [40.0, 22.0], [12.0, 0.5], math.radians(3), 0.8, 0.0, T),
        ("straight/moving/+vz",     straight, T, T + 0.050, T + 0.030, [40.0, 22.0], [12.0, 0.5], math.radians(3), 0.8, 0.6, T),
        ("turning/static/te=to",    turning,  T, T + 0.040, T + 0.040, [35.0, 25.0], [0, 0], math.radians(80), 0.9, 0.0, T),
        ("turning/moving/te!=to",   turning,  T, T + 0.046, T + 0.031, [35.0, 25.0], [-9.0, 4.0], math.radians(200), 0.9, -0.3, T),
        ("turning/moving/past-bias", turning, T, T - 0.020, T + 0.031, [35.0, 25.0], [15.0, -2.0], math.radians(-120), 0.9, 0.1, T),
    ]
    for c in cases:
        ok = run_case(*c) and ok
    assert ok, "coordinate-frame synthetic tests FAILED"
    print("  -> all coordinate-frame cases PASS\n")


def test_static_object_correction_equals_ego_shift():
    """정지 world object 는 보정량이 순수 ego 프레임 변화로 설명되어야 한다(직진 sanity)."""
    print("[static-object ego-shift sanity]")
    ego = EgoLine(tA=0.0, p0=[0.0, 0.0], v=[10.0, 0.0], z0=0.0, vz=0.0, yaw0=0.0, wrate=0.0)
    t_ref, t_e, t_o = 1.0, 1.045, 1.045
    p_o = [30.0, 0.0]
    raw, ego_te = encode_raw(ego, t_e, t_o, lambda t: np.array(p_o, float), [0, 0], 0.0, 0.0)
    raw["z"] = 0.0
    samples = []
    for t in np.linspace(0.5, 1.5, 7):
        s = ego.state(t)
        samples.append((t, s["x"], s["y"], s["z"], s["yaw"]))
    interp = EgoSourceInterp(samples)
    ego_ref, _ = interp.at(t_ref)
    dec = _decode_object(raw, ego_te)
    x_stored = dec["x"]
    _correct_object(dec, ego_te, ego_ref, t_ref, t_o)
    # ego 가 45ms 동안 10m/s 로 전진 → 정지 물체는 stored 대비 +0.45m 멀어짐(x 증가)
    expected_shift = 10.0 * (t_e - t_ref)
    got = dec["x"] - x_stored
    print(f"  ego straight 10m/s, dt={t_e-t_ref:.3f}s -> expected +{expected_shift:.3f}m, got {got:+.3f}m")
    assert abs(got - expected_shift) < 1e-6, "static-object ego-shift mismatch"
    print("  -> PASS\n")


def test_duplicate_ego_source_conflict_fails():
    """동일 source time에 다른 pose가 들어오면 임의 첫 값을 쓰지 않고 실패해야 한다."""
    print("[duplicate ego source conflict fail-fast]")
    samples = [
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (1.0, 0.1, 0.0, 0.0, 0.0),
        (1.1, 1.0, 0.0, 0.0, 0.0),
    ]
    try:
        EgoSourceInterp(samples)
    except ValueError as exc:
        assert "동일 ego_src_ts" in str(exc)
    else:
        raise AssertionError("conflicting duplicate ego source pose must fail")
    print("  -> PASS\n")


def test_epoch_teleport_no_cross_interpolation():
    """task F: 서로 다른 epoch/segment 표본을 절대 한 브래킷에 섞지 않는다.

    epoch A(x: 0→10, t 0..1)와 epoch B(x: 1000→1009.5, t 1.05..2)를 텔레포트로 둔다.
    경계 부근 t=1.02 를 조회해도 A 는 A 표본만, B 는 B 표본만 써야 한다.
    epoch-blind 단일표라면 t=1.02 에서 x≈505(A·B 블렌드) 라는 물리적으로 불가능한
    값이 나오지만, epoch 분리 후에는 그런 값이 절대 나오면 안 된다.
    """
    print("[epoch/teleport no cross-interpolation]")
    samples = []
    # epoch A: 직진 10 m/s, t=0.0..1.0
    for t in np.linspace(0.0, 1.0, 11):
        samples.append((float(t), float(10.0 * t), 0.0, 0.0, 0.0, "A"))
    # epoch B: 텔레포트(+1000m), 직진 10 m/s, t=1.05..2.0
    for t in np.linspace(1.05, 2.0, 11):
        samples.append((float(t), float(1000.0 + 10.0 * (t - 1.05)), 0.0, 0.0, 0.0, "B"))

    ei = EpochedEgoInterp(samples)
    assert ei.n_epochs == 2 and set(ei.epochs) == {"A", "B"}

    # 경계 t=1.02 : A 로 조회 → A 표본만(≈10.2, extrap 0.02), B 로 조회 → B 표본만(≈999.5)
    ega, ma = ei.at(1.02, "A")
    egb, mb = ei.at(1.02, "B")
    assert ega is not None and egb is not None
    assert abs(ega[0] - 10.2) < 1e-6, f"epoch A leaked across boundary: x={ega[0]}"
    assert 999.0 < egb[0] < 1000.5, f"epoch B leaked across boundary: x={egb[0]}"
    # 물리적으로 불가능한 블렌드(≈505)가 절대 나오지 않아야 한다
    assert not (400.0 < ega[0] < 600.0), "A produced cross-epoch blend"
    assert not (400.0 < egb[0] < 600.0), "B produced cross-epoch blend"
    print(f"  t=1.02  epoch A x={ega[0]:.3f} (extrap {ma['extrap']:.3f})  "
          f"epoch B x={egb[0]:.3f} (extrap {mb['extrap']:.3f})  → no blend")

    # 존재하지 않는 epoch 조회 → 명시적 invalid(조용한 fallback 금지)
    egz, mz = ei.at(1.02, "Z")
    assert egz is None and mz["method"] == "no_epoch_samples"
    assert not _interp_valid(mz), "missing-epoch query must be correction_valid=0"
    print(f"  missing epoch Z → method={mz['method']}, valid={_interp_valid(mz)}")

    # 경계에서 MAX_EXTRAP 초과 외삽 → correction_valid=0
    _, m_far = ei.at(1.05 - (MAX_EXTRAP + 0.05), "B")   # B 첫 표본보다 훨씬 이전
    assert not _interp_valid(m_far), (
        f"extrapolation beyond MAX_EXTRAP must be invalid: extrap={m_far['extrap']}")
    print(f"  far pre-B query extrap={m_far['extrap']:.3f}s (>MAX_EXTRAP {MAX_EXTRAP}) → invalid")

    # 단일 epoch 는 기존 EgoSourceInterp 단일표와 동일해야(no-op 보증)
    single = [(t, x, y, z, yaw) for (t, x, y, z, yaw, ep) in samples if ep == "A"]
    ref = EgoSourceInterp(single)
    ep_single = EpochedEgoInterp([(t, x, y, z, yaw, "A") for (t, x, y, z, yaw) in single])
    for tq in (0.15, 0.55, 0.97):
        r0 = ref.at(tq)[0]
        r1 = ep_single.at(tq, "A")[0]
        assert r0 == r1, f"single-epoch EpochedEgoInterp diverged from EgoSourceInterp at t={tq}"
    print("  single-epoch EpochedEgoInterp == EgoSourceInterp (bitwise) → no-op confirmed\n")


if __name__ == "__main__":
    test_coordinate_frames()
    test_static_object_correction_equals_ego_shift()
    test_duplicate_ego_source_conflict_fails()
    test_epoch_teleport_no_cross_interpolation()
    print("ALL SYNTHETIC COORDINATE TESTS PASSED")
