#!/usr/bin/env python3
"""Read-only shared loaders + independent re-derivation of the v3 source-time
correction, for the v3 correctness diagnosis. Touches NOTHING in the project.
Reads dataset/<scen>/{sync_log.csv, ego_pose, labels_3d_v2, labels_3d_v3}."""
import os, csv, math
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DS = os.path.abspath(os.environ.get("DATASET_ROOT", os.path.join(REPO, "dataset")))
SCENS = [
    x.strip() for x in os.environ.get(
        "V3_SCENARIOS", "scen05,scen77,scen144"
    ).split(",") if x.strip()
]

def rot2(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s], [s, c]], dtype=np.float64)

def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))

def load_sync(scen):
    out = {}
    with open(os.path.join(DS, scen, "sync_log.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if r.get("action") != "save":
                continue
            try:
                out[r["stem"]] = dict(t_ref=float(r["ref_src_ts"]), t_e=float(r["ego_src_ts"]),
                                      t_o=float(r["obj_src_ts"]), epoch=r.get("epoch_id", ""))
            except (KeyError, ValueError):
                continue
    return out

def load_ego(scen):
    out = {}
    d = os.path.join(DS, scen, "ego_pose")
    for fn in os.listdir(d):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(d, fn), newline="") as f:
            r = next(csv.DictReader(f))
        stem = os.path.splitext(fn)[0]
        out[stem] = dict(x=float(r["ego_x"]), y=float(r["ego_y"]), z=float(r["ego_z"]),
                         yaw=float(r["ego_yaw_rad"]), frame_id=int(r["frame_id"]))
    return out

def load_boxes(scen, kind):
    """kind in {v2,v3}. -> {stem: {track_id(int): row(dict of floats/ints)}}"""
    sub = "labels_3d_v2" if kind == "v2" else "labels_3d_v3"
    d = os.path.join(DS, scen, sub)
    out = {}
    for fn in os.listdir(d):
        if not fn.endswith(".csv"):
            continue
        stem = os.path.splitext(fn)[0]
        bx = {}
        with open(os.path.join(d, fn), newline="") as f:
            for r in csv.DictReader(f):
                rec = {}
                for k, v in r.items():
                    if k in ("track_id", "class_id", "frame_id", "correction_valid"):
                        try: rec[k] = int(float(v))
                        except: rec[k] = v
                    elif k == "vel_source":
                        rec[k] = v
                    else:
                        try: rec[k] = float(v)
                        except: rec[k] = v
                bx[rec["track_id"]] = rec
        out[stem] = bx
    return out

class EgoInterp:
    """Independent replica of correct_source_time.EgoSourceInterp.at (single epoch)."""
    def __init__(self, samples):  # samples: [(t_e,x,y,z,yaw)]
        by = {}
        for t_e, x, y, z, yaw in samples:
            key = round(float(t_e), 6)
            if key not in by:
                by[key] = (x, y, z, yaw)
        ts = sorted(by)
        self.ts = np.array(ts)
        self.xs = np.array([by[t][0] for t in ts])
        self.ys = np.array([by[t][1] for t in ts])
        self.zs = np.array([by[t][2] for t in ts])
        ry = np.array([by[t][3] for t in ts])
        self.yaws = np.unwrap(ry) if ry.size > 1 else ry
    def at(self, t):
        ts = self.ts; n = ts.size
        if n == 0: return None
        if n == 1:
            return float(self.xs[0]), float(self.ys[0]), float(self.zs[0]), wrap(float(self.yaws[0]))
        i = int(np.searchsorted(ts, t))
        if i <= 0: lo, hi = 0, 1
        elif i >= n: lo, hi = n-2, n-1
        else: lo, hi = i-1, i
        t0, t1 = ts[lo], ts[hi]; dt = t1 - t0
        frac = 0.0 if dt <= 0 else (t - t0)/dt
        ip = lambda a: float(a[lo] + (a[hi]-a[lo])*frac)
        return ip(self.xs), ip(self.ys), ip(self.zs), wrap(ip(self.yaws))

def build_ego_interp(scen, sync, ego):
    samples = []
    for stem, tm in sync.items():
        if stem in ego:
            e = ego[stem]
            samples.append((tm["t_e"], e["x"], e["y"], e["z"], e["yaw"]))
    return EgoInterp(samples)

def recompute_v3(b2, ego_te, ego_ref, t_ref, t_o):
    """Independent re-derivation of _correct_object from v2 row + ego pieces.
    Returns dict with x,y,gx,gy,yaw_ego,vx_ego,vy_ego and decomposition terms."""
    ex_r, ey_r, ez_r, yaw_r = ego_ref
    yaw_te = ego_te["yaw"]
    p_o_o = np.array([b2["gx"], b2["gy"]], dtype=np.float64)          # object world @ t_o
    ego_yaw_te = wrap(b2["yaw_global"] - b2["yaw_ego"])              # recovered t_e ego yaw
    v_ego = np.array([b2["vx_ego"], b2["vy_ego"]], dtype=np.float64) # t_e ego frame m/s
    v_world = rot2(ego_yaw_te) @ v_ego
    dt_o = t_ref - t_o
    if b2.get("vel_source", "raw") == "raw":
        dvel = v_world * dt_o
    else:
        dvel = np.zeros(2)
    p_o_ref = p_o_o + dvel
    p_e_te = np.array([ego_te["x"], ego_te["y"]], dtype=np.float64)
    p_e_ref = np.array([ex_r, ey_r], dtype=np.float64)
    rel = rot2(-yaw_r) @ (p_o_ref - p_e_ref)
    yaw_ego2 = wrap(b2["yaw_global"] - yaw_r)
    v_ego2 = rot2(-yaw_r) @ v_world
    # decomposition of (v3_rel - v2_rel):
    v2_rel = np.array([b2["x"], b2["y"]], dtype=np.float64)
    # exact decomposition: v3_rel - v2_rel = term_egorot + term_egotrans + term_objvel
    # where v2_rel = R(-yaw_te)@(p_o_o - p_e_te), yaw_te = ego_yaw_te (recovered t_e ego yaw)
    term_objvel = rot2(-yaw_r) @ dvel
    term_egotrans = rot2(-yaw_r) @ (p_e_te - p_e_ref)
    term_egorot = (rot2(-yaw_r) - rot2(-ego_yaw_te)) @ (p_o_o - p_e_te)
    return dict(x=float(rel[0]), y=float(rel[1]), gx=float(p_o_ref[0]), gy=float(p_o_ref[1]),
                yaw_ego=yaw_ego2, vx_ego=float(v_ego2[0]), vy_ego=float(v_ego2[1]),
                v_world=v_world, dt_o=dt_o, dvel=dvel,
                term_objvel=term_objvel, term_egotrans=term_egotrans, term_egorot=term_egorot,
                v2_rel=v2_rel, v3_rel=rel, ego_yaw_te=ego_yaw_te)
