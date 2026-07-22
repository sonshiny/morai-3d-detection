#!/usr/bin/env python3
"""Stage 1: invariants (rule9) + implementation reproduction (gradeC, rule5)
+ formula decomposition (rule3) + negative controls (rule8). Read-only."""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v3_lib as L

def main():
    inv = dict(n=0, dim_mismatch=0, class_mismatch=0, tid_only_v2=0, tid_only_v3=0,
               yawglobal_max=0.0)
    rep = dict(n=0)
    rep_err = {k: [] for k in ("x","y","gx","gy","yaw_ego","vx_ego","vy_ego")}
    decomp_err = []
    # negative controls accumulators
    nc = dict(smalldt_n=0, smalldt_cdist=[],
              statobj_statego_n=0, statobj_statego_cdist=[],
              worldstatic_n=0, worldstatic_gdrift=[],
              movobj_statego_n=0, movobj_statego_objfrac=[],
              statobj_movego_n=0, statobj_movego_objterm=[])
    for scen in L.SCENS:
        sync = L.load_sync(scen); ego = L.load_ego(scen)
        v2 = L.load_boxes(scen, "v2"); v3 = L.load_boxes(scen, "v3")
        interp = L.build_ego_interp(scen, sync, ego)
        for stem, b3map in v3.items():
            b2map = v2.get(stem, {})
            tm = sync.get(stem); e = ego.get(stem)
            # track_id set invariance
            s2, s3 = set(b2map), set(b3map)
            inv["tid_only_v2"] += len(s2 - s3); inv["tid_only_v3"] += len(s3 - s2)
            if tm is None or e is None:
                continue
            ego_ref = interp.at(tm["t_ref"])
            for tid, b3 in b3map.items():
                if tid not in b2map:
                    continue
                b2 = b2map[tid]
                inv["n"] += 1
                if abs(b2["w"]-b3["w"])>1e-4 or abs(b2["l"]-b3["l"])>1e-4 or abs(b2["h"]-b3["h"])>1e-4:
                    inv["dim_mismatch"] += 1
                if b2["class_id"] != b3["class_id"]:
                    inv["class_mismatch"] += 1
                inv["yawglobal_max"] = max(inv["yawglobal_max"], abs(L.wrap(b2["yaw_global"]-b3["yaw_global"])))
                # reproduction only on validly-corrected boxes
                if int(b3.get("correction_valid", 0)) != 1 or ego_ref is None:
                    continue
                rc = L.recompute_v3(b2, e, ego_ref, tm["t_ref"], tm["t_o"])
                rep["n"] += 1
                for k in rep_err:
                    rep_err[k].append(abs(rc[k] - b3[k]))
                # decomposition: term sum vs (v3_stored_rel - v2_rel)
                tsum = rc["term_objvel"] + rc["term_egotrans"] + rc["term_egorot"]
                actual = np.array([b3["x"]-b2["x"], b3["y"]-b2["y"]])
                decomp_err.append(float(np.max(np.abs(tsum - actual))))
                # ---- negative controls ----
                dt_o = abs(tm["t_ref"]-tm["t_o"]); dt_e = abs(tm["t_ref"]-tm["t_e"])
                ego_shift = math.hypot(e["x"]-ego_ref[0], e["y"]-ego_ref[1])
                ego_yawshift = abs(L.wrap(e["yaw"]-ego_ref[3]))
                vspeed = math.hypot(rc["v_world"][0], rc["v_world"][1])
                cd = b3.get("corr_dist", 0.0)
                israw = b2.get("vel_source","raw")=="raw"
                if dt_o < 0.010 and dt_e < 0.010:
                    nc["smalldt_n"] += 1; nc["smalldt_cdist"].append(cd)
                if vspeed < 0.05 and ego_shift < 0.02 and ego_yawshift < 1e-3:
                    nc["statobj_statego_n"] += 1; nc["statobj_statego_cdist"].append(cd)
                if israw and vspeed < 0.05:  # world-static object: global pos should be unchanged
                    nc["worldstatic_n"] += 1
                    nc["worldstatic_gdrift"].append(math.hypot(b3["gx"]-b2["gx"], b3["gy"]-b2["gy"]))
                if israw and vspeed > 1.0 and ego_shift < 0.02 and ego_yawshift < 1e-3:
                    # moving object + stationary ego: displacement not explained by obj-vel term
                    resid = math.hypot(*(actual - rc["term_objvel"]))
                    nc["movobj_statego_n"] += 1; nc["movobj_statego_objfrac"].append(resid)
                if vspeed < 0.05 and (ego_shift > 0.05 or ego_yawshift > 2e-3):
                    # stationary object + moving ego: obj-vel term ~0, rest explains it
                    nc["statobj_movego_n"] += 1
                    nc["statobj_movego_objterm"].append(math.hypot(*rc["term_objvel"]))
    # ---- report ----
    def st(a):
        a = np.array(a) if len(a) else np.array([0.0])
        return "mean=%.5f med=%.5f p95=%.5f max=%.5f n=%d" % (a.mean(), np.median(a), np.percentile(a,95), a.max(), len(a))
    print("=================== RULE 9 — INVARIANTS (v2 vs v3) ===================")
    print("  joined boxes: %d | dim_mismatch=%d | class_mismatch=%d | track_id only-in-v2=%d only-in-v3=%d"
          % (inv["n"], inv["dim_mismatch"], inv["class_mismatch"], inv["tid_only_v2"], inv["tid_only_v3"]))
    print("  |yaw_global(v2)-yaw_global(v3)| max = %.3e rad" % inv["yawglobal_max"])
    print("=================== RULE 5 — IMPLEMENTATION REPRODUCTION (grade C) ===================")
    print("  reproduced valid boxes: %d   (abs err |recompute - stored v3|)" % rep["n"])
    for k in ("x","y","gx","gy","yaw_ego","vx_ego","vy_ego"):
        print("    %-8s %s" % (k, st(rep_err[k])))
    print("=================== RULE 3 — FORMULA DECOMPOSITION (term sum == v3-v2) ===================")
    print("    max|(objvel+egotrans+egorot) - (v3-v2)_xy| : %s" % st(decomp_err))
    print("=================== RULE 8 — NEGATIVE CONTROLS ===================")
    print("  (a) t_ref~t_o~t_e (<10ms): n=%d  corr_dist %s  [expect ~0]" % (nc["smalldt_n"], st(nc["smalldt_cdist"])))
    print("  (b) stat obj + stat ego: n=%d  corr_dist %s  [expect ~0]" % (nc["statobj_statego_n"], st(nc["statobj_statego_cdist"])))
    print("  (c) world-static obj: n=%d  |global drift v3-v2| %s  [expect ~0]" % (nc["worldstatic_n"], st(nc["worldstatic_gdrift"])))
    print("  (d) moving obj + stat ego: n=%d  |disp - objvel_term| m %s  [expect ~0 = ego residual]" % (nc["movobj_statego_n"], st(nc["movobj_statego_objfrac"])))
    print("  (e) stat obj + moving ego: n=%d  objvel-term magnitude %s  [expect ~0]" % (nc["statobj_movego_n"], st(nc["statobj_movego_objterm"])))

if __name__ == "__main__":
    main()
