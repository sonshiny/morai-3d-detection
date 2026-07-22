#!/usr/bin/env python3
"""Stage 2: leave-one-out (LOO) temporal consistency (grade B, rules 6-7).
Reference trajectory = MORAI object world positions (v2 gx,gy) at obj_src_ts.
For each target frame k, EXCLUDE k; bracket t_ref_k with the track's other
raw world samples (both-side neighbors, same epoch, gap<=MAX_GAP); linearly
interpolate to t_ref_k => ref_world(t_ref_k). Compare v2 world (pos@t_o labeled
t_ref) vs v3 world (propagated) residual against that reference.
Also estimate constant-velocity model error 0.5|a|dt^2. Read-only."""
import sys, os, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import v3_lib as L

MAX_GAP = 0.5   # s : bracket span guard (teleport / track gap)

def pct(a, q): return float(np.percentile(a, q)) if len(a) else float("nan")
def summ(a):
    a = np.asarray(a, float)
    if not a.size: return dict(n=0)
    return dict(n=int(a.size), mean=float(a.mean()), med=float(np.median(a)),
                p95=float(np.percentile(a,95)), max=float(a.max()))

def boot_ci(delta, n=2000, seed=12345):
    d = np.asarray(delta, float)
    if d.size < 3: return (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    means = [d[rng.randint(0, d.size, d.size)].mean() for _ in range(n)]
    return (float(np.percentile(means,2.5)), float(np.percentile(means,97.5)))

def main():
    rows = []   # per-box LOO record
    for scen in L.SCENS:
        sync = L.load_sync(scen); ego = L.load_ego(scen)
        v2 = L.load_boxes(scen, "v2"); v3 = L.load_boxes(scen, "v3")
        interp = L.build_ego_interp(scen, sync, ego)
        # build per-track series
        tracks = {}
        for stem, b3map in v3.items():
            tm = sync.get(stem); e = ego.get(stem)
            if tm is None or e is None: continue
            b2map = v2.get(stem, {})
            for tid, b3 in b3map.items():
                if tid not in b2map: continue
                b2 = b2map[tid]
                if int(b3.get("correction_valid",0)) != 1: continue
                ego_ref = interp.at(tm["t_ref"])
                rc = L.recompute_v3(b2, e, ego_ref, tm["t_ref"], tm["t_o"])
                tracks.setdefault((scen,tid), []).append(dict(
                    t_o=tm["t_o"], t_ref=tm["t_ref"], epoch=tm["epoch"],
                    gx2=b2["gx"], gy2=b2["gy"], gx3=b3["gx"], gy3=b3["gy"],
                    cls=b2["class_id"], vsrc=b2.get("vel_source","raw"),
                    rng=math.hypot(b3["x"], b3["y"]),
                    ospeed=math.hypot(*rc["v_world"]), dt_o=tm["t_ref"]-tm["t_o"]))
        for (scen,tid), seq in tracks.items():
            seq.sort(key=lambda d: d["t_o"])
            n = len(seq)
            tarr = np.array([s["t_o"] for s in seq])
            gx = np.array([s["gx2"] for s in seq]); gy = np.array([s["gy2"] for s in seq])
            for k in range(n):
                s = seq[k]; tref = s["t_ref"]
                # neighbors excluding k, same epoch
                lo = hi = None
                for j in range(k-1, -1, -1):
                    if seq[j]["epoch"] != s["epoch"]: break
                    if tarr[j] <= tref: lo = j; break
                for j in range(k+1, n):
                    if seq[j]["epoch"] != s["epoch"]: break
                    if tarr[j] >= tref: hi = j; break
                if lo is None or hi is None:
                    onesided = True
                    # try one-sided nearest two same-epoch neighbors for extrap (flagged)
                    continue  # PRIMARY requires both-side; skip one-sided here
                if (tarr[hi]-tarr[lo]) > MAX_GAP: continue
                frac = (tref - tarr[lo])/(tarr[hi]-tarr[lo]) if tarr[hi]>tarr[lo] else 0.0
                rx = gx[lo] + (gx[hi]-gx[lo])*frac
                ry = gy[lo] + (gy[hi]-gy[lo])*frac
                res_v2 = math.hypot(s["gx2"]-rx, s["gy2"]-ry)
                res_v3 = math.hypot(s["gx3"]-rx, s["gy3"]-ry)
                # const-velocity model error: accel from lo->k-ish->hi second difference
                cverr = float("nan")
                if hi-lo >= 2 or (hi>lo):
                    dt1 = tarr[k]-tarr[lo] if tarr[k]>tarr[lo] else None
                    # estimate accel using lo,k,hi three raw samples (their own positions)
                    try:
                        t0,t1,t2 = tarr[lo],tarr[k],tarr[hi]
                        if t2>t1>t0:
                            v01=np.array([(gx[k]-gx[lo])/(t1-t0),(gy[k]-gy[lo])/(t1-t0)])
                            v12=np.array([(gx[hi]-gx[k])/(t2-t1),(gy[hi]-gy[k])/(t2-t1)])
                            a=(v12-v01)/(0.5*(t2-t0))
                            cverr=0.5*math.hypot(*a)*(s["dt_o"]**2)
                    except Exception: pass
                rows.append(dict(scen=scen,tid=tid,cls=s["cls"],vsrc=s["vsrc"],
                                 rng=s["rng"],ospeed=s["ospeed"],adt=abs(s["dt_o"]),
                                 res_v2=res_v2,res_v3=res_v3,cverr=cverr))
    R = rows
    print("=================== RULE 6 — LEAVE-ONE-OUT TEMPORAL CONSISTENCY (grade B) ===================")
    print("  reference = target-excluded neighbor interpolation of MORAI object world pos to t_ref")
    print("  (same MORAI object source as v3 input -> grade B temporal-consistency evidence, NOT grade-A GT)")
    print("  primary LOO boxes (both-side neighbors, gap<=%.1fs): %d" % (MAX_GAP, len(R)))
    def block(name, sub):
        if not sub: print("  [%s] n=0" % name); return
        rv2=np.array([r["res_v2"] for r in sub]); rv3=np.array([r["res_v3"] for r in sub])
        d=rv3-rv2; ci=boot_ci(d)
        print("  [%s] n=%d" % (name,len(sub)))
        print("      res_v2(m): mean=%.4f med=%.4f p95=%.4f max=%.4f" % (rv2.mean(),np.median(rv2),pct(rv2,95),rv2.max()))
        print("      res_v3(m): mean=%.4f med=%.4f p95=%.4f max=%.4f" % (rv3.mean(),np.median(rv3),pct(rv3,95),rv3.max()))
        print("      paired delta(v3-v2) mean=%.4f med=%.4f  95%%CI[%.4f,%.4f]  frac(v3<v2)=%.3f" %
              (d.mean(),np.median(d),ci[0],ci[1],float((d<0).mean())))
    block("ALL", R)
    block("vehicle(raw)", [r for r in R if r["cls"]==0 and r["vsrc"]=="raw"])
    block("pedestrian", [r for r in R if r["cls"]==1])
    block("motion-vel", [r for r in R if r["vsrc"]=="motion"])
    # speed bins (vehicles raw)
    veh=[r for r in R if r["cls"]==0 and r["vsrc"]=="raw"]
    for lo_,hi_ in [(0,1),(1,3),(3,6),(6,50)]:
        block("veh ospeed[%d,%d)m/s"%(lo_,hi_), [r for r in veh if lo_<=r["ospeed"]<hi_])
    # dt bins
    for lo_,hi_ in [(0,0.02),(0.02,0.05),(0.05,0.2)]:
        block("veh |dt|[%.2f,%.2f)s"%(lo_,hi_), [r for r in veh if lo_<=r["adt"]<hi_])
    # distance bins
    for lo_,hi_ in [(0,20),(20,40),(40,80)]:
        block("veh range[%d,%d)m"%(lo_,hi_), [r for r in veh if lo_<=r["rng"]<hi_])
    # constant-velocity model error (rule 7)
    cv=[r["cverr"] for r in veh if not math.isnan(r["cverr"])]
    print("=================== RULE 7 — CONST-VELOCITY MODEL ERROR (vehicles) ===================")
    print("  0.5|a|dt^2 estimate: %s" % summ(cv))
    print("  -> this is the theoretical floor of v3 propagation residual (separate from timestamp-jitter correction)")

if __name__ == "__main__":
    main()
