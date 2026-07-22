#!/usr/bin/env python3
"""Stage 3: correction MAGNITUDE only (task 2) + 3-5 track time-series plots.
corr_dist is a MAGNITUDE, not an accuracy metric (rule 1). Read-only inputs;
writes plots + stats JSON to experiments/v3_correctness/. """
import sys, os, math, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import v3_lib as L

OUT = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)
THR = 2.0

def summ(a):
    a = np.asarray(a, float)
    if not a.size: return dict(n=0)
    return dict(n=int(a.size), mean=round(float(a.mean()),4), med=round(float(np.median(a)),4),
                p95=round(float(np.percentile(a,95)),4), max=round(float(a.max()),4))

def main():
    stats = {"per_scene": {}, "overall": {}, "note": "corr_dist = |v3 - original-decode| ego-frame displacement; MAGNITUDE only, not accuracy"}
    all_cd=[]; all_valid=[]; all_ospeed=[]; all_dt=[]; all_cls=[]
    tracks_all={}
    for scen in L.SCENS:
        sync=L.load_sync(scen); ego=L.load_ego(scen)
        v2=L.load_boxes(scen,"v2"); v3=L.load_boxes(scen,"v3")
        interp=L.build_ego_interp(scen,sync,ego)
        cds=[]; vs=[]; dts=[]; osp=[]
        for stem,b3map in v3.items():
            tm=sync.get(stem); e=ego.get(stem)
            for tid,b3 in b3map.items():
                cd=float(b3.get("corr_dist",0.0)); cds.append(cd)
                vs.append(int(b3.get("correction_valid",1)))
                if tm and e and tid in v2.get(stem,{}):
                    ego_ref=interp.at(tm["t_ref"])
                    rc=L.recompute_v3(v2[stem][tid],e,ego_ref,tm["t_ref"],tm["t_o"])
                    osp.append(math.hypot(*rc["v_world"])); dts.append(abs(tm["t_ref"]-tm["t_o"]))
                    tracks_all.setdefault((scen,tid),[]).append(dict(
                        t=tm["t_ref"], cd=cd, os=math.hypot(*rc["v_world"]),
                        gx2=v2[stem][tid]["gx"], gy2=v2[stem][tid]["gy"],
                        gx3=b3["gx"], gy3=b3["gy"], t_o=tm["t_o"], cls=b3["class_id"]))
        cds=np.array(cds)
        stats["per_scene"][scen]=dict(
            n_boxes=len(cds), corr_dist=summ(cds),
            frac_gt_0p2m=round(float((cds>0.2).mean()),4),
            frac_gt_0p5m=round(float((cds>0.5).mean()),4),
            frac_ge_2p0m=round(float((cds>=THR).mean()),4),
            ratio_corr_over_thr_p95=round(float(np.percentile(cds,95))/THR,4),
            n_correction_valid=int(np.sum(vs)))
        all_cd+=list(cds); all_ospeed+=osp; all_dt+=dts
    all_cd=np.array(all_cd)
    stats["overall"]=dict(n_boxes=len(all_cd), corr_dist=summ(all_cd),
        frac_gt_0p2m=round(float((all_cd>0.2).mean()),4),
        frac_gt_0p5m=round(float((all_cd>0.5).mean()),4),
        frac_ge_2p0m=round(float((all_cd>=THR).mean()),4),
        max_over_thr=round(float(all_cd.max())/THR,4),
        obj_world_speed_ms=summ(all_ospeed), abs_dt_o_s=summ(all_dt))
    json.dump(stats, open(os.path.join(OUT,"magnitude_stats.json"),"w"), indent=2)
    print(json.dumps(stats, indent=2))

    # pick 5 vehicle tracks with the most motion (largest corr_dist span) and enough frames
    cand=[]
    for k,seq in tracks_all.items():
        if len(seq)<15: continue
        seq.sort(key=lambda d:d["t"])
        cd=np.array([s["cd"] for s in seq])
        if seq[0]["cls"]!=0: continue
        cand.append((cd.mean()*cd.max(), k, seq))
    cand.sort(reverse=True)
    picks=cand[:5]
    manifest=[]
    for score,(scen,tid),seq in picks:
        t=np.array([s["t"] for s in seq]); t=t-t[0]
        cd=np.array([s["cd"] for s in seq]); osp=np.array([s["os"] for s in seq])
        gx2=np.array([s["gx2"] for s in seq]); gy2=np.array([s["gy2"] for s in seq])
        gx3=np.array([s["gx3"] for s in seq]); gy3=np.array([s["gy3"] for s in seq])
        fig,ax=plt.subplots(1,2,figsize=(14,4.6))
        ax[0].plot(gx2,gy2,'-o',color="#ff3030",ms=3,lw=1,label="v2 world @t_o")
        ax[0].plot(gx3,gy3,'-o',color="#20a020",ms=3,lw=1,label="v3 world @t_ref")
        ax[0].set_title("%s track %d — world trajectory (v2 red @t_o, v3 green @t_ref)"%(scen,tid),fontsize=9)
        ax[0].set_xlabel("gx (m)"); ax[0].set_ylabel("gy (m)"); ax[0].legend(fontsize=7); ax[0].axis("equal"); ax[0].grid(alpha=.3)
        ax[1].plot(t,cd,'-',color="#7030ff",label="corr_dist |v3-decode| (m)")
        ax[1].plot(t,osp,'-',color="#ff9000",alpha=.7,label="obj world speed (m/s)")
        ax[1].axhline(THR,color="k",ls="--",lw=.8,alpha=.5,label="2.0 m matcher thr")
        ax[1].set_title("correction MAGNITUDE over time (not accuracy)",fontsize=9)
        ax[1].set_xlabel("t - t0 (s)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
        fn="track_%s_t%d.png"%(scen,tid)
        fig.tight_layout(); fig.savefig(os.path.join(PLOTS,fn),dpi=95); plt.close(fig)
        manifest.append(dict(scen=scen,track_id=int(tid),n_frames=len(seq),
            corr_dist_mean=round(float(cd.mean()),4),corr_dist_max=round(float(cd.max()),4),
            obj_speed_max=round(float(osp.max()),4),file="plots/"+fn))
        print("plot:",fn,"n=%d cd_mean=%.3f cd_max=%.3f ospeed_max=%.2f"%(len(seq),cd.mean(),cd.max(),osp.max()))
    json.dump(manifest, open(os.path.join(OUT,"track_plots_manifest.json"),"w"), indent=2)

if __name__=="__main__":
    main()
