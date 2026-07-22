#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build visual_audit/README.md and index.html from the 4 category manifests.
"""
import os
import json

HERE = os.path.dirname(os.path.abspath(__file__))


def load(sub):
    p = os.path.join(HERE, sub, "_manifest.json")
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    depth = load("depth") or []
    v2v3 = load("v2_vs_v3") or {"items": [], "extra_notes": []}
    boundary = load("boundary_50m") or []
    temporal = load("track_temporal") or []
    v2v3_items = v2v3.get("items", [])
    v2v3_notes = v2v3.get("extra_notes", [])

    md = []
    html = ["<meta charset='utf-8'><style>body{font-family:sans-serif;max-width:1500px;"
            "margin:auto;padding:20px} img{max-width:100%;border:1px solid #ccc;margin:6px 0}"
            "h2{border-bottom:2px solid #444;margin-top:40px} .cap{color:#333;font-size:14px}"
            "details{margin:8px 0}</style>"]

    md.append("# MORAI 3D-detection Visual Audit\n")
    md.append("생성 스크립트: `visual_audit/*.py` (원본 데이터는 읽기 전용, 어떤 라벨/pose/scene_info도 수정하지 않음).\n")
    md.append(f"이미지 총계: depth={len(depth)}, v2_vs_v3={len(v2v3_items)}, "
              f"boundary_50m={len(boundary)}, track_temporal={len(temporal)} "
              f"(합계 {len(depth)+len(v2v3_items)+len(boundary)+len(temporal)}).\n")
    md.append("좌표 규약: 카메라 투영은 `camera_configs.py`/`visualize_camera_proj.py`와 동일 "
              "(depth=cam_x, u=fx·(-cam_y)/depth+cx, v=fy·(-cam_z)/depth+cy). "
              "BEV는 ego 원점, x=전방, y=좌.\n")
    md.append("**중요:** depth_gt의 (u,v)는 학습 입력 좌표계(704×256)에서 생성됨 "
              "(`generate_depth_gt.py` → `scale_intrinsic_for_input`). "
              "원본 1600×900 이미지 위에 겹치기 위해 (u,v)를 (1600/704, 900/256)배 확대해 정합함.\n")

    html.append("<h1>MORAI 3D-detection Visual Audit</h1>")
    html.append(f"<p>이미지 총계: depth={len(depth)}, v2_vs_v3={len(v2v3_items)}, "
                f"boundary_50m={len(boundary)}, track_temporal={len(temporal)}.</p>")

    # 1. depth
    md.append("\n## 1. depth/ — depth GT 오버레이 (9장)\n")
    md.append("각 scene×카메라 1프레임: 좌=원본, 우=depth_gt scatter(색=depth, colorbar). "
              "이동객체(rel_speed>1)가 있는 프레임을 선택.\n")
    md.append("| scene | camera | stem | depth_pts | movers(rel>1) | max_rel(m/s) | mover in cam | file |")
    md.append("|---|---|---|---|---|---|---|---|")
    html.append("<h2>1. depth — depth GT 오버레이 (9)</h2>")
    for d in depth:
        md.append(f"| {d['scen']} | {d['cam']} | {d['stem']} | {d['n_depth_pts']} | "
                  f"{d['n_movers']} | {d['max_rel_speed']:.1f} | {d['mover_visible_in_cam']} | "
                  f"`depth/{d['file']}` |")
        html.append(f"<div class='cap'>{d['scen']} / {d['cam']} / {d['stem']} — "
                    f"depth_pts={d['n_depth_pts']}, movers={d['n_movers']}, "
                    f"max_rel={d['max_rel_speed']:.1f} m/s</div>"
                    f"<img src='depth/{d['file']}'>")

    # 2. v2_vs_v3
    md.append("\n## 2. v2_vs_v3/ — v2→v3 보정 감사 (카메라 투영 + BEV) ({}장)\n".format(len(v2v3_items)))
    md.append("v2=빨강, v3=초록, 흰 화살표=v2→v3 변위, 파랑=객체 속도, 노랑=ego 속도. "
              "패널: 카메라투영 | BEV(50m 링) | BEV 확대(±6m).\n")
    if v2v3_notes:
        md.append("카테고리 보강 노트:")
        for n in v2v3_notes:
            md.append(f"- {n}")
        md.append("")
    md.append("| scene | frame | track | class | corr_dist(m) | rel/obj/ego (m/s) | obj_dt/ego_dt (ms) | buckets | cats | cam | file |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    html.append("<h2>2. v2_vs_v3 — 보정 감사 ({})</h2>".format(len(v2v3_items)))

    def fmt(v):
        return "n/a" if v is None else f"{v:.2f}"
    for it in v2v3_items:
        md.append(f"| {it['scen']} | {it['frame_id']} | {it['track_id']} | {it['class']} | "
                  f"{it['corr_dist']:.3f} | {fmt(it['rel_speed'])}/{fmt(it['obj_speed'])}/{fmt(it['ego_speed'])} | "
                  f"{it['obj_dt_ms']:.0f}/{it['ego_dt_ms']:.0f} | {','.join(it['buckets'])} | "
                  f"{','.join(it['categories']) or '-'} | {it['camera']} | `v2_vs_v3/{it['file']}` |")
        html.append(f"<div class='cap'>{it['scen']} f{it['frame_id']} t{it['track_id']} "
                    f"{it['class']} — corr_dist={it['corr_dist']:.3f}m, "
                    f"rel/obj/ego={fmt(it['rel_speed'])}/{fmt(it['obj_speed'])}/{fmt(it['ego_speed'])} m/s, "
                    f"buckets=[{','.join(it['buckets'])}] cats=[{','.join(it['categories']) or '-'}]</div>"
                    f"<img src='v2_vs_v3/{it['file']}'>")

    # 3. boundary
    md.append("\n## 3. boundary_50m/ — 50m 경계 감사 (13장)\n")
    md.append("`pretrain_verify/g_membership_audit.json`의 13개 경계교차 박스(12 v2_only + 1 v3_only). "
              "50m 링 기준 v2 중심(빨강)·v3 중심(초록)과 radial 값 표기.\n")
    md.append("| # | scene | frame | track | kind | v2_radial | v2_side | v3_radial | v3_side | file |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    html.append("<h2>3. boundary_50m — 50m 경계 (13)</h2>")
    for i, b in enumerate(boundary, 1):
        md.append(f"| {i} | {b['scen']} | {b['frame_id']} | {b['track_id']} | {b['kind']} | "
                  f"{b['v2_radial']:.3f} | {b['v2_side']} | {b['v3_radial']:.3f} | {b['v3_side']} | "
                  f"`boundary_50m/{b['file']}` |")
        html.append(f"<div class='cap'>{b['scen']} f{b['frame_id']} t{b['track_id']} {b['kind']} — "
                    f"v2={b['v2_radial']:.3f}m({b['v2_side']}), v3={b['v3_radial']:.3f}m({b['v3_side']})</div>"
                    f"<img src='boundary_50m/{b['file']}'>")

    # 4. temporal
    md.append("\n## 4. track_temporal/ — 트랙 시계열 감사 (9장)\n")
    md.append("scene별 mean corr_dist 상위 3개 트랙. 패널: v2/v3 x | v2/v3 y | corr_dist | rel_speed. "
              "빨간 점선=플래그된 점프(>25m/s & >3m). reversals는 위치/보정 진동 지표.\n")
    md.append("| scene | track | rank | n_frames | mean_corr | median_corr | flagged_jumps | rev x/y/corr | file |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    html.append("<h2>4. track_temporal — 트랙 시계열 (9)</h2>")
    for t in temporal:
        jumps = ",".join(str(j) for j in t["flagged_jump_frames"]) or "none"
        md.append(f"| {t['scen']} | {t['track_id']} | {t['rank']} | {t['n_frames']} | "
                  f"{t['mean_corr_dist']:.3f} | {t['median_corr_dist']:.3f} | {jumps} | "
                  f"{t['v3_reversals_x']}/{t['v3_reversals_y']}/{t.get('corr_dist_reversals','?')} | "
                  f"`track_temporal/{t['file']}` |")
        html.append(f"<div class='cap'>{t['scen']} t{t['track_id']} rank{t['rank']} — "
                    f"mean_corr={t['mean_corr_dist']:.3f}m, jumps={jumps}, "
                    f"rev x/y/corr={t['v3_reversals_x']}/{t['v3_reversals_y']}/{t.get('corr_dist_reversals','?')}</div>"
                    f"<img src='track_temporal/{t['file']}'>")

    # observation points
    obs = """
## 관찰 포인트 (사용자 확인 필요)

아래는 **사람이 눈으로 직접 확인**해야 하는 항목입니다. 시각화는 수치 검증을 **대체하지 않습니다**.
본 산출물은 어떤 항목에 대해서도 **PASS를 선언하지 않습니다** — 관찰 목록일 뿐입니다.

### depth (1)
- depth 오버레이(우측)가 정적 구조물(도로 경계/차선/건물 외벽)과 픽셀 단위로 정합하는지. 근거리=보라, 원거리=빨강 순서가 지형과 맞는지.
- 이동 객체(보행자/차량) 표면에 depth 포인트가 실제로 얹히는지, 아니면 배경으로 새는지.
- (주의) depth_gt는 704×256 학습 좌표계로 생성되어 있어, 여기서는 (1600/704, 900/256)배로 확대해 겹쳤습니다. 확대 정합이 맞는지 육안 확인 필요.

### v2_vs_v3 (2)
- v3(초록) 박스가 이동 객체를 v2(빨강)보다 **이동/보정 방향으로** 옮겼는지, 그리고 그 방향이 파랑(객체 속도)·노랑(ego 속도)과 물리적으로 일관적인지.
- 흰 화살표(v2→v3 변위) 크기가 corr_dist 및 obj_dt/ego_dt(latency)와 상식적으로 비례하는지.
- **정지 객체(obj_speed≈0)인데 corr_dist가 큰 사례**(예: `scen77_f439_t5`, 보행자 obj_speed=0 이지만 0.79m 보정)가 ego 재투영으로 설명되는지, 과보정은 아닌지.
- 카메라 투영에서 초록/빨강 큐보이드가 실제 객체 외곽을 감싸는지(측면 카메라 포함), 심하게 어긋나면 투영/외부파라미터 문제일 수 있음.

### boundary_50m (3)
- 50m 경계 박스가 링 안/밖으로 갈리는 것이 맞는지: 12개 v2_only는 v2가 안(≤50)·v3가 밖(>50), 1개 v3_only는 반대인지.
- v2·v3 중심 차이가 sub-decimeter 수준인데 그 미세 이동만으로 필터 membership이 바뀌는 경계 민감성이 학습에 의미가 있는지.

### track_temporal (4)
- corr가 큰 track에 **물리적으로 불가능한 순간이동/진동**이 없는지: 위치 패널(x,y)의 v2/v3 궤적이 매끄러운지(현재 플래그된 점프 0건).
- corr_dist 패널의 **프레임 간 진동**(reversals corr 36~74)이 상대속도(rel_speed)와 연동되는 정상적 latency 효과인지, 아니면 라벨 노이즈인지.
- rel_speed 곡선이 궤적(멀어짐/다가옴)과 일관적인지.

### 공통
- 본 감사는 시각적 정합성만 확인합니다. 정량 검증(수치 기반 GT 정확도/보정 통계)은 별도 스크립트로 수행해야 하며, 여기서는 **어떤 PASS/합격 판정도 내리지 않습니다.**
"""
    md.append(obs)
    html.append("<h2>관찰 포인트 (사용자 확인 필요)</h2><pre style='white-space:pre-wrap'>"
                + obs.replace("<", "&lt;") + "</pre>")

    with open(os.path.join(HERE, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(HERE, "index.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print("wrote README.md and index.html")
    print(f"totals: depth={len(depth)} v2_vs_v3={len(v2v3_items)} "
          f"boundary_50m={len(boundary)} track_temporal={len(temporal)}")


if __name__ == "__main__":
    main()
