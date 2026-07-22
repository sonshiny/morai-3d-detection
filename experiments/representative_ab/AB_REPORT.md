# 대표 3-scene source-time v3 A/B 리포트

> 🧪 **이 리포트는 PILOT (고정 300-update / 2-epoch·미수렴) 결과다. 이 결과만으로 성능 우열(V3_PERFORMANCE)을
> 결론내릴 수 없다.** 성능 우열을 추가로 요구할 때만 별도의 production-schedule 장기 A/B
> (`experiments/representative_ab_long/`, seed 0/1, 100-epoch full schedule)를 수행한다.
> v3 label correctness는 학습과 분리된 `experiments/v3_correctness/V3_CORRECTNESS.md`의 직접 측정으로
> 판정한다. 아래 수치·checkpoint·metric 은 파일럿 산출물이며
> **덮어쓰거나 삭제하지 않고 보존**한다. 장기 A/B 는 이 파일럿 checkpoint 를 resume 하지 않는다(각 seed 의
> shared init 에서 새로 시작). 파일럿의 유효 결론은 **TECHNICAL_GO=PASS(파이프라인 정합·안정)** 까지다.

> **판정 요약: TECHNICAL_GO = PASS · V3_PERFORMANCE_GO = INCONCLUSIVE(PILOT, 미수렴) · PRODUCTION_150_GO = NO-GO.**
> 6개 run 전부 안정·정합했으나(파이프라인 GO), 고정 300-update budget 에서 v3−v2 성능 delta 는
> 작고(macro Δf1=−0.008) 지표·fold 간 방향이 엇갈려 성능 우열을 확정할 수 없다. P1 은 NO-GO 유지.

## 0. 반드시 먼저 읽을 것 (실험 성격)
- 이 formal A/B 는 이전 120-step preflight 와 **다른 실험**이다(fold별 공유 anchor·공유 init·동일 seed·
  동일 sampler·**동일 optimizer update 수(300)**).
- **validation 은 모두 v3 GT**(baseline 도 v3 val), production filter 이후 membership.
- **primary = USE_DENSE_DEPTH=0(depth OFF).** LiDAR–camera 동적 depth 동기화 미해결 → label-correction
  효과를 depth 혼입 없이 격리. depth ON 은 코드경로 확인용으로만 남긴다.
- **v3 는 source-time 보정 GT 이며 외부 독립 측정 absolute truth 가 아니다.** delta 는 v2 대비 상대 효과.
- **P1 은 판단하지 않았다**(고속 표본 부족 NO-GO). matcher/architecture/anchor 수/2.0m 불변.
- **150-scene production 은 시작하지 않았다.**
- ⚠️ 계산 제약으로 full 100-epoch schedule 이 아니라 **고정·정합 budget(모든 run 300 update)** 로
  학습했다. 성능 delta 는 **수렴 결과가 아니라 방향/안정성 지표**다.

## 1. 실험 설계
- 3-fold scene-level: FoldC(train scen05,scen77 / val scen144), FoldB(train scen05,scen144 / val scen77),
  FoldA(train scen77,scen144 / val scen05).
- fold별 v3 anchor 1회(K=900,seed42) → baseline/candidate 공유. 공유 init(seed0,fold anchor)를 warm-start.
- baseline: TRAIN_GT=v2/VAL_GT=v3. candidate: TRAIN_GT=v3/VAL_GT=v3.
- 고정 budget: 2 epoch × 300 dl-step = **600 dl-step = 300 optimizer update/run**, early-stop 비활성.
  seed=0. run별 RUN_DIR 완전 격리. 실행: `experiments/representative_ab/run_ab.sh`.

## 2. run provenance (6 runs) — 정합 확인
git HEAD `67dbaf0e8d`, depth=OFF, seed=0, batch 4×2, **update=300(전부 동일)**, val=v3(전부).
| fold | anchor SHA(공유) | init SHA(공유) | train | val | base ckpt | cand ckpt |
|---|---|---|---|---|---|---|
| C | aef60c9d786d | 97d5f185dfe3 | scen05,scen77 | scen144 | ca9b751d | 64ee8ce4 |
| B | 62ffc533a622 | 4cba202458d4 | scen05,scen144 | scen77 | 6fff8ebd | c5c95382 |
| A | e15ea7a1d251 | 68c6dcf1c809 | scen77,scen144 | scen05 | b33603f1 | 77cda2d4 |
- fold 안에서 baseline/candidate 의 **anchor SHA·init SHA·train/val·update 수 동일**(정합 OK).
- 유의: FoldC baseline 의 working-tree dirty hash(`4f39d53c`)만 나머지(`ba42033f`)와 다르다 — FoldC baseline 은
  1차 오케스트레이션에서 학습됐고 그 뒤 eval_run.py/문서/.gitignore 만 수정됐기 때문. **학습 코드
  (train.py·모델·loss·make_kmeans·anchor)는 6 run 전부 동일**(train.py 는 1차 실행 전 확정, 이후 미수정).

## 3. 전체 validation 결과 (fold별, 전체 val frame)
soft@0.15 = softcalibrated(quality^0.5)@score≥0.15, recall_thr 2.0m. cdist=matched center distance(m).
| fold | role | val_loss | soft P/R/F1 | raw F1 | cdist mean/med/p95 | veh F1 | ped F1 |
|---|---|---|---|---|---|---|---|
| C(1421f) | base | 1.589 | .239/.209/**.223** | .225 | 1.243/1.257/1.913 | .261 | .071 |
| C | cand | 1.590 | .216/.208/**.212** | .211 | 1.254/1.256/1.957 | .239 | **.105** |
| B(1359f) | base | 1.571 | .147/.350/**.207** | .209 | 1.131/1.135/1.887 | .225 | .015 |
| B | cand | 1.570 | .147/.336/**.204** | .209 | **1.126**/1.123/1.898 | .223 | .018 |
| A(557f) | base | 1.524 | .107/.151/**.126** | .155 | 0.950/0.963/1.835 | .152 | .076 |
| A | cand | 1.529 | .098/.141/**.116** | .148 | 0.968/0.990/1.854 | .137 | .073 |

## 4. filtered production P3 (fold별, 전체 val) — GT-conditioned recall
membership = production validation filter(_load_labels_v2), rel_speed 미매칭 0. 매칭 2.0m.
`[bin] base→cand recall (n_gt)` — corr_dist=amount_removed(제거된 시간오염, ≠residual). 고속 bin 표본부족 경고.
- **FoldC**: [0,1) .382→.356 (n1913) · [1,3) .130→**.147** (n1032) · [3,6) .080→**.095** (n1831) · [6,8) 0/0 (n58) · [8,∞) n0
- **FoldB**: [0,1) .514→.496 (n1047) · [1,3) .455→.422 (n754) · [3,6) .182→.181 (n1389) · [6,8) .113→.113 (n62) · [8,∞) 0/0 (n9)
- **FoldA**: [0,1) .484→.400 (n95) · [1,3) .500→.533 (n60) · [3,6) .091→.086 (n811) · [6,8) .077→.077 (n52) · [8,∞) n0
- ⚠️ [6,8)·[8,∞) 는 표본 극소(전 fold n≤62, [8,∞) 는 FoldB 만 n9) → **이 구간으로 우열/P1 판단 금지.**

## 5. paired delta 및 집계 (candidate − baseline) — `ab_summary.json` / `paired_delta.csv`
| fold | Δf1(soft@.15) | Δrecall | Δcdist(box boot, 95%CI) | Δcdist 저속[0,1)(CI) | Δval_loss |
|---|---|---|---|---|---|
| C | −0.011 | −0.001 | **+0.049 [0.037, 0.060]** | (저속 worse) | +0.001 |
| B | −0.002 | −0.014 | −0.004 [−0.012, 0.004] | ~0 | −0.001 |
| A | −0.010 | −0.010 | +0.011 [−0.016, 0.039] | ~0 | +0.005 |
| **macro** | **−0.008** | **−0.008** | **+0.018** | **+0.033** | +0.002 |
- **box-level paired bootstrap**(공통 track_id pairing, 3000 resample). fold 부호 일치성: Δf1 signs=[−1](3 fold 모두 v3 소폭 낮음).
- 그러나 **cdist 방향은 fold 간 엇갈림**(B 는 v3 가 오히려 소폭 개선, C/A 는 악화; C 만 CI 가 0 제외).
- **ped(보행자) F1 은 FoldC 에서 v3 개선**(.071→.105), 중속 P3 recall 도 FoldC 에서 v3 소폭 개선.
- pooled 아님. macro 평균·fold별·paired 를 모두 제시. n_tracks/scene 작은 구간 CI 과신 금지.

## 6. 판정 (세 축 분리)
### TECHNICAL_GO = **PASS**
6 run 전부: NaN/Inf/OOM 없음(전부 완주, loss finite), provenance 정합(fold 내 anchor/init/seed/update 동일),
P3 join>0(gt/pred 전부 >0, membership=filter), val 항상 v3, checkpoint save/resume·run 격리·cross-run
fail-fast 동작. (FoldC baseline dirty-hash 차이는 학습 후 eval/문서 수정 때문이며 학습 코드는 동일.)
### V3_PERFORMANCE_GO = **INCONCLUSIVE**
- overall F1 은 3 fold 모두 v3 가 **소폭 낮음**(macro −0.008)이나 크기가 작고, **center distance 는 fold 간
  방향이 엇갈리며**(B 개선/C·A 악화), **ped F1·중속 P3 recall 은 FoldC 에서 v3 개선** 등 지표가 상충한다.
- delta 크기(≈0.003–0.05)가 300-update 미수렴·GPU 재실행 변동 범위와 비슷해 신호/잡음 분리가 안 된다.
- 저속 clean-control([0,1)) 에서 v3 가 체계적으로 개선된다는 근거도, 명확히 악화된다는 근거도 없다(FoldC cdist
  만 유의 악화). → **GO 선언하지 않는다. NO-GO 도 아니다(미수렴·소표본).**
- **필요 후속(근거 있음): (1) seed=1 paired repeat**(현 delta 가 재실행 변동보다 작음 — 규정상 반복 권고),
  **(2) 수렴까지(또는 훨씬 긴 matched budget) 재학습** 후 재판정. 명령은 §8.
### PRODUCTION_150_GO = **NO-GO** (항상)
150-scene 전체 anchor/preflight 부재. depth 동적 동기화 미해결. 별도 절차(HANDOFF_150_SCENES.md).

## 7. seed 반복 필요성 — **필요(권고)**
Δf1 macro=−0.008, fold별 |Δ|≤0.011 로 preflight 재실행 변동(|Δloss|≈0.06 수준)과 구분이 어렵다. 규정
("차이가 GPU 재실행 변동보다 작으면 seed=1 paired repeat")에 따라 **seed=1 반복 필요**. 임의 단일 run 으로
결론내지 않았다. 시간 제약으로 이번엔 seed=0 만 실행하고 seed=1 은 후속으로 명시한다.

## 8. 재현/재개/후속 명령
```bash
# 재현(seed 0):
AB_EPOCHS=2 AB_MAX_STEPS=300 AB_SEED=0 bash experiments/representative_ab/run_ab.sh
python3 experiments/representative_ab/aggregate.py --boot 3000
# 권고 후속: seed=1 반복(공유 init/anchor 는 seed 로 분리 저장됨)
AB_EPOCHS=2 AB_MAX_STEPS=300 AB_SEED=1 bash experiments/representative_ab/run_ab.sh
# 개별 run 재개: RUN_DIR=<fold>/<role> RESUME=auto … python3 train.py (cross-run 정합 검증됨)
```

## 9. 실패·중단 기록
- 1차 오케스트레이션에서 full-val eval 이 train.validate 의 저-threshold NMS sweep 으로 CPU-bound(매우 느림).
  → eval_run.py 를 production 매칭 함수 기반 단일-forward·보고 threshold 한정으로 교체(수치는 동일 함수·동일
  threshold 에서 일치), 오케스트레이터를 **resumable**(checkpoint/eval 재사용)로 만들어 FoldC baseline 학습을
  재사용하고 재실행. 그 외 NaN/Inf/OOM/crash 없음.

## 10. 산출물
`ab_summary.json`, `paired_delta.csv`, fold별 `*/run_config.json`·`*/eval_metrics.json`(ckpt_sha256 포함),
fold anchor `*/anchor/anchor_kmeans_meta.json`. checkpoint(.pth)·init(.pth)·training log 는 Git 제외.
