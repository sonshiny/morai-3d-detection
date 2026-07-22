# 학습 전 검증 최종 보고 (pretrain verification) — v2 (독립 검토 blocker 반영)

작업 위치: `/home/autonav/projects/morai-3d-detection`
대상: scen05, scen77, scen144 (3,337 frame / 10,516 box). **장시간/full training 미시작, commit/push 안 함.**

> 독립 검토에서 확인된 3개 blocker(P3 join 0건 / P3 membership 오류 / anchor 검사 불충분)를
> 수정하고 재검증했다. 아래 §0 에 blocker 처리 내역을, §3·§6·§8 에 재검증 결과를 반영했다.

---

## 0. 독립 검토 blocker 처리 (재검증 완료)
1. **P3 recall join 0건** — loader stem 이 `"scenario#seg/live_N"` 인데 P3 GT stem 은 `"live_N"` 이라
   frame join 이 0건이었고, scenario 가 `"scen144"` 로 하드코드돼 있었다.
   → preflight 가 `batch["scenario"][i]` 와 base stem(`stem.split("/")[-1]`)으로 키를 만들도록 수정,
   scenario 하드코드 제거. 재실행 결과 **네 config 모두 frames=120, gt=240, pred>0** 로 join 정상.
2. **P3 production membership** — recall 채점은 raw 10516 이 아니라 **MoraiTemporalDataset 의 실제
   validation filter 이후 GT** 를 써야 한다. → `load_filtered_gt_with_relspeed` 추가(=`_load_labels_v2`
   필터 후 GT 에 rel_speed attach; scen144 v3 = **4834 box**, rel_speed 미매칭 0). GT distribution
   audit 10516 은 `--gt-only` 로 분리 유지. **evaluated frame/GT/pred 가 0 이면 fail-fast**(ValueError)
   + 테스트 추가.
3. **anchor 검사 불충분** — 기존엔 파일↔meta SHA 만 봤다. → `anchor_meta_matches_run` 으로 meta 의
   **k, seed, anchor GT version, resolved train/val scenarios, 입력 train-label aggregate hash** 가 현재
   run 과 일치하는지 검사(불일치 시 GPU 이전 fail-fast). make_kmeans `metadata_is_valid` 에 seed +
   input hash 검사를 추가해 **seed/label 이 바뀌면 stale anchor 를 재사용하지 않음**.

**재검증(실제 `train.py` __main__):**
- 3-scene anchor 를 다른 split(val=scen05)에 지정 → **GPU 이전 fail-fast**:
  `train_scenarios / val_scenarios / input_label_sha256` 3건 불일치 출력.
- 올바른 split(val=scen144) → `versioned anchor 검증 통과` 후 학습 진행.
- tampered anchor(1비트) → SHA-256 mismatch fail-fast.

## 1. 변경 파일과 이유
| 파일 | 이유 |
|---|---|
| `train.py` | B(`TRAIN/VAL_GT_VERSION`+fallback), C(split resolve·출력·기록), D(`ANCHOR_DIR` SHA **+ meta 정합(split/k/seed/gt/input-hash) fail-fast**), H(`USE_DENSE_DEPTH` override+`load_depth`), `SEED`. |
| `make_kmeans.py` | D(`--gt-version`/label-dir 인자화, meta 에 input-label SHA·anchor SHA·seed·split 기록, versioned 출력, v3 correction_valid 제외, `anchor_meta_matches_run`, **`metadata_is_valid` stale(seed/input) 방지**). |
| `anchor_generator.py` | D(`ANCHOR_FULL_FILE`/`ANCHOR_XY_FILE` env override; root anchor 미덮어쓰기). |
| `correct_source_time.py`(untracked 확장) | F(`EpochedEgoInterp`: epoch 단위 분리, cross-epoch 보간 차단; 단일 epoch=no-op). |
| `eval_relative_speed.py`(신규) | E/P3(ego ±3 local regression, 상대속도 bin, greedy 2.0m 매칭, **production-filter GT + empty fail-fast**). |
| `g_membership_audit.py`(신규) | G(50m membership 감사). |
| `test_*`(신규 3 + 기존 2 확장) | teleport / anchor(정책·wrong-split·stale) / gt-split / P3(정의·매칭·**empty fail-fast**·filtered membership). |

## 2. 수정하지 않은 것
raw labels / labels_3d_v2 / **labels_3d_v3** / ego_pose / sync_log / images / lidar / depth_gt /
scene_info*.json 전부 불변(`raw_input_invariant_vs_baseline=True`). labels_3d_v3 **재생성 안 함**(F 는
`--report-only` 로 통계 bitwise 동일=no-op 만 확인). loss_calculator.py·morai_dataset.py(사용자 P2/P4)·
root anchor(mtime Jul 19)·P1/matcher/architecture 미변경.

## 3. 테스트 명령과 정확한 출력
```
python3 -m pytest -q test_source_time_correction.py test_velocity_valid_loss.py   → 8 passed
python3 -m pytest -q test_anchor_policy.py test_gt_version_split.py test_relative_speed_eval.py → 16 passed
python3 -m pytest -q (위 5개 파일)                                                  → 24 passed in 4.21s
git diff --check                                                                    → exit 0
```
24 = 좌표/velocity 8 + anchor 7(정책5 + wrong-split/stale-seed 2) + gt-split 2 + P3 7(정의5 + empty fail-fast 1 + filtered-membership 1).
- F no-op: `correct_source_time.py --report-only` 통계가 기존 리포트와 완전 일치(3337f/10516b, all valid).

## 4. hash manifest (`pretrain_verify/pretrain_manifest.json`)
- git HEAD `67dbaf0e8d6c`, diff SHA `e29fdefa579d7e7a…`, diff_check_clean=True.
- counts: v2 3337f/10516b, v3 3337f/10516b, depth_gt 10011(=3337×3).
- **raw_input_invariant_vs_baseline=True**.
- anchor(A/B 공유): gt=v3 k=900 seed=42, full SHA `aef60c9d786d2ba8…`(meta=실측), input-label SHA `c65c2dbd78a6193f…`(1916 파일), train=[scen05,scen77]/val=[scen144], visible 4982.
- preflight 공유 init `782799804d4115c8…`, anchor buffer `fb7205d2708d10de…`.

## 5. split 과 box membership
train=scen05+scen77(1916f), val=scen144(1421f). baseline=train**v2**/val**v3**, candidate=train**v3**/val**v3**, val metric 항상 v3. val/train 로더 (scen,stem) 순서는 GT 버전 무관 동일(test).
**50m default-filter membership** (production `_load_labels_v2`):
| | v2 | v3 | common | v2-only | v3-only |
|---|---:|---:|---:|---:|---:|
| natural kept | **9124** | **9113** | **9112** | **12** | **1** |
기대값 정확 일치, 13건 **전부 50m 경계 crossing**(sub-decimeter). natural(A/B) / common-intersection(진단) 두 mode 지원. scen144 val 필터 후 = **4834**(P3 recall membership).

## 6. relative-speed(P3)
- ego: source-time ego pose 에서 **같은 epoch 안 ±3 local linear regression**(인접 차분 금지, 경계 표본수 기록). object: `R(ego_yaw)·[vx_ego,vy_ego]` m/s. rel = `|v_obj−v_ego|`. 매칭 greedy·class-aware·2.0m(train.py by-distance 규약). **GT-conditioned recall 만**(unmatched pred 별도, precision/F1 아님). corr_dist=**amount_removed**.
- **(A) GT distribution audit — raw 10516**(`p3_gt_distribution_v3.json`):
  | bin(m/s) | [0,1) | [1,3) | [3,6) | [6,8) | [8,inf) |
  |---|---:|---:|---:|---:|---:|
  | n_gt | 3482 | 2017 | 4836 | **172** | **9** |
  | n_tracks | 15 | 19 | 42 | **7** | **1** |
  총 10516. 합성 테스트: bin 경계, ego local-linreg 등속 복원+epoch 미교차, 상대속도 정의, 매칭, unmatched-pred 제외, **empty fail-fast** — 모두 PASS.
- **(B) production recall — validation-filter membership**(preflight, 실제 예측):
  | config | frames | gt | pred | matched | recall [0,1) | recall [1,3) |
  |---|---:|---:|---:|---:|---:|---:|
  | PROD baseline v2→v3 | 120 | 240 | 36 | 3 | 0.051 | 0.000 |
  | PROD candidate v3→v3 | 120 | 240 | 61 | 8 | 0.136 | 0.000 |
  | ISO baseline v2→v3 | 120 | 240 | 341 | 33 | 0.525 | 0.011 |
  | ISO candidate v3→v3 | 120 | 240 | 285 | 29 | 0.441 | 0.017 |
  - membership=production_validation_filter, rel_speed 미매칭 0. **n_gt/pred 모두 >0 → join 정상**(구 버그 해소).
  - ⚠️ preflight val 은 앞 120 frame subsample 이라 scen144 앞부분이 저속 → **[3,6)+ bin GT=0**. 고속 recall 은 이 preflight 로 판단 불가; 고속 신호는 (A) 전체 분포로만 본다. 또한 100-step 모델이라 recall 절대값은 품질 지표 아님(plumbing).

## 7. depth timing — 확인 / 미확인
**확인**: depth_gt 10011개 로드 무오류. ON: fwd/bwd/opt 성공, depth 항 활성(총 loss 5.7→2.0), nonfinite 0, peak ~6.6–7.0 GiB. OFF: 성공(depth 항 없음 → 3.8→1.5), 스위치 실동작 확증. depth_gt (u,v)는 **704×256 학습입력 좌표계**(시각감사에서 발견·보정; `_load_depth_maps` 가 동일 stride 로 소비, 오버레이 노면/차선/보행자 밀착).
**미확인/주의**: LiDAR source clock 이 camera 와 안정 정렬 안 됨(recv gap p95 57.85ms). depth_gt 는 LiDAR 유래 → **동적 객체 capture-time 동기화 미검증**(“capture-time-synchronized dynamic GT” 라 부르지 않음).

## 8. 대표 A/B GO / NO-GO → **GO**
20 warm-up + 100 optimizer step preflight(production 클래스 + 실제 train.py scratch+resume):
| config | upd | loss | grad/param finite | peak GiB | val_loss | soft@0.15 P/R/F1 |
|---|---:|---|---|---:|---:|---|
| PROD baseline v2→v3 (depth ON) | 120 | 5.723→2.040 | T/T | 6.60 | 1.815 | .083/.013/.022 |
| PROD candidate v3→v3 (depth ON) | 120 | 5.727→2.056 | T/T | 7.01 | 1.825 | .131/.033/.053 |
| ISO baseline v2→v3 (depth OFF) | 120 | 3.798→1.493 | T/T | 7.82 | 1.848 | .097/.138/.114 |
| ISO candidate v3→v3 (depth OFF) | 120 | 3.802→1.502 | T/T | 8.22 | 1.862 | .102/.121/.110 |
- **NaN/Inf/OOM 없음**. baseline vs candidate **sampler(batch stem) 순서 동일=True**, 공유 init(763/763; depth OFF 757/763) 동일, **anchor SHA 동일**, **checkpoint save/resume param-identical=True**. P3 production recall 산출(위 §6B).
- 실제 train.py: split 출력·anchor meta(SHA+split/k/seed/gt/input) 검증·depth override·scratch→저장→`재개(start_epoch=2)` 정상. wrong-split/tampered anchor 는 GPU 이전 fail-fast.
- upd/s≈0.74–0.77(grid_sample fallback). 비결정: 동일 seed/init 100-step 재실행 max|Δloss|=**0.0565** → bitwise 동일 주장 안 함.
→ **GO**(동일 조건 확보 + 안정).

## 9. full production training GO / NO-GO → **조건부 NO-GO**
- 대표 3-scene A/B 정식 학습: **GO**. 전체 데이터셋 production: **아직 NO-GO** — 시작 전 필요: ① 전체 split anchor 재생성(현재 anchor 는 3-scene 전용, meta fail-fast 로 강제), ② 전체 split preflight 재실행, ③ depth 동적 사용 시 LiDAR–camera 동기화 정량화.

## 10. 남은 위험
**Blocking(전체 production 전)**: 전체 split anchor 미생성, 전체 split preflight 미수행.
**Non-blocking(모니터링)**: depth 동적 동기화 미검증(aux w0.2); GPU 비결정(|Δloss|≤0.057); P1 고속 표본 부족(§11); preflight P3 고속 bin GT=0(val subsample 저속); grid_sample fallback(느림, 무해); 최대 보정(0.75–0.91m)이 **정지 보행자**에 집중 → 과보정 여부 육안 확인(시각감사).

## 11. P1 결정 → **NO-GO** (`J_p1_decision.md`)
≥8 m/s = 9 box / **1 track**, 6–8 = 172 box / **7 track**. 고속 표본·track·CI 부족. corr_dist 는 제거량(≠residual). 재고엔 독립기준 대비 residual vs 속도 측정 + 다수 고속 track/scene 필요. matcher/focal/quality/decoder/temporal GNN/anchor 수/2.0m 미변경.

## 12. 장시간 학습 미시작 확인
**어떤 full/장시간 training 도 시작하지 않았다.** GPU 작업은 preflight(설정별 120 update)와 train.py 의 소수-step scratch/resume 검증뿐, 전부 NUM_EPOCHS/MAX_STEPS 상한.

---
### 상태 (독립 검토 blocker 반영 후)
- [x] 전체 테스트 통과(24 passed; 기존 7 포함) · git diff --check exit 0
- [x] raw 데이터 hash 불변 · v3 3337f/10516b · depth 10011
- [x] train v2/val v3 & train v3/val v3 실동작(4 config, NaN/Inf/OOM 없음)
- [x] 두 run split·frame order·anchor hash·initial checkpoint 동일 · checkpoint save/resume
- [x] 기존 전체/by-class/by-distance metric 은 P3(별도 모듈)로 불변
- [x] epoch 경계 교차 보간 차단(test) · P3 empty 시 fail-fast(test)
- [x] anchor: wrong-split/stale-label/wrong-seed GPU 이전 fail-fast(실 entrypoint 검증)
- [x] P3 production recall = validation-filter membership(4834), join 정상(gt/pred>0)
- [ ] **전체 production 학습은 NO-GO**(전체 split anchor·preflight·depth 동기화 선행) — 승인하지 않음
