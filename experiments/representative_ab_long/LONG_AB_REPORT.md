# 장기(수렴) 대표 3-fold × 2-seed A/B — budget 동결 & 실행 계획

> ## ⛔ 상태: 장기 A/B = **NOT STARTED (미착수)**
> - **TECHNICAL_GO(infra) = PASS** — runner/anchor/init/collision-guard/env plumbing 검증 완료.
> - **V3_PERFORMANCE_GO = INCONCLUSIVE (변동 없음)** — 장기 run 이 아직 없어 파일럿 판정을 대체하지 못함.
> - **PRODUCTION_150_GO = NO-GO** · **P1 = NO-GO** (불변).
> - **미착수 사유**: 아래에서 **실측**한 production 기본 schedule(100 epoch, epoch마다 full validation)의
>   12-run 예상 비용이 **≈ 503 GPU-시간 ≈ 20.9 GPU-일**(단일 RTX 4070 SUPER)이다. 이는 단일 세션에서
>   완주·평가가 불가능하고, GPU 를 여러 날 점유하는 되돌리기 어려운 커밋이다. 규정(Phase 2.4)에 따라
>   **임의로 budget 을 축소하지 않고**, 정확히 계산해 보고하고 **여기서 멈춘다**. validation 전략 변경은
>   budget 변경이므로 **사용자 승인(Phase 2.5) 후에만** 진행한다.

---

## 1. Production 기본 schedule (train.py 실측 기본값)
| 항목 | 값 | 근거 |
|---|---|---|
| NUM_EPOCHS | **100** | `train.py:2100` `int(os.environ.get('NUM_EPOCHS','100'))` |
| batch / grad-accum / effective | 4 / 2 / **8** | `train.py:2101-2102` |
| validation cadence | **매 epoch 끝**(full) | `VAL_EVERY_STEPS` 기본 0 → epoch 기반; `FAST_VAL_MAX_FRAMES` 기본 0 → full |
| early-stop patience | 10 (기본) → **A/B 는 999999 로 비활성** | `train.py:2105`; baseline/candidate update 수 고정 위해 |
| periodic checkpoint | **10 epoch 마다** `checkpoint_epoch{N}.pth` | `train.py:3019-3022` (100ep → 10개/run) |
| best checkpoint | primary softcalibrated f1@0.15 / raw f1@0.25 / val_loss (각 fixed+epoch-tagged, epoch-tagged 는 교체) | `save_best_with_epoch` `train.py:2056-2068` |
| final weights | `morai_autonav_weights.pth` | `train.py:3032` |

- **budget 은 이 production 기본값(100 epoch, full train loader, MAX_STEPS 미사용)으로 동결**한다.
  `MAX_STEPS_PER_EPOCH` 로 데이터를 자르지 않는다. fold 간 frame 수 차이로 epoch당 update 수가 다른 것은
  아래 §3 에 기록한다. **같은 fold 의 baseline/candidate 는 동일 frame → update 수 정확히 동일**하다.

## 2. 실측 timing (RTX 4070 SUPER, depth OFF, batch 4)
FoldC 구성(train scen05+scen77=1916f / val scen144=1421f)에서 1-epoch 짧은 run 을 warm 상태로 차분 측정.
| 측정 | 값 | 방법 |
|---|---|---|
| **per dl-step** | **0.611 s** | warm 차분 (D:120step,val8 = 91s) − (Cbis:30step,val8 = 36s) = 55s / 90step |
| **per val-frame** | **≈ 1.05 s** | (B:90step,val300 = 388s, warm) vs Cbis → 300-frame val ≈ 315s → 1.05 s/frame |
| startup(warm) | ≈ 18 s/run | Cbis 36s − 30×0.611; run 1회당 1번(100 epoch 에 상각 → 무시) |
- ⚠️ **validation 이 지배적 비용**: `train.validate()` 는 0.01/0.03 저-threshold NMS sweep 을 포함해 CPU-bound
  (파일럿 §9 와 동일 현상). **full val(1421 frame) ≈ 25 분/epoch.** 100 epoch → **≈ 41 시간(FoldC)**.
- cold-cache 첫 run(A:30step,val300 = 438s)은 warm B(388s)보다 느려 차분에서 제외(디스크/CUDA 예열 상각).
- 최종 full-val 은 `eval_run.py`(single-forward, 보고 threshold 한정)로 **빠르게** 수행(저-thr sweep 생략).

## 3. fold별 규모 (frame / step / update, 100 epoch)
| fold | val | train scenes | trainF | step/ep | upd/ep | step/run | **upd/run** | train시간/run |
|---|---|---|---|---|---|---|---|---|
| A | scen05 | scen77+scen144 | 2780 | 695 | 347 | 69,500 | **34,700** | 11.8 h |
| B | scen77 | scen05+scen144 | 1978 | 495 | 247 | 49,500 | **24,700** | 8.4 h |
| C | scen144 | scen05+scen77 | 1916 | 479 | 239 | 47,900 | **23,900** | 8.1 h |
- update 수는 fold 마다 다르나(기록함), **fold 내 baseline=candidate 동일**.

## 4. 12-run budget (train + monitoring-val) — 3 시나리오
| 시나리오 | monitoring val | per-run(A/B/C) | per-seed(6run) | **12-run(2seed)** | GPU-days |
|---|---|---|---|---|---|
| **S1 = production 기본** | full, **매 epoch** | 28.0 / 48.0 / 49.6 h | 251 h | **503 h** | **20.9** |
| S2 (파일럿式) | subset 150f, 매 epoch | 16.2 / 12.8 / 12.5 h | 83 h | **166 h** | 6.9 |
| S3 | subset 150f, 10 epoch마다 | 12.2 / 8.8 / 8.6 h | 59 h | **119 h** | 4.9 |
- **최종 판정용 full v3 validation(Phase 5)** 은 세 시나리오 모두 종료 후 `eval_run.py` 로 수행(빠름, 위 비용에 미포함, run당 수 분).
- **storage**: ≈ 3.0 GB/run(periodic 10 + best-set + last 0.43G + final 0.15G) × 12 = **≈ 36 GB**. 여유 871 GB → OK, 무제한 생성 없음.

## 5. 핵심 결정 사항 (사용자 승인 필요)
production **기본** schedule(S1)은 epoch마다 full validation 을 돌려 **≈ 21 GPU-일**이 된다(비현실적).
파일럿은 이를 **subset monitoring(150f) + 종료 후 full eval** 로 회피했다(정당한 production 관행). 그러나
이는 "매 epoch full validation" 이라는 기본값에서 벗어나므로 **budget/방법론 변경 → 승인 대상**이다.
아래 중 하나를 승인해 주세요(임의 선택하지 않음):
- **(a) S1 그대로** (full val 매 epoch, ≈21 GPU-일) — 가장 충실하나 매우 김.
- **(b) S2** (subset150 monitoring 매 epoch + 종료 후 full eval, ≈7 GPU-일) — 파일럿과 동일 방법.
- **(c) S3** (subset150 monitoring 10 epoch마다 + 종료 후 full eval, ≈5 GPU-일) — 가장 실용적.
- **(d) 보류** — 파일럿 + 이 budget 문서만 인수인계, 장기 run 은 동료 PC/후속 세션에서.

## 6. 실행 runner (검증 완료, 미실행)
`experiments/representative_ab_long/run_ab_long.sh`
- **파일럿과 완전 분리된 root** `experiments/representative_ab_long/` — 파일럿 checkpoint 를 읽지 않음.
- seed 격리: `seed0/`, `seed1/` (모델/학습 seed 만; **anchor seed 는 항상 42**, seed 무관 → fold당 1회, 두 seed 공유).
- baseline(train v2/val v3) · candidate(train v3/val v3), fold별 공유 anchor(K=900) + seed별 공유 init warm-start.
- **collision guard**: run 별 `run_meta.json`(seed/fold/role/train_gt/val/anchor_sha/init_sha/num_epochs/commit)
  과 의도가 불일치하면 **fail-fast**(자동 덮어쓰기·오탑재 resume 방지). resumable(guard 통과 시 last_checkpoint 재개).
- 검증: `bash -n` OK · guard match→resume / mismatch→fail-fast OK · train.py env 인터페이스는 파일럿과 §2 smoke 로 입증.

### 실행 명령 (승인된 시나리오의 env 로)
```bash
cd /home/autonav/projects/morai-3d-detection
# 예: S2(파일럿式) seed 0 → 완료 후 seed 1
AB_SEED=0 AB_EPOCHS=100 AB_MAX_STEPS=0 AB_VAL_MON=150 \
  bash experiments/representative_ab_long/run_ab_long.sh
AB_SEED=1 AB_EPOCHS=100 AB_MAX_STEPS=0 AB_VAL_MON=150 \
  bash experiments/representative_ab_long/run_ab_long.sh
# S1(기본, full val): AB_VAL_MON=0 · S3: (train.py 는 epoch 기반이라 10-epoch 간헐 val 은 후속 소폭 수정 필요)
# 각 run 종료 후 eval_metrics.json(full v3) 이 run_dir 에 생성됨. 집계는 Phase 5 절차.
```
- 각 run provenance(Phase 4): run_meta.json + run_config.json(train.py) + eval_metrics.json 에
  code commit / clean-worktree / seed / fold / role / GT versions / anchor_sha / init_sha / epochs·updates(target/actual)
  / ckpt sha / depth OFF / grid_sample fallback 기록.

## 7. 판정 및 해제 조건
- **TECHNICAL_GO(infra) = PASS** (runner/guard/plumbing).
- **V3_PERFORMANCE = INCONCLUSIVE (파일럿 그대로)** — 장기 12-run 완주 + Phase 5 집계 전에는 재판정 불가.
- 해제(장기 A/B 착수) 조건: **§5 시나리오 승인** + **다일(多日) GPU 점유 승인**. 그 후 위 명령으로 seed0→seed1 실행,
  완주 시 Phase 5(집계/판정) → 본 문서를 결과로 갱신.
- **pass tag(`v3-source-time-representative-pass-v1`) 는 생성하지 않았다** — 성능 GO 확정 + 사용자 최종 승인 후 별도 단계.
