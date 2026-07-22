# Git 배포 staging 계획 (사용자 승인 전 commit/tag/push 하지 않음)

> **아직 아무것도 commit/tag/push 하지 않았다.** 이 문서는 승인 시 실행할 계획이다.
> main 직접 commit 금지, force push 금지, history rewrite 금지.

## Branch / Tag / Remote
- 새 branch: `source-time-v3-handoff` (main 에서 분기)
- release tag(예정): `v3-source-time-representative-pass-v1`  — **대표 A/B 판정이 TECHNICAL_GO
  이고 V3_PERFORMANCE 가 최소 INCONCLUSIVE-이상일 때만.** NO-GO 면 tag 보류.
- remote: `origin`

## 포함(track) 대상 — 60→54개 후보(모두 <5MB, 바이너리 없음)
- production/eval/audit 코드: `train.py, make_kmeans.py, anchor_generator.py, morai_dataset.py,
  loss_calculator.py, correct_source_time.py, eval_relative_speed.py, g_membership_audit.py`
- 테스트: `test_source_time_correction.py, test_velocity_valid_loss.py, test_anchor_policy.py,
  test_gt_version_split.py, test_relative_speed_eval.py`
- 문서: `pretrain_verify/FINAL_REPORT.md, pretrain_verify/J_p1_decision.md,
  experiments/representative_ab/AB_REPORT.md, HANDOFF_150_SCENES.md, .gitignore`
- 작은 manifest/JSON: `pretrain_verify/{pretrain_manifest.json, preflight_report.json,
  g_membership_audit.json, p3_gt_distribution_v3.json, a_freeze_data.json, audit_current.json}`
- 재현 스크립트: `scripts/*.sh, scripts/audit_150_dataset.py,
  experiments/representative_ab/{run_ab.sh, build_init.py, eval_run.py, aggregate.py}`
- A/B 산출(작은 것만): `experiments/representative_ab/ab_summary.json, paired_delta.csv,
  <fold>/<role>/{run_config.json, eval_metrics.json}, <fold>/anchor/dataset_split.json`
- visual_audit: 생성 스크립트 + `README.md` + `index.html` + `*/_manifest.json` (PNG 제외)

## 제외(ignore) 대상 — .gitignore 로 보장
- `dataset/`, depth_gt, `labels_3d_v3` 산출물(데이터 별도 채널)
- `*.pth, *.pt`(모든 checkpoint/init), anchor `*.npy` + `anchor_kmeans_meta.json` + `dataset_split_*.json`
- `runs/`, `experiments/**/*.pth`, `experiments/**/*.log`, `experiments/**/training_curves.png`
- 세션 scratch: `experiments/_*/`, `pretrain_verify/pf_*/`, `pretrain_verify/_recon.sh`,
  `pretrain_verify/_show_preflight.py`, `experiments/representative_ab/_smoke.sh`, `_tmp*`, `*.log`
- visual_audit PNG 71개: `visual_audit/**/*.png`
- **3-scene anchor(`anchors/…/*.npy,*meta.json`)는 gitignore 대상** — 150-scene production anchor 로
  배포 금지. 동료 PC 는 최종 split 승인 후 재생성(HANDOFF §14).

## 검증(로컬, 실측)
- 크기: add 후보 60→54개, **>5MB 파일 0개**(git add -A --dry-run + stat 확인).
- 로컬 dir 크기: `pretrain_verify` 3.3G(대부분 .pth, ignore), `experiments` 5.2G(대부분 .pth/.log, ignore),
  `visual_audit` 30M(PNG ignore). **force-add 절대 금지.**

## commit 계획 (승인 후, 이 순서)
1. `feat: add source-time v3 training and validation pipeline`
   - train.py(RUN_DIR/GT-split/anchor fail-fast/depth switch/seed), make_kmeans, anchor_generator,
     correct_source_time(epoch-safe), eval_relative_speed, g_membership_audit, morai_dataset(사용자 P4 유지).
2. `test: add pretraining, anchor, and relative-speed verification`
   - test_* 5개 + scripts/audit_150_dataset.py.
3. `docs: add representative A/B report and 150-scene handoff`
   - FINAL_REPORT, J_p1_decision, AB_REPORT, HANDOFF_150_SCENES, scripts/*.sh, 작은 manifest/JSON,
     visual_audit README/index/manifest + 생성 스크립트.

## commit 전 필수 점검(실행 예정, 아직 안 함)
```
git checkout -b source-time-v3-handoff
git add -A --dry-run            # staged 목록 검토(>5MB/바이너리 없음 재확인)
git status --porcelain --ignored | grep '^!!' | grep -E '\.pth|\.npy|dataset/' # ignore 확인
python3 -m pytest -q test_*.py -rs   # dataset-required 는 skip 표시
python3 -m py_compile train.py make_kmeans.py eval_relative_speed.py correct_source_time.py
git diff --check
# clean-clone smoke: 임시 clone 에서 dataset 없이 단위테스트 pass + 데이터 테스트 skip 확인
# 문서 내 명령/경로가 env 기반인지 확인(절대경로 하드코딩 없음)
```
- 위 전부 통과 후에만 사용자에게 최종 승인 요청 → 승인 시 commit, 그 다음 tag/push(force 금지).

## 미해결/유의
- clean clone 에서 dataset-required 테스트가 **조용히 skip → 전체 PASS 착시** 방지: `-rs` 로 skip 사유
  노출, HANDOFF 부록에 unit vs integration 구분 문서화 완료.
- 대표 anchor 하드코딩 테스트(`test_anchor_policy`)는 대표 데이터 없으면 skip(needs_anchor/needs_data),
  순수 단위(`test_gt_version_mapping`, bin/matching/fail-fast)는 데이터 없이 pass.
