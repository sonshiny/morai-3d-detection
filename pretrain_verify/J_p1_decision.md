# Task J — P1 (anisotropic regression loss) decision

**Decision: P1 = NO-GO (this stage).** P1 anisotropic loss is NOT implemented, and the data
does not yet justify or allow validating it.

## Basis (P3 relative-speed distribution, v3 GT, ±3 local-linear-regression ego velocity)

| bin (m/s) | n_gt | n_tracks | n_frames | amount_removed p50 | p95 | max |
|-----------|-----:|---------:|---------:|-------------------:|----:|----:|
| [0,1)     | 3482 | 15 | 1773 | 0.0143 | 0.1324 | 0.5479 |
| [1,3)     | 2017 | 19 | 1033 | 0.0643 | 0.2675 | 0.7981 |
| [3,6)     | 4836 | 42 | 2335 | 0.1407 | 0.3495 | 0.9088 |
| [6,8)     |  172 |  7 |  146 | 0.2080 | 0.5355 | 0.7647 |
| [8,inf)   |    9 |  1 |    9 | 0.3029 | 0.5080 | 0.6117 |
| total     |10516 |    |      |        |        |        |

`amount_removed` = per-box v2→v3 correction distance (corr_dist), i.e. the **time-contamination
that P4 removed**. It is explicitly NOT the v3 residual noise floor.

## Why NO-GO

1. **High-speed regime is severely undersampled — no track diversity.**
   - `[8,inf)`: **9 boxes but only 1 track across all 3 scenes, 9 frames.** A single object.
     No confidence interval is meaningful; nothing about ≥8 m/s behavior can be estimated.
   - `[6,8)`: 172 boxes but only **7 tracks**. Boxes are temporally correlated (the same 7 objects
     over 146 frames), so the effective independent sample size ≈ 7, not 172.
   P1's whole value proposition is better handling of fast objects; the dataset has essentially
   no fast objects to fit or validate against.

2. **We have NOT measured a speed-dependent v3 *residual* error.**
   `amount_removed` grows with relative speed (p50 0.014 → 0.303 m across bins), which only shows
   P4 corrected *more* where relative motion was larger — expected, and it is the removed error,
   not what remains. Justifying P1 requires evidence that v3 still has *residual* speed-dependent
   noise after correction; that measurement does not exist here (it needs an independent
   higher-rate reference, not corr_dist).

3. **Preflight recall by speed bin is not usable as evidence.** The preflight model is a
   ~100-step warm-up (recall ≈ 0), so per-bin recall cannot rank models or reveal a speed effect.

## What would flip P1 to reconsider
- A direct measurement of v3 residual center/velocity error vs. relative speed against an
  independent reference (not corr_dist), showing the residual actually grows with speed.
- Many more high-speed **tracks and scenes** (≥8 m/s currently = 1 track) so any high-speed
  metric has a usable confidence interval.

Until both exist, P1 stays NO-GO. Matcher, focal, quality, decoder, temporal GNN, anchor count,
and the 2.0 m recall threshold were **not** changed.
