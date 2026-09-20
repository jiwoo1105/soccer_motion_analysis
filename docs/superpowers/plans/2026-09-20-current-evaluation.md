# Current evaluation publication plan

**Goal:** Publish the measured September 20 workflow and update the soccer portfolio pages without changing the experimental conclusions.

**Architecture:** `evaluate.py` is the current cache-based entry point. `scoring/current_metrics.py` computes trunk/head/body measurements on original source-frame indices. `redetect_balls.py` generates independent ankle-ROI detector candidates. Older XZ/SAM2 tools remain explicitly historical.

**Constraints:** Preserve input videos, pose/ball caches and unrelated portfolio content. Publish no private videos, pose tracks, credentials or local agent settings. Use the existing 13-clip cohort with whole-video exclusions 6-1 and 7-3. Equal weights are provisional; no grade fitting. Source frame counts, confidence masks, detector settings and frozen calibration must be explicit.

- [x] Add invariant and missing-data tests before extracting numeric functions. Interfaces: `load_pose(path, source_frames) -> dict`, `body_alignment(pose, window=9) -> dict`, `trunk_angle(pose) -> float | None`, `analyze_headup(pose, candidates) -> dict`, `score_measurements(row, calibration) -> dict`.
- [x] Extract the existing measured numeric pipeline without imports from ignored `output/` reports. Retain source-frame gaps and SG support, define absent totals as null, and publish a manifest plus frozen calibration.
- [x] Add the independent ROI detector CLI and tests. It emits the existing `*_candidates.json` schema and preserves all detections and original pixel coordinates.
- [x] Run all 11 retained local clips through the published entry point and compare full-precision measurements, 45 accepted event frames and scores with the saved September 20 result. Reproduce 5/9/13 sensitivity and weight comparisons.
- [x] Rewrite README and current-method documentation; label historical code/docs and archive selected prior diagnostics. Publish only aggregate results and diagrams, with sample sizes, exclusions and validation limits.
- [x] Correct portfolio pp4–5, retain its design, and render/inspect all pages using bundled LibreOffice with system Korean fonts. Other blocks and package assets must be unchanged.
- [ ] Review changed code and staged content, run numeric/unit/CLI checks, then commit and push the reviewed branch to the authorized GitHub main branch using a normal fast-forward push. Verify the remote revision and document links.

## Verification completed before publication

- 55 unit tests pass; full retained cohort matches the archived numeric/event snapshot.
- ROI geometry matches all 2,853 source frames; 3-1 real detector/pose smoke matches existing caches.
- Portfolio rewritten again in plain language following user feedback; all 11 pages rendered, soccer pages 4–5 reviewed, other blocks/assets unchanged.
- Numeric reviewer findings resolved: missing body support, valid timing-shift support, missing-value snapshots and scientific metadata.
- Normal push and remote revision verification are the final steps after this commit.
