# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added

- Support for the `qwen3.8_27b` model backend (`Qwen/Qwen3.8-27B`).
- CoC labeling over TIER IV T4 scenes: `base_config_vlm_t4` plus `data/t4`,
  front-camera resolution for flattened T4 clip ids, and `T4VectorLoader`,
  which reads ego speed from each scene's `derived/scalars.npz`.
- Lane meta-actions on T4, via `meta_action.t4_lane`: a lanelet2-backed stand-in
  for the trajdata vector map, enabled by `t4_labeling --maps-root`.

### Fixed

- Lane segmentation no longer crashes on a scene where the ego never matches a
  lane; the lateral-offset column is read as float rather than object.

### Changed

- `trajdata` is imported lazily, so datasets without a trajdata cache no longer
  require the pinned NVlabs fork to be installed.
- The segment generators fall back to clip-relative segment timestamps when no
  trajdata cache is configured, instead of dropping every segment.

## [1.0.0] - 2026-05-27

### Added

- Initial public code release of the Chain-of-Causation autolabeling pipeline.
