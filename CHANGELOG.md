# Changelog

All notable changes to Lumin Fusion are documented here.

---

## [Recent Updates] - 2026-02-07

### Added
- KD-Tree acceleration for datasets with >1000 sectors (scipy dependency)
- Improved tie-breaking in overlap resolution (volume + centroid distance)
- Performance benchmarks in README

### Changed
- **LuminOrigin mitosis (diversity mode):** Now selects D closest points to new point instead of last D points, improving local coherence in high-curvature regions
- **LuminResolution overlap handling:** When bounding box volumes are within 1%, uses centroid distance as tie-breaker
- **LuminResolution search strategy:** Automatically switches to KD-Tree acceleration when >1000 sectors are present

### Performance
- 2-3x faster inference on datasets with >1000 sectors
- Up to 2.2x faster on large datasets (5K+ points, 50D+)
- No performance regression on small datasets

### Compatibility
- ✅ 100% backward compatible
- ✅ All 17 validation tests pass without modification
- ✅ Saved models (.npy) are identical in format
- ✅ API unchanged (drop-in replacement)

---

## [Initial Release] - 2026-02-01

### Core Features
- Sequential ingestion with epsilon-controlled mitosis
- Adaptive sector partitioning (Phase B.2 optimization)
- Deterministic inference via bounding box lookup
- Two non-negotiable conditions (precision guarantees)
- Full save/load support (.npy format)

### Components
- `Normalizer`: Three normalization strategies
- `LuminOrigin`: Sector construction engine
- `LuminResolution`: Lightweight inference engine
- `LuminPipeline`: Main interface

### Validation
- 17-test comprehensive suite
- Condition 1 & 2 verification
- Order stability checks
- Edge case coverage (50D, low epsilon, purity/diversity)
 
