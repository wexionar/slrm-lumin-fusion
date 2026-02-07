# Lumin Fusion

> Logical compression engine for high-dimensional regression using geometric sector decomposition.

Lumin Fusion compresses datasets by partitioning normalized space into sectors, each governed by a local linear law together forming a global piecewise-linear model `Y = W·X + B`. Instead of storing raw data or training a dense neural network, the engine retains only the minimal geometric description needed to reconstruct any point within a user-defined precision threshold (epsilon).

---

## How It Works

```
Raw Data → Normalize → Sort (optional) → Ingest → Sectors → Resolve → Predict
```

1. **Normalization** — Input data is scaled into a controlled range (`[-1,1]` or `[0,1]`) so that epsilon operates uniformly across all dimensions.

2. **Ingestion (Origin)** — Data is fed point by point. Each point is tested against the current sector's local linear law. If the prediction error exceeds epsilon, the sector is closed (mitosis) and a new one begins. Each closed sector stores its bounding box `[min, max]` and its law `[W, B]`.

3. **Resolution** — At inference time, a query point is located within the sector space. If it falls inside a bounding box, that sector's law is applied. Overlapping sectors are resolved by selecting the one with the smallest volume (most geometrically specific). Points outside all bounding boxes fall back to the nearest sector by centroid distance.

---

## Architecture

| Component | Role |
|---|---|
| `Normalizer` | Fits and applies normalization. Supports `symmetric_minmax`, `symmetric_maxabs`, and `direct`. |
| `LuminOrigin` | Ingestion engine. Builds sectors via sequential point-by-point processing with epsilon-controlled mitosis. |
| `LuminResolution` | Inference engine. Resolves predictions from a pre-built sector array. Operates independently — does not need Origin at runtime. |
| `LuminPipeline` | Main interface. Orchestrates normalization → ingestion → resolution. Handles save/load. |


The design deliberately separates model construction (Origin) from model usage (Resolution), allowing lightweight inference without retraining.

This approach enables compact, interpretable models that approximate complex functions without gradient training.

---

## Recent Improvements

**Performance Optimizations:**
- KD-Tree acceleration for datasets with >1000 sectors (2-3x faster inference)
- Improved overlap tie-breaking (volume + centroid distance)
- Smarter diversity mode mitosis (selects closest points instead of last D)
- Vectorized bounding box operations

All optimizations maintain 100% backward compatibility. The 17-test validation suite passes without modification.

---

## Key Design Decisions

**Two non-negotiable conditions govern the engine:**

- **Condition 1** — Any point retained in the compressed model must be inferred within epsilon.
- **Condition 2** — Any point discarded during compression must also be inferred within epsilon. This must hold regardless of input order.

**Reproducibility vs. Diversity (`sort_input`):**

By default (`sort_input=True`), the engine sorts normalized data by Euclidean distance from the origin before ingestion. This makes the output fully deterministic — the same dataset always produces the same sectors, regardless of row order. Setting `sort_input=False` preserves the original input order, which may produce different sector layouts depending on how the data arrives.

**Minimum node requirement:**

`lstsq` needs at least `D+1` points to solve a determined linear system in `D` dimensions. Origin enforces this: epsilon is not checked until a sector accumulates enough nodes. This prevents underdetermined systems from producing unreliable predictions early in sector construction.

**Overlap resolution:**

In high dimensions, bounding boxes overlap extensively. When a point falls inside multiple sectors, the one with the smallest bounding box volume is selected. If volumes are within 1% (tie), the nearest sector by centroid distance is chosen. Degenerate sectors (near-zero range on any axis) are clamped to prevent numerical artifacts.

These constraints ensure the engine behaves as a geometric compression system rather than a traditional black-box learner.

---

## Usage

```python
import numpy as np
from lumin_fusion import LuminPipeline

# Prepare data: shape (N, D+1), last column is Y
rng = np.random.default_rng(42)
X = rng.uniform(-100, 100, (2000, 10))
W = rng.uniform(-2, 2, 10)
Y = X @ W + 5.0
data = np.c_[X, Y]

# Train
pipeline = LuminPipeline(epsilon_val=0.05, epsilon_type='absolute', mode='diversity')
pipeline.fit(data)
print(f"Sectors generated: {pipeline.n_sectors}")

# Predict on training data
Y_pred = pipeline.predict(X)
print(f"Max error: {np.max(np.abs(Y - Y_pred)):.6f}")

# Predict on unseen data
X_new = rng.uniform(-120, 120, (5, 10))
Y_new_pred = pipeline.predict(X_new)
print(Y_new_pred)

# Save and load
pipeline.save("model.npy")
pipeline_loaded = LuminPipeline.load("model.npy")

# Predict after loading
Y_pred_loaded = pipeline_loaded.predict(X)
print(f"Reloaded max error: {np.max(np.abs(Y - Y_pred_loaded)):.6f}")
```

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `epsilon_val` | float | `0.02` | Precision threshold (0 to 1). |
| `epsilon_type` | str | `'absolute'` | `'absolute'` or `'relative'`. Relative scales epsilon by `\|Y\|`. |
| `mode` | str | `'diversity'` | `'diversity'` carries closest D nodes into new sectors after mitosis. `'purity'` starts clean. |
| `norm_type` | str | `'symmetric_minmax'` | Normalization strategy. |
| `sort_input` | bool | `True` | If True, sorts data before ingestion for full reproducibility. |

---

## Model File Format (`.npy`)

The saved model contains everything Resolution needs to infer — Origin is not required at runtime:

```
sectors      : array [min, max, W, B] per sector
s_min        : column minimums (all columns including Y)
s_max        : column maximums
s_range      : column ranges
s_maxabs     : max(abs) per column (only if symmetric_maxabs)
norm_type    : normalization type used
D            : dimensionality of X
epsilon_val  : epsilon value
epsilon_type : epsilon type
mode         : diversity or purity
sort_input   : whether sorting was enabled
```

---

## Performance

Typical performance characteristics (Intel i7-12700K, single thread):

| Dataset | Sectors | Training | Inference (1000 pts) | Model Size |
|---------|---------|----------|---------------------|------------|
| 500 × 5D | 1 | 0.06s | 7.4ms | ~1KB |
| 2K × 20D | 1 | 4.5s | 11.6ms | ~8KB |
| 5K × 50D | 1 | 60s | 12.8ms | ~50KB |
| 2K × 10D (ε=0.001) | 1755 | 2.2s | 73ms* | ~140KB |

*KD-Tree acceleration active (>1000 sectors)

---

## Running Tests

```bash
python lumin_fusion_test.py
```

The test suite covers 17 cases across all engine components:

- Condition 1 & 2 precision guarantees
- Order stability across permutations
- `sort_input=True` reproducibility and `sort_input=False` diversity
- All three normalization types
- Edge cases: perfectly linear data, high dimensionality (50D), very low epsilon, purity vs diversity, save/load cycle integrity

---

## Files

| File | Description |
|---|---|
| `lumin_fusion.py` | Full engine: Normalizer, Origin, Resolution, Pipeline. |
| `lumin_fusion_test.py` | Validation and integrity test suite (17 tests). |
| `README.md` | This file. |

---

## License

MIT
 
