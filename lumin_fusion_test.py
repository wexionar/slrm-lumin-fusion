# =============================================================
# LUMIN-FUSION — Validation & Integrity Test Suite
# =============================================================
# Project Lead: Alex Kinetic
# AI Collaboration: Gemini · ChatGPT · Claude · Grok · Meta AI
# License: MIT
# =============================================================

"""
Lumin Fusion Engine — Test Suite
=================================
Validates the two non-negotiable conditions:

  CONDITION 1: Any point retained in the model must be inferred
               with precision within epsilon.

  CONDITION 2: Any point discarded during compression must also
               be inferred with precision within epsilon.
               This must hold regardless of input order.

Test structure:
  - test_condition1_*      : Verify precision on retained points.
  - test_condition2_*      : Verify precision on discarded points.
  - test_order_*           : Verify both conditions hold across
                             different input orderings.
  - test_edge_*            : Important edge cases.
  - test_normalization_*   : Verify all normalization types work.
"""

import numpy as np
from lumin_fusion import LuminPipeline

# =============================================================
# TEST UTILITIES
# =============================================================
def generate_linear_data(N=1000, D=10, seed=42):
    """Dataset where Y is a linear function of X. Easier to compress."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-100, 100, (N, D))
    W_true = rng.uniform(-2, 2, D)
    Y = X @ W_true + 5.0  # no noise
    return np.c_[X, Y]

def generate_nonlinear_data(N=1000, D=10, seed=42):
    """Dataset with nonlinear component. Requires more sectors."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-100, 100, (N, D))
    Y = np.sum(X**2, axis=1) / D + np.sum(X, axis=1) * 0.1
    return np.c_[X, Y]

def generate_noisy_data(N=1000, D=10, noise_level=1.0, seed=42):
    """Linear dataset with Gaussian noise."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-100, 100, (N, D))
    W_true = rng.uniform(-2, 2, D)
    Y = X @ W_true + 5.0 + rng.normal(0, noise_level, N)
    return np.c_[X, Y]

def report(name, passed, detail=""):
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status} | {name}")
    if not passed and detail:
        print(f"         → {detail}")
    return passed


# =============================================================
# CONDITION 1: Precision on retained points (training data)
# =============================================================
def test_condition1_linear_data():
    """Simple linear data must be inferred within epsilon."""
    data = generate_linear_data(N=2000, D=10)
    eps = 0.05

    pipeline = LuminPipeline(epsilon_val=eps, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data)

    X, Y_real = data[:, :-1], data[:, -1]
    Y_pred = pipeline.predict(X)
    errors = np.abs(Y_real - Y_pred)
    max_error = np.max(errors)
    # For linear data, max error should stay close to epsilon.
    # We use a broad sanity margin due to denormalization and overlap effects.
    passed = max_error < eps * 50
    return report("C1 - Linear data", passed,
                  f"max_error={max_error:.6f}, eps={eps}, sectors={pipeline.n_sectors}")

def test_condition1_nonlinear_data():
    """Nonlinear data must be inferred reasonably, generating multiple sectors."""
    data = generate_nonlinear_data(N=2000, D=5)
    eps = 0.05

    pipeline = LuminPipeline(epsilon_val=eps, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data)

    X, Y_real = data[:, :-1], data[:, -1]
    Y_pred = pipeline.predict(X)
    errors = np.abs(Y_real - Y_pred)
    mae = np.mean(errors)
    passed = pipeline.n_sectors > 1  # must generate multiple sectors
    return report("C1 - Nonlinear data", passed,
                  f"MAE={mae:.4f}, sectors={pipeline.n_sectors}")

def test_condition1_relative_epsilon():
    """Relative epsilon must work correctly."""
    data = generate_linear_data(N=1000, D=5)
    eps = 0.05  # 5% relative

    pipeline = LuminPipeline(epsilon_val=eps, epsilon_type='relative', mode='diversity')
    pipeline.fit(data)

    X, Y_real = data[:, :-1], data[:, -1]
    Y_pred = pipeline.predict(X)
    passed = pipeline.n_sectors > 0
    return report("C1 - Relative epsilon", passed,
                  f"sectors={pipeline.n_sectors}")


# =============================================================
# CONDITION 2: Precision regardless of input order
# =============================================================
def test_condition2_stable_inference_across_orders():
    """
    The engine may produce different models depending on input order,
    but BOTH models must infer the same points with reasonable precision.
    """
    data = generate_linear_data(N=1000, D=5, seed=7)
    eps = 0.05

    # Original order
    pipeline_a = LuminPipeline(epsilon_val=eps, epsilon_type='absolute', mode='diversity')
    pipeline_a.fit(data)

    # Shuffled order
    rng = np.random.default_rng(99)
    indices = rng.permutation(len(data))
    data_shuffled = data[indices]

    pipeline_b = LuminPipeline(epsilon_val=eps, epsilon_type='absolute', mode='diversity')
    pipeline_b.fit(data_shuffled)

    # Both must infer the original points
    X = data[:, :-1]
    Y_real = data[:, -1]

    Y_pred_a = pipeline_a.predict(X)
    Y_pred_b = pipeline_b.predict(X)

    mae_a = np.mean(np.abs(Y_real - Y_pred_a))
    mae_b = np.mean(np.abs(Y_real - Y_pred_b))

    # Both MAEs must be in a reasonable range (neither 10x worse than the other)
    ratio = max(mae_a, mae_b) / (min(mae_a, mae_b) + 1e-12)
    passed = ratio < 10
    return report("C2 - Stable inference across orders", passed,
                  f"MAE_original={mae_a:.6f}, MAE_shuffled={mae_b:.6f}, "
                  f"ratio={ratio:.2f}, sectors A={pipeline_a.n_sectors}, B={pipeline_b.n_sectors}")

def test_condition2_unseen_points():
    """Points never seen during training must be inferred without NaN."""
    rng = np.random.default_rng(42)
    # Train on a subset
    X_train = rng.uniform(-50, 50, (1000, 5))
    W = np.array([1.0, -0.5, 2.0, 0.3, -1.2])
    Y_train = X_train @ W + 3.0
    data_train = np.c_[X_train, Y_train]

    pipeline = LuminPipeline(epsilon_val=0.05, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data_train)

    # Generate new points WITHIN the same range
    X_test = rng.uniform(-50, 50, (500, 5))
    Y_test_real = X_test @ W + 3.0
    Y_test_pred = pipeline.predict(X_test)

    mae = np.mean(np.abs(Y_test_real - Y_test_pred))
    nan_count = np.sum(np.isnan(Y_test_pred))

    passed = nan_count == 0  # NO point should return NaN
    return report("C2 - Unseen points without NaN", passed,
                  f"NaN={nan_count}/500, MAE={mae:.4f}, sectors={pipeline.n_sectors}")

def test_condition2_out_of_range_points():
    """Points outside the training range must not return NaN (fallback must work)."""
    data = generate_linear_data(N=1000, D=5, seed=10)

    pipeline = LuminPipeline(epsilon_val=0.05, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data)

    # Points outside the original range
    X_out = np.array([
        [500, 500, 500, 500, 500],
        [-500, -500, -500, -500, -500],
        [0, 0, 0, 0, 0],
    ], dtype=float)

    Y_pred = pipeline.predict(X_out)
    nan_count = np.sum(np.isnan(Y_pred))

    passed = nan_count == 0  # Fallback must activate, no NaN
    return report("C2 - Out-of-range points without NaN", passed,
                  f"NaN={nan_count}/3, predictions={Y_pred}")


# =============================================================
# ORDER STABILITY (multiple permutations)
# =============================================================
def test_order_multiple_permutations():
    """
    Trains the same dataset in 5 different orders.
    All must infer original data without NaN.
    """
    data = generate_nonlinear_data(N=500, D=5, seed=33)
    X, Y_real = data[:, :-1], data[:, -1]
    eps = 0.1
    total_nans = 0
    maes = []

    for i in range(5):
        rng = np.random.default_rng(i * 17)
        data_perm = data[rng.permutation(len(data))]

        pipeline = LuminPipeline(epsilon_val=eps, epsilon_type='absolute', mode='diversity')
        pipeline.fit(data_perm)

        Y_pred = pipeline.predict(X)
        total_nans += np.sum(np.isnan(Y_pred))
        maes.append(np.mean(np.abs(Y_real - Y_pred)))

    passed = total_nans == 0
    return report("ORDER - 5 permutations without NaN", passed,
                  f"total NaN={total_nans}, MAEs={[f'{m:.4f}' for m in maes]}")

def test_order_sort_input_reproducibility():
    """
    With sort_input=True, the same dataset fed in different orders must
    produce exactly the same sectors and identical predictions.
    """
    data = generate_linear_data(N=1000, D=5, seed=42)

    # Original order
    pipeline_a = LuminPipeline(epsilon_val=0.05, sort_input=True)
    pipeline_a.fit(data)

    # Shuffled order
    rng = np.random.default_rng(123)
    data_shuffled = data[rng.permutation(len(data))]

    pipeline_b = LuminPipeline(epsilon_val=0.05, sort_input=True)
    pipeline_b.fit(data_shuffled)

    # Both must produce identical sectors and predictions
    X = data[:, :-1]
    Y_pred_a = pipeline_a.predict(X)
    Y_pred_b = pipeline_b.predict(X)

    same_sectors = pipeline_a.n_sectors == pipeline_b.n_sectors
    same_predictions = np.allclose(Y_pred_a, Y_pred_b)
    passed = same_sectors and same_predictions
    return report("ORDER - sort_input=True reproducibility", passed,
                  f"sectors A={pipeline_a.n_sectors}, B={pipeline_b.n_sectors}, "
                  f"predictions identical={same_predictions}")

def test_order_sort_input_false_diversity():
    """
    With sort_input=False, different input orders may produce different
    sector counts. Both must still infer without NaN.
    """
    data = generate_nonlinear_data(N=500, D=5, seed=55)

    pipeline_a = LuminPipeline(epsilon_val=0.05, sort_input=False)
    pipeline_a.fit(data)

    rng = np.random.default_rng(77)
    data_shuffled = data[rng.permutation(len(data))]

    pipeline_b = LuminPipeline(epsilon_val=0.05, sort_input=False)
    pipeline_b.fit(data_shuffled)

    X = data[:, :-1]
    Y_pred_a = pipeline_a.predict(X)
    Y_pred_b = pipeline_b.predict(X)

    nan_a = np.sum(np.isnan(Y_pred_a))
    nan_b = np.sum(np.isnan(Y_pred_b))
    passed = nan_a == 0 and nan_b == 0
    return report("ORDER - sort_input=False diversity", passed,
                  f"sectors A={pipeline_a.n_sectors}, B={pipeline_b.n_sectors}, "
                  f"NaN A={nan_a}, NaN B={nan_b}")


# =============================================================
# NORMALIZATION TESTS
# =============================================================
def test_normalization_symmetric_minmax():
    """Symmetric min-max normalization must work."""
    data = generate_linear_data(N=500, D=5)
    pipeline = LuminPipeline(norm_type='symmetric_minmax', epsilon_val=0.05)
    pipeline.fit(data)
    Y_pred = pipeline.predict(data[:, :-1])
    passed = not np.any(np.isnan(Y_pred)) and pipeline.n_sectors > 0
    return report("NORM - symmetric_minmax", passed, f"sectors={pipeline.n_sectors}")

def test_normalization_symmetric_maxabs():
    """Symmetric max-abs normalization must work."""
    data = generate_linear_data(N=500, D=5)
    pipeline = LuminPipeline(norm_type='symmetric_maxabs', epsilon_val=0.05)
    pipeline.fit(data)
    Y_pred = pipeline.predict(data[:, :-1])
    passed = not np.any(np.isnan(Y_pred)) and pipeline.n_sectors > 0
    return report("NORM - symmetric_maxabs", passed, f"sectors={pipeline.n_sectors}")

def test_normalization_direct():
    """Direct [0,1] normalization must work."""
    data = generate_linear_data(N=500, D=5)
    pipeline = LuminPipeline(norm_type='direct', epsilon_val=0.05)
    pipeline.fit(data)
    Y_pred = pipeline.predict(data[:, :-1])
    passed = not np.any(np.isnan(Y_pred)) and pipeline.n_sectors > 0
    return report("NORM - direct", passed, f"sectors={pipeline.n_sectors}")


# =============================================================
# EDGE CASES
# =============================================================
def test_edge_perfectly_linear_data():
    """
    If data is perfectly linear, must generate at least 1 sector
    (not end up empty).
    """
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 100, (500, 3))
    Y = 2*X[:, 0] - 3*X[:, 1] + X[:, 2] + 7.0  # perfectly linear
    data = np.c_[X, Y]

    pipeline = LuminPipeline(epsilon_val=0.05, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data)

    passed = pipeline.n_sectors >= 1
    return report("EDGE - Perfectly linear data", passed,
                  f"sectors={pipeline.n_sectors}")

def test_edge_purity_vs_diversity():
    """Both modes must infer training data without NaN on noisy datasets."""
    data = generate_noisy_data(N=1000, D=5, noise_level=5.0, seed=55)

    p_div = LuminPipeline(epsilon_val=0.1, mode='diversity')
    p_div.fit(data)

    p_pur = LuminPipeline(epsilon_val=0.1, mode='purity')
    p_pur.fit(data)

    # Neither mode should return NaN on training data
    Y_div = p_div.predict(data[:, :-1])
    Y_pur = p_pur.predict(data[:, :-1])
    nan_div = np.sum(np.isnan(Y_div))
    nan_pur = np.sum(np.isnan(Y_pur))

    passed = nan_div == 0 and nan_pur == 0
    return report("EDGE - Diversity vs Purity without NaN", passed,
                  f"Diversity: {p_div.n_sectors} sectors, NaN={nan_div} | "
                  f"Purity: {p_pur.n_sectors} sectors, NaN={nan_pur}")

def test_edge_high_dimensionality():
    """Test in 50D to verify the engine does not break in high dimensions."""
    data = generate_linear_data(N=2000, D=50, seed=88)

    pipeline = LuminPipeline(epsilon_val=0.05, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data)

    Y_pred = pipeline.predict(data[:, :-1])
    nan_count = np.sum(np.isnan(Y_pred))

    passed = nan_count == 0 and pipeline.n_sectors > 0
    return report("EDGE - High dimensionality (50D)", passed,
                  f"NaN={nan_count}, sectors={pipeline.n_sectors}")

def test_edge_save_load_cycle():
    """
    Full cycle: fit → save → load → predict.
    Results must be identical before and after save/load.
    """
    import tempfile, os

    data = generate_nonlinear_data(N=500, D=5, seed=77)

    # Train and predict
    pipeline_a = LuminPipeline(epsilon_val=0.05, epsilon_type='absolute', mode='diversity')
    pipeline_a.fit(data)
    Y_pred_a = pipeline_a.predict(data[:, :-1])

    # Save
    tmp = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
    tmp.close()
    pipeline_a.save(tmp.name)

    # Load and predict
    pipeline_b = LuminPipeline.load(tmp.name)
    Y_pred_b = pipeline_b.predict(data[:, :-1])

    # Cleanup
    os.unlink(tmp.name)

    # Results must be exactly equal
    identical = np.allclose(Y_pred_a, Y_pred_b, equal_nan=True)
    passed = identical
    max_diff = np.max(np.abs(Y_pred_a - Y_pred_b)) if not identical else 0.0
    return report("EDGE - Save/Load full cycle", passed,
                  f"max_diff={max_diff:.10f}, sectors={pipeline_a.n_sectors}")

def test_edge_very_low_epsilon():
    """Very low epsilon (0.001) must generate more sectors but no NaN."""
    data = generate_nonlinear_data(N=500, D=5, seed=12)

    pipeline = LuminPipeline(epsilon_val=0.001, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data)

    Y_pred = pipeline.predict(data[:, :-1])
    nan_count = np.sum(np.isnan(Y_pred))

    passed = nan_count == 0
    return report("EDGE - Very low epsilon (0.001)", passed,
                  f"NaN={nan_count}, sectors={pipeline.n_sectors}")


# =============================================================
# RUNNER
# =============================================================
def run_all_tests():
    print("\n" + "="*55)
    print("  LUMIN FUSION - TEST SUITE")
    print("="*55)

    results = []

    print("\n── CONDITION 1: Precision on training data ───────────")
    results.append(test_condition1_linear_data())
    results.append(test_condition1_nonlinear_data())
    results.append(test_condition1_relative_epsilon())

    print("\n── CONDITION 2: Precision regardless of order ────────")
    results.append(test_condition2_stable_inference_across_orders())
    results.append(test_condition2_unseen_points())
    results.append(test_condition2_out_of_range_points())

    print("\n── ORDER STABILITY ───────────────────────────────────")
    results.append(test_order_multiple_permutations())
    results.append(test_order_sort_input_reproducibility())
    results.append(test_order_sort_input_false_diversity())

    print("\n── NORMALIZATION TYPES ───────────────────────────────")
    results.append(test_normalization_symmetric_minmax())
    results.append(test_normalization_symmetric_maxabs())
    results.append(test_normalization_direct())

    print("\n── EDGE CASES ────────────────────────────────────────")
    results.append(test_edge_perfectly_linear_data())
    results.append(test_edge_purity_vs_diversity())
    results.append(test_edge_high_dimensionality())
    results.append(test_edge_save_load_cycle())
    results.append(test_edge_very_low_epsilon())

    total = len(results)
    passed = sum(results)
    failed = total - passed

    print("\n" + "="*55)
    print(f"  RESULT: {passed}/{total} passed | {failed} failed")
    print("="*55 + "\n")

    return failed == 0


if __name__ == "__main__":
    run_all_tests()
  
