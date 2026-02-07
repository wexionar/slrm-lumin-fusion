# =============================================================
# LUMIN-FUSION v2.0 — Optimized Resolution Engine
# =============================================================
# Project Lead: Alex Kinetic
# AI Collaboration: Gemini · ChatGPT · Claude · Grok · Meta AI
# License: MIT
# =============================================================

"""
Lumin Fusion Engine v2.0 - Optimizations
==========================================
Logical compression engine for high-dimensional regression.

CHANGES FROM v1.0:
  1. Fast sector search using KD-Tree for >1000 sectors
  2. Improved overlap tie-breaking (volume + centroid distance)
  3. Smarter mitosis in diversity mode (closest points, not last D)
  4. Vectorized bounding box checks
  5. All 17 tests still pass - backward compatible

Architecture unchanged:
  - Normalizer      : Normalizes input data
  - LuminOrigin     : Ingests data, builds sectors with mitosis
  - LuminResolution : Inference engine (OPTIMIZED)
  - LuminPipeline   : Main interface
"""

import numpy as np


# =============================================================
# NORMALIZER (unchanged)
# =============================================================
class Normalizer:
    """
    Supported types:
      - 'symmetric_minmax' : range [-1, 1] using min and max per column.
      - 'symmetric_maxabs' : range [-1, 1] using max(abs) per column.
      - 'direct'           : range [0, 1] using min and max per column.
    """
    VALID_TYPES = ('symmetric_minmax', 'symmetric_maxabs', 'direct')

    def __init__(self, norm_type='symmetric_minmax'):
        if norm_type not in self.VALID_TYPES:
            raise ValueError(f"norm_type must be one of {self.VALID_TYPES}")
        self.norm_type = norm_type
        self.s_min = None
        self.s_max = None
        self.s_range = None
        self.s_maxabs = None

    def fit(self, data):
        """Computes normalization parameters from data."""
        self.s_min = data.min(axis=0)
        self.s_max = data.max(axis=0)
        self.s_range = self.s_max - self.s_min
        self.s_range = np.where(self.s_range == 0, 1e-9, self.s_range)
        if self.norm_type == 'symmetric_maxabs':
            self.s_maxabs = np.max(np.abs(data), axis=0)
            self.s_maxabs = np.where(self.s_maxabs == 0, 1e-9, self.s_maxabs)

    def transform(self, data):
        """Normalizes full data (X + Y)."""
        if self.norm_type == 'symmetric_minmax':
            return 2 * (data - self.s_min) / self.s_range - 1
        elif self.norm_type == 'symmetric_maxabs':
            return data / self.s_maxabs
        else:  # direct
            return (data - self.s_min) / self.s_range

    def transform_x(self, X):
        """Normalizes only X columns (excludes Y)."""
        if self.norm_type == 'symmetric_minmax':
            return 2 * (X - self.s_min[:-1]) / self.s_range[:-1] - 1
        elif self.norm_type == 'symmetric_maxabs':
            return X / self.s_maxabs[:-1]
        else:  # direct
            return (X - self.s_min[:-1]) / self.s_range[:-1]

    def inverse_transform_y(self, y_norm):
        """Denormalizes Y values (last column)."""
        if self.norm_type == 'symmetric_minmax':
            return (y_norm + 1) * self.s_range[-1] / 2 + self.s_min[-1]
        elif self.norm_type == 'symmetric_maxabs':
            return y_norm * self.s_maxabs[-1]
        else:  # direct
            return y_norm * self.s_range[-1] + self.s_min[-1]


# =============================================================
# ORIGIN — Ingestion & Compression Engine (OPTIMIZED)
# =============================================================
class LuminOrigin:
    """
    Ingests normalized data point by point.
    Builds sectors where each one contains a local linear law (W, B).

    OPTIMIZATION v2.0: Smarter mitosis in diversity mode.
    """
    def __init__(self, epsilon_val=0.02, epsilon_type='absolute', mode='diversity'):
        self.epsilon_val = epsilon_val
        self.epsilon_type = epsilon_type
        self.mode = mode
        self.sectors = []
        self._current_nodes = []
        self.D = None

    def _calculate_law(self, nodes):
        """Computes W, B via least squares."""
        if len(nodes) < 2:
            return None, None
        nodes_np = np.array(nodes)
        X, Y = nodes_np[:, :-1], nodes_np[:, -1]
        A = np.c_[X, np.ones(X.shape[0])]
        try:
            res = np.linalg.lstsq(A, Y, rcond=None)[0]
            return res[:-1], res[-1]
        except np.linalg.LinAlgError:
            return None, None

    def _get_threshold(self, y_real):
        """Returns error threshold based on epsilon type."""
        if self.epsilon_type == 'relative':
            return abs(y_real) * self.epsilon_val
        return self.epsilon_val

    def _close_sector(self):
        """Closes current sector and appends to list."""
        min_nodes = (self.D + 1) if self.D else 2
        if len(self._current_nodes) < min_nodes:
            return
        nodes = np.array(self._current_nodes)
        W, B = self._calculate_law(self._current_nodes)
        if W is None:
            return
        
        sector = np.concatenate([
            np.min(nodes[:, :-1], axis=0),   # bounding box min
            np.max(nodes[:, :-1], axis=0),   # bounding box max
            W,                                # weights
            [B]                               # bias
        ])
        self.sectors.append(sector)

    def ingest(self, point):
        """
        Ingests a single normalized point.
        OPTIMIZATION v2.0: In diversity mode, carry the D points closest
        to the new point (instead of last D), for smoother transitions.
        """
        point = np.array(point, dtype=float)
        if self.D is None:
            self.D = len(point) - 1

        y_real = point[-1]
        min_nodes = self.D + 1

        # Accumulate until we have enough nodes
        if len(self._current_nodes) < min_nodes:
            self._current_nodes.append(point.tolist())
            return

        W, B = self._calculate_law(self._current_nodes)
        y_pred = np.dot(point[:-1], W) + B
        error = abs(y_real - y_pred)
        threshold = self._get_threshold(y_real)

        if error <= threshold:
            # Point is explained by current law
            self._current_nodes.append(point.tolist())
        else:
            # MITOSIS: close current sector, start new one
            self._close_sector()
            
            if self.mode == 'diversity':
                # OPTIMIZATION: Instead of last D nodes, take the D closest to new point
                nodes_array = np.array(self._current_nodes)
                distances = np.linalg.norm(nodes_array[:, :-1] - point[:-1], axis=1)
                closest_indices = np.argsort(distances)[:self.D]
                self._current_nodes = [self._current_nodes[i] for i in closest_indices]
            else:
                # Purity: start fresh
                self._current_nodes = []
            
            self._current_nodes.append(point.tolist())

    def finalize(self):
        """Close the last sector."""
        self._close_sector()

    def get_sectors(self):
        """Returns all closed sectors."""
        return self.sectors


# =============================================================
# RESOLUTION — Inference Engine (HEAVILY OPTIMIZED)
# =============================================================
class LuminResolution:
    """
    Inference engine. Resolves predictions from pre-built sector array.
    
    OPTIMIZATIONS v2.0:
      1. KD-Tree for fast nearest sector search (>1000 sectors)
      2. Vectorized bounding box checks
      3. Improved tie-breaking (volume + centroid distance)
      4. Batch processing optimizations
    """
    def __init__(self, sectors, D):
        self.D = D
        sectors_array = np.array(sectors)
        
        # Parse sector layout: [min(D), max(D), W(D), B(1)]
        self.mins = sectors_array[:, :D]
        self.maxs = sectors_array[:, D:2*D]
        self.Ws = sectors_array[:, 2*D:3*D]
        self.Bs = sectors_array[:, 3*D]
        
        # Precompute centroids for fallback
        self.centroids = (self.mins + self.maxs) / 2.0
        
        # OPTIMIZATION 1: Build KD-Tree if many sectors
        self.use_fast_search = len(sectors) > 1000
        if self.use_fast_search:
            try:
                from scipy.spatial import KDTree
                self.centroid_tree = KDTree(self.centroids)
            except ImportError:
                self.use_fast_search = False

    def _predict_with_sector(self, x, sector_idx):
        """Apply sector's law to point x."""
        return np.dot(x, self.Ws[sector_idx]) + self.Bs[sector_idx]

    def resolve(self, X):
        """
        Resolves predictions for array X (normalized).
        
        OPTIMIZATION v2.0: Vectorized + KD-Tree acceleration.
        """
        X = np.atleast_2d(X)
        results = np.zeros(len(X))

        if self.use_fast_search:
            return self._resolve_fast(X)
        else:
            return self._resolve_standard(X)

    def _resolve_standard(self, X):
        """Standard resolution (original algorithm)."""
        results = np.zeros(len(X))
        
        for i, x in enumerate(X):
            # VECTORIZED: Check all bounding boxes at once
            in_bounds = np.all((self.mins <= x) & (x <= self.maxs), axis=1)
            candidates = np.where(in_bounds)[0]

            if len(candidates) == 0:
                # Fallback: nearest sector by centroid
                distances = np.linalg.norm(self.centroids - x, axis=1)
                nearest = np.argmin(distances)
                results[i] = self._predict_with_sector(x, nearest)

            elif len(candidates) == 1:
                # Single match
                results[i] = self._predict_with_sector(x, candidates[0])

            else:
                # OPTIMIZATION 2: Improved tie-breaking
                # First criterion: smallest volume
                ranges = np.clip(
                    self.maxs[candidates] - self.mins[candidates],
                    1e-6, None
                )
                log_volumes = np.sum(np.log(ranges), axis=1)
                
                min_vol = np.min(log_volumes)
                max_vol = np.max(log_volumes)
                
                # If volumes are very similar (within 1%), use centroid distance
                if (max_vol - min_vol) < 0.01:
                    centroid_dists = np.linalg.norm(self.centroids[candidates] - x, axis=1)
                    best = candidates[np.argmin(centroid_dists)]
                else:
                    # Clear winner by volume
                    best = candidates[np.argmin(log_volumes)]
                
                results[i] = self._predict_with_sector(x, best)

        return results

    def _resolve_fast(self, X):
        """
        Fast resolution using KD-Tree for initial filtering.
        OPTIMIZATION: Only check nearest K sectors instead of all.
        """
        results = np.zeros(len(X))
        k_search = min(20, len(self.centroids))  # Check at most 20 nearest sectors
        
        for i, x in enumerate(X):
            # Find k nearest sectors by centroid
            _, nearest_indices = self.centroid_tree.query(x, k=k_search)
            
            # Check only these k sectors for bounding box containment
            in_bounds = np.all(
                (self.mins[nearest_indices] <= x) & (x <= self.maxs[nearest_indices]),
                axis=1
            )
            local_candidates = nearest_indices[in_bounds]

            if len(local_candidates) == 0:
                # None of the k nearest contain x -> use closest centroid
                nearest = nearest_indices[0]
                results[i] = self._predict_with_sector(x, nearest)

            elif len(local_candidates) == 1:
                results[i] = self._predict_with_sector(x, local_candidates[0])

            else:
                # Tie-breaking among local candidates
                ranges = np.clip(
                    self.maxs[local_candidates] - self.mins[local_candidates],
                    1e-6, None
                )
                log_volumes = np.sum(np.log(ranges), axis=1)
                
                min_vol = np.min(log_volumes)
                max_vol = np.max(log_volumes)
                
                if (max_vol - min_vol) < 0.01:
                    centroid_dists = np.linalg.norm(
                        self.centroids[local_candidates] - x, axis=1
                    )
                    best = local_candidates[np.argmin(centroid_dists)]
                else:
                    best = local_candidates[np.argmin(log_volumes)]
                
                results[i] = self._predict_with_sector(x, best)

        return results


# =============================================================
# PIPELINE (minimal changes)
# =============================================================
class LuminPipeline:
    """
    Orchestrates: normalization -> ingestion -> resolution.
    
    v2.0: Uses optimized LuminOrigin and LuminResolution.
    """
    def __init__(self, epsilon_val=0.02, epsilon_type='absolute',
                 mode='diversity', norm_type='symmetric_minmax',
                 sort_input=True):
        self.normalizer = Normalizer(norm_type)
        self.epsilon_val = epsilon_val
        self.epsilon_type = epsilon_type
        self.mode = mode
        self.sort_input = sort_input
        self.origin = None
        self.resolution = None
        self.D = None

    def fit(self, data):
        """Trains the full engine."""
        # Normalize
        self.normalizer.fit(data)
        data_norm = self.normalizer.transform(data)
        self.D = data.shape[1] - 1

        # Sort by distance from origin if enabled
        if self.sort_input:
            distances = np.linalg.norm(data_norm[:, :-1], axis=1)
            sort_indices = np.argsort(distances)
            data_norm = data_norm[sort_indices]

        # Ingest
        self.origin = LuminOrigin(
            epsilon_val=self.epsilon_val,
            epsilon_type=self.epsilon_type,
            mode=self.mode
        )
        for point in data_norm:
            self.origin.ingest(point)
        self.origin.finalize()

        # Prepare resolution
        sectors = self.origin.get_sectors()
        if len(sectors) == 0:
            raise ValueError("No sectors were generated. Check data or epsilon.")
        self.resolution = LuminResolution(sectors, self.D)

        return self

    def predict(self, X):
        """Predicts Y for array X (raw, unnormalized)."""
        X = np.atleast_2d(X)

        if X.shape[1] != self.D:
            raise ValueError(
                f"X has {X.shape[1]} columns, expected {self.D}. "
                f"Input must match dimensionality used during fit()."
            )

        # Normalize X
        X_norm = self.normalizer.transform_x(X)

        # Resolve
        y_norm = self.resolution.resolve(X_norm)

        # Denormalize Y
        return self.normalizer.inverse_transform_y(y_norm)

    @property
    def n_sectors(self):
        return len(self.origin.sectors) if self.origin else 0

    def save(self, filename="lumin_model.npy"):
        """Saves model to .npy."""
        if self.origin is None or self.resolution is None:
            raise ValueError("Model has not been trained. Run fit() first.")

        model_data = {
            'sectors':      np.array(self.origin.sectors),
            's_min':        self.normalizer.s_min,
            's_max':        self.normalizer.s_max,
            's_range':      self.normalizer.s_range,
            's_maxabs':     self.normalizer.s_maxabs if self.normalizer.s_maxabs is not None else np.array([]),
            'norm_type':    self.normalizer.norm_type,
            'D':            self.D,
            'epsilon_val':  self.epsilon_val,
            'epsilon_type': self.epsilon_type,
            'mode':         self.mode,
            'sort_input':   self.sort_input,
        }
        np.save(filename, model_data)
        return filename

    @classmethod
    def load(cls, filename="lumin_model.npy"):
        """Loads model from .npy."""
        model_data = np.load(filename, allow_pickle=True).item()

        pipeline = cls(
            epsilon_val=model_data['epsilon_val'],
            epsilon_type=model_data['epsilon_type'],
            mode=model_data['mode'],
            norm_type=model_data['norm_type'],
            sort_input=model_data.get('sort_input', True),
        )

        # Reconstruct normalizer
        pipeline.normalizer.s_min   = model_data['s_min']
        pipeline.normalizer.s_max   = model_data['s_max']
        pipeline.normalizer.s_range = model_data['s_range']
        if len(model_data['s_maxabs']) > 0:
            pipeline.normalizer.s_maxabs = model_data['s_maxabs']

        # Reconstruct resolution
        pipeline.D = model_data['D']
        sectors = model_data['sectors']
        pipeline.resolution = LuminResolution(sectors, pipeline.D)

        # Create dummy origin for n_sectors property
        pipeline.origin = type('obj', (object,), {'sectors': sectors})()

        return pipeline

    def get_metadata(self):
        """Returns metadata of the trained model."""
        return {
            'n_sectors': self.n_sectors,
            'D': self.D,
            'epsilon_val': self.epsilon_val,
            'epsilon_type': self.epsilon_type,
            'mode': self.mode,
            'norm_type': self.normalizer.norm_type,
            'sort_input': self.sort_input,
        }


# =============================================================
# DEMO
# =============================================================
if __name__ == "__main__":
    import time
    
    print("="*60)
    print("LUMIN FUSION v2.0 - OPTIMIZATION DEMO")
    print("="*60)
    
    # Generate test data
    rng = np.random.default_rng(42)
    N, D = 5000, 20
    X = rng.uniform(-100, 100, (N, D))
    W = rng.uniform(-2, 2, D)
    Y = X @ W + 5.0
    data = np.c_[X, Y]
    
    print(f"\nDataset: {N} points in {D}D")
    
    # Train
    start = time.perf_counter()
    pipeline = LuminPipeline(epsilon_val=0.05, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data)
    train_time = time.perf_counter() - start
    
    print(f"Training time: {train_time:.3f}s")
    print(f"Sectors generated: {pipeline.n_sectors}")
    print(f"Fast search: {pipeline.resolution.use_fast_search}")
    
    # Predict
    X_test = rng.uniform(-120, 120, (1000, D))
    
    start = time.perf_counter()
    Y_pred = pipeline.predict(X_test)
    inference_time = time.perf_counter() - start
    
    print(f"\nInference time (1000 points): {inference_time*1000:.2f}ms")
    print(f"Per-point latency: {inference_time/1000*1000:.4f}ms")
    
    # Accuracy on training data
    Y_train_pred = pipeline.predict(X)
    max_error = np.max(np.abs(Y - Y_train_pred))
    mae = np.mean(np.abs(Y - Y_train_pred))
    
    print(f"\nTraining accuracy:")
    print(f"  MAE: {mae:.6f}")
    print(f"  Max error: {max_error:.6f}")
    print(f"  Target epsilon: {0.05}")
    
    print("\n" + "="*60)
    print("✓ Demo complete. Run lumin_fusion_test.py for full validation.")
    print("="*60)
 
