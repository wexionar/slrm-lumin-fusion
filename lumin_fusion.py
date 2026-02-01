# =============================================================
# LUMIN-FUSION — Origin & Resolution Engine
# =============================================================
# Project Lead: Alex Kinetic
# AI Collaboration: Gemini · ChatGPT · Claude · Grok · Meta AI
# License: MIT
# =============================================================

"""
Lumin Fusion Engine
====================
Logical compression engine for high-dimensional regression.

Architecture:
  - Normalizer      : Normalizes input data into a controlled range.
  - LuminOrigin     : Ingests normalized data point by point, building
                      sectors. Each sector encapsulates a local linear
                      law (W, B) that explains its points within epsilon.
                      When a point cannot be explained, the sector splits
                      (mitosis) and a new one begins.
  - LuminResolution : Inference engine. Given a point, finds the correct
                      sector and applies its local law to predict Y.
  - LuminPipeline   : Orchestrates the full flow: normalization ->
                      ingestion -> resolution. Main interface.

Fixes applied:
  1. Last sector was never closed -> closed via finalize() after ingestion.
  2. Points outside all bounding boxes returned NaN -> fallback to nearest
     sector by centroid distance.
  3. Bounding box overlap -> resolved by selecting the sector with smallest
     volume (most geometrically specific). Degenerate sectors with near-zero
     ranges are clamped to prevent them from always winning.
  4. X normalization in predict() no longer uses a dummy Y column.
  5. Input shape validation added to predict().
"""

import numpy as np


# =============================================================
# NORMALIZER
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
        # Prevent division by zero on constant columns
        self.s_range = np.where(self.s_range == 0, 1e-9, self.s_range)
        if self.norm_type == 'symmetric_maxabs':
            self.s_maxabs = np.max(np.abs(data), axis=0)
            self.s_maxabs = np.where(self.s_maxabs == 0, 1e-9, self.s_maxabs)

    def transform(self, data):
        """Normalizes full data (X + Y) according to the configured type."""
        if self.norm_type == 'symmetric_minmax':
            return 2 * (data - self.s_min) / self.s_range - 1
        elif self.norm_type == 'symmetric_maxabs':
            return data / self.s_maxabs
        else:  # direct
            return (data - self.s_min) / self.s_range

    def transform_x(self, X):
        """
        Normalizes only X columns (excludes Y).
        Uses the first D parameters fitted on the full dataset (X + Y).
        """
        if self.norm_type == 'symmetric_minmax':
            return 2 * (X - self.s_min[:-1]) / self.s_range[:-1] - 1
        elif self.norm_type == 'symmetric_maxabs':
            return X / self.s_maxabs[:-1]
        else:  # direct
            return (X - self.s_min[:-1]) / self.s_range[:-1]

    def inverse_transform_y(self, y_norm):
        """Denormalizes Y values (last column) back to original scale."""
        if self.norm_type == 'symmetric_minmax':
            return (y_norm + 1) * self.s_range[-1] / 2 + self.s_min[-1]
        elif self.norm_type == 'symmetric_maxabs':
            return y_norm * self.s_maxabs[-1]
        else:  # direct
            return y_norm * self.s_range[-1] + self.s_min[-1]


# =============================================================
# ORIGIN — Ingestion & Compression Engine
# =============================================================
class LuminOrigin:
    """
    Ingests normalized data point by point.
    Builds sectors where each one contains a local linear law (W, B)
    that explains its points within an epsilon margin.

    Parameters:
      epsilon_val  : numeric epsilon value (0 to 1).
      epsilon_type : 'absolute' or 'relative'.
      mode         : 'diversity' (retains noise) or 'purity' (eliminates it).
    """
    def __init__(self, epsilon_val=0.02, epsilon_type='absolute', mode='diversity'):
        self.epsilon_val = epsilon_val
        self.epsilon_type = epsilon_type
        self.mode = mode
        self.sectors = []          # closed sectors [min, max, W, B]
        self._current_nodes = []   # nodes of the sector being built
        self.D = None              # dimensionality of X

    def _calculate_law(self, nodes):
        """Computes W, B via least squares over the given nodes.
        Requires minimum D+1 nodes for a determined system."""
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
        """Returns the error threshold based on epsilon type."""
        if self.epsilon_type == 'relative':
            return abs(y_real) * self.epsilon_val
        return self.epsilon_val

    def _close_sector(self):
        """Closes the current sector and appends it to the sector list.
        Only closes if it has enough nodes (D+1 minimum)."""
        min_nodes = (self.D + 1) if self.D else 2
        if len(self._current_nodes) < min_nodes:
            return
        nodes = np.array(self._current_nodes)
        W, B = self._calculate_law(self._current_nodes)
        if W is None:
            return
        # Sector layout: [min per coordinate, max per coordinate, W, B]
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
        If the current local law explains it within epsilon -> add to sector.
        If not -> close the current sector (mitosis) and start a new one.

        CRITICAL: lstsq requires a minimum of D+1 points to solve a
        determined system in D dimensions. With fewer points the prediction
        is unreliable, so epsilon is not checked until we have enough nodes.
        """
        point = np.array(point, dtype=float)
        if self.D is None:
            self.D = len(point) - 1

        y_real = point[-1]
        min_nodes = self.D + 1  # minimum for a determined system

        # Until we have enough nodes, add without checking epsilon
        if len(self._current_nodes) < min_nodes:
            self._current_nodes.append(point.tolist())
            return

        W, B = self._calculate_law(self._current_nodes)
        y_pred = np.dot(point[:-1], W) + B
        error = abs(y_real - y_pred)
        threshold = self._get_threshold(y_real)

        if error <= threshold:
            # Point is explained by the current law
            self._current_nodes.append(point.tolist())
        else:
            # MITOSIS: close current sector, open a new one
            self._close_sector()
            if self.mode == 'diversity':
                # Carry the last D nodes as base for the new sector.
                # This gives us D nodes + the new point = D+1, enough to
                # evaluate epsilon on the very next ingestion step.
                self._current_nodes = self._current_nodes[-self.D:] + [point.tolist()]
            else:  # purity
                # Start clean with only the new point
                self._current_nodes = [point.tolist()]

    def finalize(self):
        """
        Closes the last sector.
        Must be called after all data has been ingested.
        """
        self._close_sector()
        self._current_nodes = []

    def get_sectors(self):
        """Returns sectors as a numpy array."""
        if not self.sectors:
            return np.array([])
        return np.array(self.sectors)


# =============================================================
# RESOLUTION — Inference Engine
# =============================================================
class LuminResolution:
    """
    Resolves predictions using the sectors generated by Origin.

    Overlap strategy: when a point falls inside multiple bounding boxes,
    the sector with the smallest volume is selected. Smaller volume means
    the sector is geometrically more specific to that region of space.
    Degenerate sectors (near-zero range on any axis) are clamped to
    prevent numerical artifacts in log-volume computation.

    Fallback: when a point falls outside all bounding boxes, the nearest
    sector by centroid distance is used.
    """
    def __init__(self, sectors, D):
        """
        sectors : numpy array of shape (N, 3*D + 1) with [min, max, W, B].
        D       : dimensionality of X.
        """
        self.D = D
        self.sectors = sectors
        self.mins = sectors[:, :D]
        self.maxs = sectors[:, D:2*D]
        self.weights = sectors[:, 2*D:3*D]
        self.biases = sectors[:, -1]
        # Centroid of each sector (midpoint of bounding box)
        self.centroids = (self.mins + self.maxs) / 2.0

    def _predict_with_sector(self, x, idx):
        """Prediction using a specific sector."""
        return np.dot(x, self.weights[idx]) + self.biases[idx]

    def resolve(self, X):
        """
        Resolves predictions for an array of normalized X points.

        Logic per point:
          1. Find all sectors where the point falls inside the bounding box.
          2. If multiple -> select the one with smallest bounding box volume
             (most geometrically specific).
          3. If none -> fallback to nearest sector by centroid distance.
        """
        X = np.atleast_2d(X)
        n_points = X.shape[0]
        results = np.full(n_points, np.nan)

        for i in range(n_points):
            x = X[i]

            # Find sectors that contain this point (inside bounding box)
            inside_mask = np.all(
                (x >= self.mins - 1e-9) & (x <= self.maxs + 1e-9),
                axis=1
            )
            candidates = np.where(inside_mask)[0]

            if len(candidates) == 1:
                # Simple case: exactly one sector contains the point
                results[i] = self._predict_with_sector(x, candidates[0])

            elif len(candidates) > 1:
                # Overlap: select the sector with smallest bounding box volume.
                # Clamp ranges to 1e-6 to prevent degenerate sectors
                # (near-zero range) from collapsing log-volume to -inf
                # and always winning the selection.
                ranges = np.clip(
                    self.maxs[candidates] - self.mins[candidates],
                    1e-6, None
                )
                log_volumes = np.sum(np.log(ranges), axis=1)
                best = candidates[np.argmin(log_volumes)]
                results[i] = self._predict_with_sector(x, best)

            else:
                # Fallback: point outside all bounding boxes.
                # Use the nearest sector by centroid distance.
                distances = np.linalg.norm(self.centroids - x, axis=1)
                nearest = np.argmin(distances)
                results[i] = self._predict_with_sector(x, nearest)

        return results


# =============================================================
# PIPELINE
# =============================================================
class LuminPipeline:
    """
    Orchestrates the full flow: normalization -> ingestion -> resolution.
    Main interface for the engine.

    Parameters:
      epsilon_val  : numeric epsilon value (0 to 1).
      epsilon_type : 'absolute' or 'relative'.
      mode         : 'diversity' (retains noise) or 'purity' (eliminates it).
      norm_type    : normalization type ('symmetric_minmax', 'symmetric_maxabs', 'direct').
      sort_input   : if True, sorts normalized data by Euclidean distance from
                     the origin before ingestion. This makes the model fully
                     reproducible: same dataset always produces the same sectors,
                     regardless of the original row order.
                     If False, preserves the original input order, which may
                     produce different sector layouts depending on data arrival.
                     Default: True.
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
        """
        Trains the full engine.
        data : numpy array of shape (N, D+1) where the last column is Y.
        """
        # Normalize
        self.normalizer.fit(data)
        data_norm = self.normalizer.transform(data)
        self.D = data.shape[1] - 1

        # Sort by Euclidean distance from origin if enabled.
        # This makes the ingestion order deterministic regardless of how
        # the original dataset was ordered. Cost: O(N log N), negligible
        # compared to the ingestion loop itself.
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
        self.origin.finalize()  # CRITICAL: close the last sector

        # Prepare resolution
        sectors = self.origin.get_sectors()
        if len(sectors) == 0:
            raise ValueError("No sectors were generated. Check data or epsilon.")
        self.resolution = LuminResolution(sectors, self.D)

        return self

    def predict(self, X):
        """
        Predicts Y for an array X (raw, unnormalized values).
        X : numpy array of shape (N, D) or (D,).
        """
        X = np.atleast_2d(X)

        # Input validation
        if X.shape[1] != self.D:
            raise ValueError(
                f"X has {X.shape[1]} columns, expected {self.D}. "
                f"Input must match the dimensionality used during fit()."
            )

        # Normalize X directly using dedicated transform_x
        X_norm = self.normalizer.transform_x(X)

        # Resolve in normalized space
        y_norm = self.resolution.resolve(X_norm)

        # Denormalize Y back to original scale
        return self.normalizer.inverse_transform_y(y_norm)

    @property
    def n_sectors(self):
        return len(self.origin.sectors) if self.origin else 0

    def save(self, filename="lumin_model.npy"):
        """
        Saves the model to .npy.
        Contains everything Resolution needs to infer:
          - sectors      : array [min, max, W, B] per sector.
          - s_min        : column minimums (all columns, including Y).
          - s_max        : column maximums.
          - s_range      : column ranges.
          - s_maxabs     : max(abs) per column (only if symmetric_maxabs).
          - norm_type    : normalization type used.
          - D            : dimensionality of X.
          - epsilon_val  : epsilon used in Origin.
          - epsilon_type : epsilon type.
          - mode         : diversity or purity.
        """
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
        """
        Loads a model from .npy and reconstructs the pipeline,
        ready to infer with predict().
        Does not need original data or Origin.
        """
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

        # Reconstruct resolution directly from sectors
        pipeline.D = model_data['D']
        sectors = model_data['sectors']
        pipeline.resolution = LuminResolution(sectors, pipeline.D)

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
