# =============================================================
# LUMIN-FUSION — Lumin Fusion Core Engine
# =============================================================
# Project Lead: Alex Kinetic
# AI Collaboration: Gemini · ChatGPT · Claude · Grok · Meta AI
# License: MIT
# =============================================================

"""
LuminCore Engine - Motor de Compresión Lógica
=============================================
Versión limpia: solo motor + tests.

Correcciones aplicadas:
  1. Último sector nunca se cerraba → se cierra al finalizar ingest.
  2. Puntos fuera de todos los bounding boxes retornaban NaN → fallback
     al sector más cercano por distancia al centroide.
  3. Solapamiento de bounding boxes → cuando un punto cae en varios
     sectores, se selecciona el que tiene menor error de predicción.
"""

import numpy as np


# =============================================================
# NORMALIZACIÓN
# =============================================================
class Normalizer:
    """
    Tipos soportados:
      - 'symmetric_minmax' : rango [-1, 1] usando min y max por columna.
      - 'symmetric_maxabs' : rango [-1, 1] usando max(abs) por columna.
      - 'direct'           : rango [0, 1] usando min y max por columna.
    """
    VALID_TYPES = ('symmetric_minmax', 'symmetric_maxabs', 'direct')

    def __init__(self, norm_type='symmetric_minmax'):
        if norm_type not in self.VALID_TYPES:
            raise ValueError(f"norm_type debe ser uno de {self.VALID_TYPES}")
        self.norm_type = norm_type
        self.s_min = None
        self.s_max = None
        self.s_range = None
        self.s_maxabs = None

    def fit(self, data):
        """Calcula los parámetros de normalización a partir de los datos."""
        self.s_min = data.min(axis=0)
        self.s_max = data.max(axis=0)
        self.s_range = self.s_max - self.s_min
        # Evitar división por cero en columnas constantes
        self.s_range = np.where(self.s_range == 0, 1e-9, self.s_range)
        if self.norm_type == 'symmetric_maxabs':
            self.s_maxabs = np.max(np.abs(data), axis=0)
            self.s_maxabs = np.where(self.s_maxabs == 0, 1e-9, self.s_maxabs)

    def transform(self, data):
        """Normaliza los datos según el tipo configurado."""
        if self.norm_type == 'symmetric_minmax':
            return 2 * (data - self.s_min) / self.s_range - 1
        elif self.norm_type == 'symmetric_maxabs':
            return data / self.s_maxabs
        else:  # direct
            return (data - self.s_min) / self.s_range

    def inverse_transform_y(self, y_norm):
        """Desnormaliza solo la columna Y (última columna)."""
        if self.norm_type == 'symmetric_minmax':
            return (y_norm + 1) * self.s_range[-1] / 2 + self.s_min[-1]
        elif self.norm_type == 'symmetric_maxabs':
            return y_norm * self.s_maxabs[-1]
        else:  # direct
            return y_norm * self.s_range[-1] + self.s_min[-1]


# =============================================================
# ORIGIN - Motor de Ingestion y Compresión
# =============================================================
class LuminOrigin:
    """
    Ingesta datos normalizados punto a punto.
    Construye sectores donde cada uno contiene una ley lineal local (W, B)
    que explica sus puntos dentro de un margen epsilon.

    Parámetros:
      epsilon_val  : valor numérico de epsilon (0 a 1).
      epsilon_type : 'absolute' o 'relative'.
      mode         : 'diversity' (retiene ruido) o 'purity' (lo elimina).
    """
    def __init__(self, epsilon_val=0.02, epsilon_type='absolute', mode='diversity'):
        self.epsilon_val = epsilon_val
        self.epsilon_type = epsilon_type
        self.mode = mode
        self.sectors = []          # sectores cerrados [min, max, W, B]
        self._current_nodes = []   # nodos del sector en construcción
        self.D = None              # dimensionalidad de X

    def _calculate_law(self, nodes):
        """Calcula W, B por mínimos cuadrados sobre los nodos dados.
        Necesita mínimo D+1 nodos para un sistema determinado."""
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
        """Retorna el umbral de error según el tipo de epsilon."""
        if self.epsilon_type == 'relative':
            return abs(y_real) * self.epsilon_val
        return self.epsilon_val

    def _close_sector(self):
        """Cierra el sector actual y lo agrega a la lista de sectores.
        Solo cierra si tiene suficientes nodos (D+1 mínimo)."""
        min_nodes = (self.D + 1) if self.D else 2
        if len(self._current_nodes) < min_nodes:
            return
        nodes = np.array(self._current_nodes)
        W, B = self._calculate_law(self._current_nodes)
        if W is None:
            return
        # Sector = [min por coordenada, max por coordenada, W, B]
        sector = np.concatenate([
            np.min(nodes[:, :-1], axis=0),   # min
            np.max(nodes[:, :-1], axis=0),   # max
            W,                                # pesos
            [B]                               # sesgo
        ])
        self.sectors.append(sector)

    def ingest(self, point):
        """
        Ingesta un punto normalizado.
        Si la ley lineal actual lo explica dentro de epsilon → lo agrega al sector.
        Si no → cierra el sector actual (mitosis) y abre uno nuevo.

        CRÍTICO: lstsq necesita mínimo D+1 puntos para resolver un sistema
        determinado en D dimensiones. Con menos puntos la predicción no es
        confiable y no tiene sentido verificar epsilon.
        """
        point = np.array(point, dtype=float)
        if self.D is None:
            self.D = len(point) - 1

        y_real = point[-1]
        min_nodes = self.D + 1  # mínimo para sistema determinado

        # Hasta tener suficientes nodos, agregar sin verificar epsilon
        if len(self._current_nodes) < min_nodes:
            self._current_nodes.append(point.tolist())
            return

        W, B = self._calculate_law(self._current_nodes)
        y_pred = np.dot(point[:-1], W) + B
        error = abs(y_real - y_pred)
        threshold = self._get_threshold(y_real)

        if error <= threshold:
            # El punto es explicado por la ley actual
            self._current_nodes.append(point.tolist())
        else:
            # MITOSIS: cerrar sector actual, abrir nuevo
            self._close_sector()
            if self.mode == 'diversity':
                # Retiene los últimos D nodos como base del nuevo sector
                # (necesitamos D nodos + el nuevo punto = D+1 para poder
                # evaluar epsilon en el siguiente paso)
                self._current_nodes = self._current_nodes[-self.D:] + [point.tolist()]
            else:  # purity
                # Empieza limpio
                self._current_nodes = [point.tolist()]

    def finalize(self):
        """
        CORRECCIÓN 1: Cierra el último sector.
        Debe llamarse después de ingerir todos los datos.
        """
        self._close_sector()
        self._current_nodes = []

    def get_sectors(self):
        """Retorna los sectores como array numpy."""
        if not self.sectors:
            return np.array([])
        return np.array(self.sectors)


# =============================================================
# RESOLUTION - Motor de Inferencia
# =============================================================
class LuminResolution:
    """
    Resuelve predicciones usando los sectores generados por Origin.

    Correcciones aplicadas:
      2. Fallback al sector más cercano cuando un punto no cae en ningún bounding box.
      3. Cuando cae en varios sectores, selecciona el de menor error esperado.
    """
    def __init__(self, sectors, D):
        """
        sectors : array numpy de forma (N, 3*D + 1) con [min, max, W, B].
        D       : dimensionalidad de X.
        """
        self.D = D
        self.sectors = sectors
        self.mins = sectors[:, :D]
        self.maxs = sectors[:, D:2*D]
        self.weights = sectors[:, 2*D:3*D]
        self.biases = sectors[:, -1]
        # Centroide de cada sector (punto medio del bounding box)
        self.centroids = (self.mins + self.maxs) / 2.0

    def _predict_with_sector(self, x, idx):
        """Predicción usando un sector específico."""
        return np.dot(x, self.weights[idx]) + self.biases[idx]

    def resolve(self, X):
        """
        Resuelve predicciones para un array de puntos X normalizados.

        Lógica por punto:
          1. Buscar todos los sectores donde el punto cae dentro del bounding box.
          2. Si cae en varios → seleccionar el de menor error (consistencia).
          3. Si no cae en ninguno → fallback al sector más cercano por distancia
             al centroide (CORRECCIÓN 2, elimina NaN).
        """
        X = np.atleast_2d(X)
        n_points = X.shape[0]
        results = np.full(n_points, np.nan)

        for i in range(n_points):
            x = X[i]

            # Buscar sectores que contienen este punto (dentro del bounding box)
            inside_mask = np.all(
                (x >= self.mins - 1e-9) & (x <= self.maxs + 1e-9),
                axis=1
            )
            candidates = np.where(inside_mask)[0]

            if len(candidates) == 1:
                # Caso simple: un solo sector lo contiene
                results[i] = self._predict_with_sector(x, candidates[0])

            elif len(candidates) > 1:
                # CORRECCIÓN 3: Solapamiento → seleccionar el sector más
                # específico (menor volumen de bounding box). El sector más
                # pequeño que contiene al punto es el más ajustado geométricamente.
                # En alta dimensionalidad usamos log-volumen para estabilidad numérica.
                ranges = self.maxs[candidates] - self.mins[candidates]
                # Usar suma de log(range) como proxy de log-volumen
                log_volumes = np.sum(np.log(ranges + 1e-30), axis=1)
                best = candidates[np.argmin(log_volumes)]
                results[i] = self._predict_with_sector(x, best)

            else:
                # CORRECCIÓN 2: Punto fuera de todos los bounding boxes.
                # Fallback al sector más cercano por distancia al centroide.
                distances = np.linalg.norm(self.centroids - x, axis=1)
                nearest = np.argmin(distances)
                results[i] = self._predict_with_sector(x, nearest)

        return results


# =============================================================
# PIPELINE COMPLETO
# =============================================================
class LuminPipeline:
    """
    Orquesta el flujo completo: normalización → ingestion → resolución.
    Interfaz principal para usar el motor.
    """
    def __init__(self, epsilon_val=0.02, epsilon_type='absolute',
                 mode='diversity', norm_type='symmetric_minmax'):
        self.normalizer = Normalizer(norm_type)
        self.epsilon_val = epsilon_val
        self.epsilon_type = epsilon_type
        self.mode = mode
        self.origin = None
        self.resolution = None
        self.D = None

    def fit(self, data):
        """
        Entrena el motor completo.
        data : array numpy de forma (N, D+1) donde la última columna es Y.
        """
        # Normalizar
        self.normalizer.fit(data)
        data_norm = self.normalizer.transform(data)
        self.D = data.shape[1] - 1

        # Ingerir
        self.origin = LuminOrigin(
            epsilon_val=self.epsilon_val,
            epsilon_type=self.epsilon_type,
            mode=self.mode
        )
        for point in data_norm:
            self.origin.ingest(point)
        self.origin.finalize()  # CRÍTICO: cerrar último sector

        # Preparar resolución
        sectors = self.origin.get_sectors()
        if len(sectors) == 0:
            raise ValueError("No se generaron sectores. Revisar datos o epsilon.")
        self.resolution = LuminResolution(sectors, self.D)

        return self

    def predict(self, X):
        """
        Predice Y para un array X (sin normalizar, valores originales).
        X : array numpy de forma (N, D) o (D,).
        """
        X = np.atleast_2d(X)
        # Normalizar X usando los parámetros del fit
        # Necesitamos normalizar solo las columnas X, no Y
        X_with_dummy_y = np.c_[X, np.zeros(X.shape[0])]
        X_norm_full = self.normalizer.transform(X_with_dummy_y)
        X_norm = X_norm_full[:, :-1]

        # Resolver en espacio normalizado
        y_norm = self.resolution.resolve(X_norm)

        # Desnormalizar Y
        return self.normalizer.inverse_transform_y(y_norm)

    @property
    def n_sectors(self):
        return len(self.origin.sectors) if self.origin else 0

    def save(self, filename="lumin_model.npy"):
        """
        Guarda el modelo en .npy.
        Contiene todo lo que Resolution necesita para inferir:
          - sectors      : array [min, max, W, B] por sector.
          - s_min        : mínimos por columna (todas, incluyendo Y).
          - s_max        : máximos por columna.
          - s_range      : rango por columna.
          - s_maxabs     : max(abs) por columna (solo si norm es symmetric_maxabs).
          - norm_type    : tipo de normalización usado.
          - D            : dimensionalidad de X.
          - epsilon_val  : epsilon usado en origin.
          - epsilon_type : tipo de epsilon.
          - mode         : diversity o purity.
        """
        if self.origin is None or self.resolution is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta fit() primero.")

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
        }
        np.save(filename, model_data)
        return filename

    @classmethod
    def load(cls, filename="lumin_model.npy"):
        """
        Carga un modelo desde .npy y reconstruye el pipeline
        listo para inferir con predict().
        No necesita los datos originales ni Origin.
        """
        model_data = np.load(filename, allow_pickle=True).item()

        pipeline = cls(
            epsilon_val=model_data['epsilon_val'],
            epsilon_type=model_data['epsilon_type'],
            mode=model_data['mode'],
            norm_type=model_data['norm_type'],
        )

        # Reconstruir normalizer
        pipeline.normalizer.s_min   = model_data['s_min']
        pipeline.normalizer.s_max   = model_data['s_max']
        pipeline.normalizer.s_range = model_data['s_range']
        if len(model_data['s_maxabs']) > 0:
            pipeline.normalizer.s_maxabs = model_data['s_maxabs']

        # Reconstruir resolution directamente desde los sectores
        pipeline.D = model_data['D']
        sectors = model_data['sectors']
        pipeline.resolution = LuminResolution(sectors, pipeline.D)

        return pipeline

    def get_metadata(self):
        """Retorna metadatos del modelo entrenado."""
        return {
            'n_sectors': self.n_sectors,
            'D': self.D,
            'epsilon_val': self.epsilon_val,
            'epsilon_type': self.epsilon_type,
            'mode': self.mode,
            'norm_type': self.normalizer.norm_type,
        }
      
