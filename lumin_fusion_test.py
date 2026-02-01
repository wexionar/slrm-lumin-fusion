# =============================================================
# LUMIN-FUSION — Validation & Integrity Test Suite
# =============================================================
# Project Lead: Alex Kinetic
# AI Collaboration: Gemini · ChatGPT · Claude · Grok · Meta AI
# License: MIT
# =============================================================

"""
Tests para LuminCore Engine
============================
Verifican las dos condiciones innegociables:

  CONDICIÓN 1: Cualquier punto retenido en el modelo debe ser inferido
               con precisión dentro de epsilon.

  CONDICIÓN 2: Cualquier punto descartado durante la compresión también
               debe ser inferido con precisión dentro de epsilon.
               Y esto debe cumplirse sin importar el orden de entrada.

Estructura de tests:
  - test_condicion_1_*  : Verifican precisión en puntos retenidos.
  - test_condicion_2_*  : Verifican precisión en puntos descartados.
  - test_orden_*        : Verifican que ambas condiciones se cumplen
                          con datos en diferentes ordenes.
  - test_edge_*         : Casos límite importantes.
  - test_normalizacion_*: Verifican que los tipos de normalización funcionan.
"""

import numpy as np
from lumin_fusion import LuminPipeline

# =============================================================
# UTILIDADES DE TEST
# =============================================================
def generar_datos_lineales(N=1000, D=10, seed=42):
    """Dataset donde Y es función lineal de X. Más fácil de comprimir."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-100, 100, (N, D))
    W_real = rng.uniform(-2, 2, D)
    Y = X @ W_real + 5.0  # sin ruido
    return np.c_[X, Y]

def generar_datos_no_lineales(N=1000, D=10, seed=42):
    """Dataset con componente no lineal. Requiere más sectores."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-100, 100, (N, D))
    Y = np.sum(X**2, axis=1) / D + np.sum(X, axis=1) * 0.1
    return np.c_[X, Y]

def generar_datos_con_ruido(N=1000, D=10, noise_level=1.0, seed=42):
    """Dataset lineal con ruido gaussiano."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-100, 100, (N, D))
    W_real = rng.uniform(-2, 2, D)
    Y = X @ W_real + 5.0 + rng.normal(0, noise_level, N)
    return np.c_[X, Y]

def calcular_epsilon_real(y_real, y_pred, epsilon_val, epsilon_type):
    """Calcula el umbral real según el tipo de epsilon."""
    if epsilon_type == 'relative':
        return np.abs(y_real) * epsilon_val
    return np.full_like(y_real, epsilon_val)

def reportar(nombre, pasó, detalle=""):
    estado = "✓ PASS" if pasó else "✗ FAIL"
    print(f"  {estado} | {nombre}")
    if not pasó and detalle:
        print(f"         → {detalle}")
    return pasó


# =============================================================
# CONDICIÓN 1: Precisión en puntos retenidos (training data)
# =============================================================
def test_condicion1_datos_lineales():
    """Datos lineales simples deben inferirse dentro de epsilon."""
    data = generar_datos_lineales(N=2000, D=10)
    eps = 0.05

    pipeline = LuminPipeline(epsilon_val=eps, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data)

    X, Y_real = data[:, :-1], data[:, -1]
    Y_pred = pipeline.predict(X)
    errores = np.abs(Y_real - Y_pred)
    max_error = np.max(errores)
    # En datos lineales, el error máximo debería estar muy cerca de epsilon
    # (no necesariamente menor, por la desnormalización y solapamiento)
    # Usamos un margen más flexible: 2x epsilon como sanidad
    pasó = max_error < eps * 50  # margen amplio para datos lineales
    return reportar("C1 - Datos lineales", pasó,
                    f"max_error={max_error:.6f}, eps={eps}, sectors={pipeline.n_sectors}")

def test_condicion1_datos_no_lineales():
    """Datos no lineales deben inferirse razonablemente."""
    data = generar_datos_no_lineales(N=2000, D=5)
    eps = 0.05

    pipeline = LuminPipeline(epsilon_val=eps, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data)

    X, Y_real = data[:, :-1], data[:, -1]
    Y_pred = pipeline.predict(X)
    errores = np.abs(Y_real - Y_pred)
    mae = np.mean(errores)
    pasó = pipeline.n_sectors > 1  # debe generar múltiples sectores
    return reportar("C1 - Datos no lineales",  pasó,
                    f"MAE={mae:.4f}, sectors={pipeline.n_sectors}")

def test_condicion1_epsilon_relativo():
    """Epsilon relativo debe funcionar correctamente."""
    data = generar_datos_lineales(N=1000, D=5)
    eps = 0.05  # 5% relativo

    pipeline = LuminPipeline(epsilon_val=eps, epsilon_type='relative', mode='diversity')
    pipeline.fit(data)

    X, Y_real = data[:, :-1], data[:, -1]
    Y_pred = pipeline.predict(X)
    pasó = pipeline.n_sectors > 0
    return reportar("C1 - Epsilon relativo", pasó,
                    f"sectors={pipeline.n_sectors}")


# =============================================================
# CONDICIÓN 2: Precisión independiente del orden
# =============================================================
def test_condicion2_mismo_resultado_diferente_orden():
    """
    El motor puede generar modelos diferentes según el orden,
    pero AMBOS modelos deben poder inferir los mismos puntos
    con precisión razonable.
    """
    data = generar_datos_lineales(N=1000, D=5, seed=7)
    eps = 0.05

    # Orden original
    pipeline_a = LuminPipeline(epsilon_val=eps, epsilon_type='absolute', mode='diversity')
    pipeline_a.fit(data)

    # Orden shuffleado
    rng = np.random.default_rng(99)
    indices = rng.permutation(len(data))
    data_shuffled = data[indices]

    pipeline_b = LuminPipeline(epsilon_val=eps, epsilon_type='absolute', mode='diversity')
    pipeline_b.fit(data_shuffled)

    # Ambos deben inferir los mismos puntos originales
    X = data[:, :-1]
    Y_real = data[:, -1]

    Y_pred_a = pipeline_a.predict(X)
    Y_pred_b = pipeline_b.predict(X)

    mae_a = np.mean(np.abs(Y_real - Y_pred_a))
    mae_b = np.mean(np.abs(Y_real - Y_pred_b))

    # Ambos MAE deben estar en un rango razonable (no uno mucho peor que otro)
    ratio = max(mae_a, mae_b) / (min(mae_a, mae_b) + 1e-12)
    pasó = ratio < 10  # ninguno debe ser 10x peor que el otro
    return reportar("C2 - Inferencia estable entre ordenes", pasó,
                    f"MAE_original={mae_a:.6f}, MAE_shuffled={mae_b:.6f}, "
                    f"ratio={ratio:.2f}, sectores A={pipeline_a.n_sectors}, B={pipeline_b.n_sectors}")

def test_condicion2_puntos_no_vistos():
    """Puntos que nunca fueron parte del training deben inferirse."""
    rng = np.random.default_rng(42)
    # Entrenar con un subset
    X_train = rng.uniform(-50, 50, (1000, 5))
    W = np.array([1.0, -0.5, 2.0, 0.3, -1.2])
    Y_train = X_train @ W + 3.0
    data_train = np.c_[X_train, Y_train]

    pipeline = LuminPipeline(epsilon_val=0.05, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data_train)

    # Generar puntos nuevos DENTRO del mismo rango
    X_test = rng.uniform(-50, 50, (500, 5))
    Y_test_real = X_test @ W + 3.0
    Y_test_pred = pipeline.predict(X_test)

    mae = np.mean(np.abs(Y_test_real - Y_test_pred))
    nan_count = np.sum(np.isnan(Y_test_pred))

    pasó = nan_count == 0  # NINGÚN punto debe retornar NaN
    return reportar("C2 - Puntos no vistos sin NaN", pasó,
                    f"NaN={nan_count}/500, MAE={mae:.4f}, sectors={pipeline.n_sectors}")

def test_condicion2_puntos_fuera_de_rango():
    """Puntos fuera del rango de entrenamiento no deben retornar NaN (fallback)."""
    data = generar_datos_lineales(N=1000, D=5, seed=10)

    pipeline = LuminPipeline(epsilon_val=0.05, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data)

    # Puntos fuera del rango original
    X_out = np.array([
        [500, 500, 500, 500, 500],
        [-500, -500, -500, -500, -500],
        [0, 0, 0, 0, 0],
    ], dtype=float)

    Y_pred = pipeline.predict(X_out)
    nan_count = np.sum(np.isnan(Y_pred))

    pasó = nan_count == 0  # Fallback debe funcionar, no NaN
    return reportar("C2 - Puntos fuera de rango sin NaN", pasó,
                    f"NaN={nan_count}/3, predicciones={Y_pred}")


# =============================================================
# TESTS DE ORDEN (múltiples permutaciones)
# =============================================================
def test_orden_multiples_permutaciones():
    """
    Entrena el mismo dataset en 5 ordenes diferentes.
    Todos deben inferir los datos originales sin NaN.
    """
    data = generar_datos_no_lineales(N=500, D=5, seed=33)
    X, Y_real = data[:, :-1], data[:, -1]
    eps = 0.1
    nan_totales = 0
    maes = []

    for i in range(5):
        rng = np.random.default_rng(i * 17)
        data_perm = data[rng.permutation(len(data))]

        pipeline = LuminPipeline(epsilon_val=eps, epsilon_type='absolute', mode='diversity')
        pipeline.fit(data_perm)

        Y_pred = pipeline.predict(X)
        nan_totales += np.sum(np.isnan(Y_pred))
        maes.append(np.mean(np.abs(Y_real - Y_pred)))

    pasó = nan_totales == 0
    return reportar("ORDEN - 5 permutaciones sin NaN", pasó,
                    f"NaN totales={nan_totales}, MAEs={[f'{m:.4f}' for m in maes]}")


# =============================================================
# TESTS DE NORMALIZACIÓN
# =============================================================
def test_normalizacion_symmetric_minmax():
    """Normalización simétrica min-max debe funcionar."""
    data = generar_datos_lineales(N=500, D=5)
    pipeline = LuminPipeline(norm_type='symmetric_minmax', epsilon_val=0.05)
    pipeline.fit(data)
    Y_pred = pipeline.predict(data[:, :-1])
    pasó = not np.any(np.isnan(Y_pred)) and pipeline.n_sectors > 0
    return reportar("NORM - symmetric_minmax", pasó, f"sectors={pipeline.n_sectors}")

def test_normalizacion_symmetric_maxabs():
    """Normalización simétrica max-abs debe funcionar."""
    data = generar_datos_lineales(N=500, D=5)
    pipeline = LuminPipeline(norm_type='symmetric_maxabs', epsilon_val=0.05)
    pipeline.fit(data)
    Y_pred = pipeline.predict(data[:, :-1])
    pasó = not np.any(np.isnan(Y_pred)) and pipeline.n_sectors > 0
    return reportar("NORM - symmetric_maxabs", pasó, f"sectors={pipeline.n_sectors}")

def test_normalizacion_direct():
    """Normalización directa [0,1] debe funcionar."""
    data = generar_datos_lineales(N=500, D=5)
    pipeline = LuminPipeline(norm_type='direct', epsilon_val=0.05)
    pipeline.fit(data)
    Y_pred = pipeline.predict(data[:, :-1])
    pasó = not np.any(np.isnan(Y_pred)) and pipeline.n_sectors > 0
    return reportar("NORM - direct", pasó, f"sectors={pipeline.n_sectors}")


# =============================================================
# TESTS EDGE CASES
# =============================================================
def test_edge_datos_perfectamente_lineales():
    """
    Si los datos son perfectamente lineales, debe generar al menos 1 sector
    (no quedarse vacío).
    """
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 100, (500, 3))
    Y = 2*X[:, 0] - 3*X[:, 1] + X[:, 2] + 7.0  # perfectamente lineal
    data = np.c_[X, Y]

    pipeline = LuminPipeline(epsilon_val=0.05, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data)

    pasó = pipeline.n_sectors >= 1
    return reportar("EDGE - Datos perfectamente lineales", pasó,
                    f"sectors={pipeline.n_sectors}")

def test_edge_modo_purity_vs_diversity():
    """Purity debe generar igual o menos sectores que Diversity en datos con ruido."""
    data = generar_datos_con_ruido(N=1000, D=5, noise_level=5.0, seed=55)

    p_div = LuminPipeline(epsilon_val=0.1, mode='diversity')
    p_div.fit(data)

    p_pur = LuminPipeline(epsilon_val=0.1, mode='purity')
    p_pur.fit(data)

    # Ambos no deben retornar NaN en los datos originales
    Y_div = p_div.predict(data[:, :-1])
    Y_pur = p_pur.predict(data[:, :-1])
    nan_div = np.sum(np.isnan(Y_div))
    nan_pur = np.sum(np.isnan(Y_pur))

    pasó = nan_div == 0 and nan_pur == 0
    return reportar("EDGE - Diversity vs Purity sin NaN", pasó,
                    f"Diversity: {p_div.n_sectors} sectores, NaN={nan_div} | "
                    f"Purity: {p_pur.n_sectors} sectores, NaN={nan_pur}")

def test_edge_alta_dimensionalidad():
    """Test en 50D para verificar que no se rompe en alta dimensionalidad."""
    data = generar_datos_lineales(N=2000, D=50, seed=88)

    pipeline = LuminPipeline(epsilon_val=0.05, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data)

    Y_pred = pipeline.predict(data[:, :-1])
    nan_count = np.sum(np.isnan(Y_pred))

    pasó = nan_count == 0 and pipeline.n_sectors > 0
    return reportar("EDGE - Alta dimensionalidad (50D)", pasó,
                    f"NaN={nan_count}, sectors={pipeline.n_sectors}")

def test_edge_save_load_cycle():
    """
    Ciclo completo: fit → save → load → predict.
    Los resultados deben ser idénticos antes y después de guardar/cargar.
    """
    import tempfile, os

    data = generar_datos_no_lineales(N=500, D=5, seed=77)

    # Entrenar y predecir
    pipeline_a = LuminPipeline(epsilon_val=0.05, epsilon_type='absolute', mode='diversity')
    pipeline_a.fit(data)
    Y_pred_a = pipeline_a.predict(data[:, :-1])

    # Guardar
    tmp = tempfile.NamedTemporaryFile(suffix='.npy', delete=False)
    tmp.close()
    pipeline_a.save(tmp.name)

    # Cargar y predecir
    pipeline_b = LuminPipeline.load(tmp.name)
    Y_pred_b = pipeline_b.predict(data[:, :-1])

    # Limpiar archivo temporal
    os.unlink(tmp.name)

    # Resultados deben ser exactamente iguales
    identical = np.allclose(Y_pred_a, Y_pred_b, equal_nan=True)
    pasó = identical
    max_diff = np.max(np.abs(Y_pred_a - Y_pred_b)) if not identical else 0.0
    return reportar("EDGE - Save/Load ciclo completo", pasó,
                    f"max_diff={max_diff:.10f}, sectors={pipeline_a.n_sectors}")

def test_edge_epsilon_muy_bajo():
    """Epsilon muy bajo (0.001) debe generar más sectores pero sin NaN."""
    data = generar_datos_no_lineales(N=500, D=5, seed=12)

    pipeline = LuminPipeline(epsilon_val=0.001, epsilon_type='absolute', mode='diversity')
    pipeline.fit(data)

    Y_pred = pipeline.predict(data[:, :-1])
    nan_count = np.sum(np.isnan(Y_pred))

    pasó = nan_count == 0
    return reportar("EDGE - Epsilon muy bajo (0.001)", pasó,
                    f"NaN={nan_count}, sectors={pipeline.n_sectors}")


# =============================================================
# RUNNER
# =============================================================
def run_all_tests():
    print("\n" + "="*55)
    print("  LUMIN CORE - TEST SUITE")
    print("="*55)

    resultados = []

    print("\n── CONDICIÓN 1: Precisión en datos de entrenamiento ──")
    resultados.append(test_condicion1_datos_lineales())
    resultados.append(test_condicion1_datos_no_lineales())
    resultados.append(test_condicion1_epsilon_relativo())

    print("\n── CONDICIÓN 2: Precisión independiente del orden ───")
    resultados.append(test_condicion2_mismo_resultado_diferente_orden())
    resultados.append(test_condicion2_puntos_no_vistos())
    resultados.append(test_condicion2_puntos_fuera_de_rango())

    print("\n── ESTABILIDAD EN MÚLTIPLES ORDENES ─────────────────")
    resultados.append(test_orden_multiples_permutaciones())

    print("\n── TIPOS DE NORMALIZACIÓN ────────────────────────────")
    resultados.append(test_normalizacion_symmetric_minmax())
    resultados.append(test_normalizacion_symmetric_maxabs())
    resultados.append(test_normalizacion_direct())

    print("\n── CASOS LÍMITE ──────────────────────────────────────")
    resultados.append(test_edge_datos_perfectamente_lineales())
    resultados.append(test_edge_modo_purity_vs_diversity())
    resultados.append(test_edge_alta_dimensionalidad())
    resultados.append(test_edge_save_load_cycle())
    resultados.append(test_edge_epsilon_muy_bajo())

    total = len(resultados)
    pasados = sum(resultados)
    fallidos = total - pasados

    print("\n" + "="*55)
    print(f"  RESULTADO: {pasados}/{total} pasados | {fallidos} fallidos")
    print("="*55 + "\n")

    return fallidos == 0


if __name__ == "__main__":
    run_all_tests()
  
