"""
Predicción sobre datos nuevos: carga un Excel, lo transforma
y aplica un modelo entrenado para generar predicciones.
"""
import numpy as np
from data_loader import cargar_dataset
from data_transformer import transformar_dataset, FEATURES_FINALES
from utils import info, error


def predecir(ruta_excel: str, modelo, tarea: str = 'clasificacion') -> np.ndarray | None:
    df = cargar_dataset(ruta_excel)
    if df is None:
        return None

    df = transformar_dataset(df)
    features_presentes = [f for f in FEATURES_FINALES if f in df.columns]
    if len(features_presentes) < len(FEATURES_FINALES):
        faltantes = set(FEATURES_FINALES) - set(features_presentes)
        error(f"Faltan columnas en el dataset nuevo: {faltantes}")
        return None

    X = df[FEATURES_FINALES].values

    try:
        if tarea == 'clasificacion' and hasattr(modelo, 'predict_proba'):
            pred = (modelo.predict(X) > 0.5).astype(int).flatten()
        elif tarea == 'clasificacion':
            pred = modelo.predict(X).flatten()
        else:
            pred = modelo.predict(X).flatten()
        info(f"Prediccion completada: {len(pred)} registros")
        return pred
    except Exception as e:
        error(f"Error al predecir: {e}")
        return None
