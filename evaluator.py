"""
Evaluación de modelos:
- Clasificación: accuracy, precision, recall, F1, matriz de confusión
- Regresión: MAE, MSE, RMSE, R²
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from utils import info


def evaluar_clasificacion(y_test: np.ndarray,
                          predicciones: dict[str, np.ndarray]) -> pd.DataFrame:
    filas = {}
    for nombre, y_pred in predicciones.items():
        filas[nombre] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        }
    df = pd.DataFrame(filas).T
    return df


def reporte_clasificacion(y_test: np.ndarray, y_pred: np.ndarray,
                          nombre_modelo: str = ""):
    labels = sorted(np.unique(np.concatenate([y_test, y_pred])))
    if len(labels) == 2:
        target_names = ['No demora', 'Si demora']
    else:
        target_names = [str(l) for l in labels]
    info(f"Matriz de confusion ({nombre_modelo}):")
    print(confusion_matrix(y_test, y_pred, labels=labels))
    info(f"Classification report ({nombre_modelo}):")
    print(classification_report(y_test, y_pred, labels=labels,
                                target_names=target_names,
                                zero_division=0))


def evaluar_regresion(y_test: np.ndarray,
                      predicciones: dict[str, np.ndarray]) -> pd.DataFrame:
    filas = {}
    for nombre, y_pred in predicciones.items():
        mse = mean_squared_error(y_test, y_pred)
        filas[nombre] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'MSE': mse,
            'RMSE': np.sqrt(mse),
            'R2': r2_score(y_test, y_pred),
        }
    df = pd.DataFrame(filas).T
    return df


def tabla_comparativa(df_ml: pd.DataFrame, df_dl: pd.DataFrame,
                      tarea: str = 'clasificacion') -> pd.DataFrame:
    if tarea == 'clasificacion':
        metrica = 'F1-Score'
    else:
        metrica = 'R2'
    mejor_ml = df_ml[metrica].idxmax() if not df_ml.empty else 'N/A'
    mejor_ml_val = df_ml[metrica].max() if not df_ml.empty else 0

    info(f"Mejor modelo ML ({tarea}): {mejor_ml} ({metrica}={mejor_ml_val:.4f})")
    return df_ml
