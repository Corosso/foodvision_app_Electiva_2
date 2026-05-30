"""
Modelos de Machine Learning tradicional:
- Clasificación: Árbol, Random Forest, Regresión Logística, KNN
- Regresión: Árbol, Random Forest, KNN
"""
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from utils import info

RANDOM_STATE = 42


def entrenar_clasificacion(X: np.ndarray, y: np.ndarray,
                           test_size: float = 0.2) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)

    modelos = {
        'Arbol de Decision': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'Regresion Logistica': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'KNN': KNeighborsClassifier(n_neighbors=5),
    }
    resultados = {}
    predicciones = {}

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        predicciones[nombre] = y_pred
        resultados[nombre] = {'modelo': modelo}

    info(f"Modelos de clasificacion entrenados: {len(modelos)} ({X_train.shape[0]} train, {X_test.shape[0]} test)")
    return {
        'modelos': modelos,
        'predicciones': predicciones,
        'resultados': resultados,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
    }


def entrenar_regresion(X: np.ndarray, y: np.ndarray,
                       test_size: float = 0.2) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE)

    modelos = {
        'Arbol de Decision': DecisionTreeRegressor(random_state=RANDOM_STATE),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        'KNN': KNeighborsRegressor(n_neighbors=5),
    }
    resultados = {}
    predicciones = {}

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        predicciones[nombre] = y_pred
        resultados[nombre] = {'modelo': modelo}

    info(f"Modelos de regresion entrenados: {len(modelos)} ({X_train.shape[0]} train, {X_test.shape[0]} test)")
    return {
        'modelos': modelos,
        'predicciones': predicciones,
        'resultados': resultados,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
    }
