"""
Modelos de Deep Learning con TensorFlow/Keras:
- Red Básica: 1 capa oculta con 64 neuronas
- Red Profunda: 128->64->32 con Dropout y EarlyStopping
"""
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from utils import info

RANDOM_STATE = 42
EPOCHS = 80
BATCH_SIZE = 32

tf.random.set_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


def crear_red_basica(input_dim: int, tarea: str = 'clasificacion') -> keras.Sequential:
    m = keras.Sequential([layers.Dense(64, activation='relu', input_shape=(input_dim,))])
    if tarea == 'clasificacion':
        m.add(layers.Dense(1, activation='sigmoid'))
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        m.add(layers.Dense(1, activation='linear'))
        m.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return m


def crear_red_profunda(input_dim: int, tarea: str = 'clasificacion') -> keras.Sequential:
    m = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
    ])
    if tarea == 'clasificacion':
        m.add(layers.Dense(1, activation='sigmoid'))
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        m.add(layers.Dense(1, activation='linear'))
        m.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return m


def _entrenar_red(red: keras.Sequential, X_train, y_train, X_test, y_test,
                  tarea: str) -> dict:
    early_stop = EarlyStopping(monitor='val_loss', patience=10,
                               restore_best_weights=True, verbose=0)
    inicio = time.time()
    hist = red.fit(X_train, y_train, validation_data=(X_test, y_test),
                   epochs=EPOCHS, batch_size=BATCH_SIZE,
                   callbacks=[early_stop], verbose=0)
    tiempo = time.time() - inicio

    if tarea == 'clasificacion':
        y_pred = (red.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    else:
        y_pred = red.predict(X_test, verbose=0).flatten()

    return {
        'modelo': red,
        'historial': hist.history,
        'y_pred': y_pred,
        'tiempo': tiempo,
        'epochs_reales': len(hist.history['loss']),
    }


def entrenar_dl_clasificacion(X: np.ndarray, y: np.ndarray,
                              test_size: float = 0.2) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    input_dim = X_train.shape[1]

    info("Entrenando Red Basica (clasificacion)...")
    basica = crear_red_basica(input_dim, 'clasificacion')
    r_basica = _entrenar_red(basica, X_train, y_train, X_test, y_test, 'clasificacion')

    info("Entrenando Red Profunda (clasificacion)...")
    profunda = crear_red_profunda(input_dim, 'clasificacion')
    r_profunda = _entrenar_red(profunda, X_train, y_train, X_test, y_test, 'clasificacion')

    return {
        'Red Basica': r_basica,
        'Red Profunda': r_profunda,
        'X_test': X_test, 'y_test': y_test,
    }


def entrenar_dl_regresion(X: np.ndarray, y: np.ndarray,
                          test_size: float = 0.2) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE)
    input_dim = X_train.shape[1]

    info("Entrenando Red Basica (regresion)...")
    basica = crear_red_basica(input_dim, 'regresion')
    r_basica = _entrenar_red(basica, X_train, y_train, X_test, y_test, 'regresion')

    info("Entrenando Red Profunda (regresion)...")
    profunda = crear_red_profunda(input_dim, 'regresion')
    r_profunda = _entrenar_red(profunda, X_train, y_train, X_test, y_test, 'regresion')

    return {
        'Red Basica': r_basica,
        'Red Profunda': r_profunda,
        'X_test': X_test, 'y_test': y_test,
    }
