"""
FoodVision AI - Sistema de prediccion para restaurantes.
Aplicacion de consola que entrena y compara modelos ML y DL
para predecir retrasos en entregas y ventas diarias.
"""
import sys
import numpy as np
from data_loader import cargar_dataset
from data_cleaner import limpiar_dataset
from data_transformer import transformar_dataset, FEATURES_FINALES, TARGETS
from ml_models import entrenar_clasificacion, entrenar_regresion
from dl_models import entrenar_dl_clasificacion, entrenar_dl_regresion
from evaluator import (evaluar_clasificacion, evaluar_regresion,
                       reporte_clasificacion)
from visualizer import (matrices_confusion, barras_metricas,
                        scatter_regresion, barras_errores_regresion,
                        curvas_entrenamiento_dl, comparacion_dl_clasificacion,
                        scatter_regresion_dl, comparacion_dl_regresion)
from predictor import predecir
from utils import (limpiar_pantalla, titulo, info, error, pausa,
                   RUTA_DATASET, ruta_output)

# Estado global compartido entre opciones del menú
df = None
df_clean = None
df_trans = None
resultado_ml = None
resultado_dl = None
tarea_actual = None


def menu_principal():
    limpiar_pantalla()
    titulo("FoodVision AI - Sistema de Prediccion para Restaurantes")
    print("""
  1. Cargar dataset
  2. Mostrar dataset original
  3. Limpiar datos
  4. Transformar datos
  5. Entrenar modelos ML
  6. Entrenar modelos DL
  7. Comparar metricas
  8. Metricas detalladas de un modelo
  9. Generar y guardar graficas
  10. Predecir con datos nuevos
  11. Salir
""")
    return input("  Selecciona una opcion (1-11): ").strip()


def opcion_1():
    global df
    ruta = input("  Ruta del archivo Excel [Enter = restaurante.xlsx]: ").strip()
    if not ruta:
        ruta = RUTA_DATASET
    df = cargar_dataset(ruta)
    if df is not None:
        info(f"Columnas: {list(df.columns)}")
    pausa()


def opcion_2():
    if df is None:
        error("Primero carga el dataset (opcion 1).")
    else:
        print(df.head(10).to_string())
        info(f"Forma: {df.shape}")
    pausa()


def opcion_3():
    global df, df_clean
    if df is None:
        error("Primero carga el dataset (opcion 1).")
        pausa(); return
    df_clean = limpiar_dataset(df.copy())
    pausa()


def opcion_4():
    global df_clean, df_trans
    base = df_clean if df_clean is not None else df
    if base is None:
        error("Primero carga el dataset (opcion 1).")
        pausa(); return
    df_trans = transformar_dataset(base.copy())
    if df_trans is not None:
        print(df_trans[FEATURES_FINALES + TARGETS].head().to_string())
    pausa()


def opcion_5():
    global df_trans, resultado_ml, tarea_actual
    if df_trans is None:
        error("Primero transforma los datos (opcion 4).")
        pausa(); return
    print("\n  Tareas disponibles:")
    print("    c = Clasificacion (demora_entrega)")
    print("    r = Regresion (ventas_dia)")
    t = input("  Elige tarea [c/r]: ").strip().lower()

    X = df_trans[FEATURES_FINALES].values
    if t == 'c':
        y = df_trans['demora_entrega'].values
        resultado_ml = entrenar_clasificacion(X, y)
        tarea_actual = 'clasificacion'
    elif t == 'r':
        y = df_trans['ventas_dia'].values
        resultado_ml = entrenar_regresion(X, y)
        tarea_actual = 'regresion'
    else:
        error("Opcion invalida.")
        pausa(); return

    preds = resultado_ml['predicciones']
    y_test = resultado_ml['y_test']
    if tarea_actual == 'clasificacion':
        df_m = evaluar_clasificacion(y_test, preds)
    else:
        df_m = evaluar_regresion(y_test, preds)
    print(df_m.round(4).to_string())
    pausa()


def opcion_6():
    global df_trans, resultado_dl, tarea_actual
    if df_trans is None:
        error("Primero transforma los datos (opcion 4).")
        pausa(); return
    print("\n  Tareas disponibles:")
    print("    c = Clasificacion (demora_entrega)")
    print("    r = Regresion (ventas_dia)")
    t = input("  Elige tarea [c/r]: ").strip().lower()

    X = df_trans[FEATURES_FINALES].values
    if t == 'c':
        y = df_trans['demora_entrega'].values
        resultado_dl = entrenar_dl_clasificacion(X, y)
        tarea_actual = 'clasificacion'

        r_b = resultado_dl['Red Basica']
        r_p = resultado_dl['Red Profunda']
        info(f"Red Basica:  epochs={r_b['epochs_reales']}, tiempo={r_b['tiempo']:.2f}s")
        info(f"Red Profunda: epochs={r_p['epochs_reales']}, tiempo={r_p['tiempo']:.2f}s")
    elif t == 'r':
        y = df_trans['ventas_dia'].values
        resultado_dl = entrenar_dl_regresion(X, y)
        tarea_actual = 'regresion'

        r_b = resultado_dl['Red Basica']
        r_p = resultado_dl['Red Profunda']
        info(f"Red Basica:  tiempo={r_b['tiempo']:.2f}s")
        info(f"Red Profunda: tiempo={r_p['tiempo']:.2f}s")
    else:
        error("Opcion invalida.")
    pausa()


def opcion_7():
    if resultado_ml is None and resultado_dl is None:
        error("Primero entrena modelos (opciones 5 y/o 6).")
        pausa(); return

    titulo("Tabla Comparativa")
    if resultado_ml is not None and tarea_actual == 'clasificacion':
        df_ml = evaluar_clasificacion(resultado_ml['y_test'],
                                       resultado_ml['predicciones'])
        print("--- ML (Clasificacion) ---")
        print(df_ml.round(4).to_string())
    elif resultado_ml is not None:
        df_ml = evaluar_regresion(resultado_ml['y_test'],
                                   resultado_ml['predicciones'])
        print("--- ML (Regresion) ---")
        print(df_ml.round(2).to_string())

    if resultado_dl is not None:
        y_test = resultado_dl['y_test']
        r_b = resultado_dl['Red Basica']
        r_p = resultado_dl['Red Profunda']
        print("\n--- DL ---")
        if tarea_actual == 'clasificacion':
            from sklearn.metrics import accuracy_score, f1_score
            acc_b = accuracy_score(y_test, r_b['y_pred'])
            f1_b = f1_score(y_test, r_b['y_pred'], average='weighted', zero_division=0)
            acc_p = accuracy_score(y_test, r_p['y_pred'])
            f1_p = f1_score(y_test, r_p['y_pred'], average='weighted', zero_division=0)
            info(f"Red Basica:  Acc={acc_b:.4f}, F1={f1_b:.4f}")
            info(f"Red Profunda: Acc={acc_p:.4f}, F1={f1_p:.4f}")
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            rmse_b = np.sqrt(mean_squared_error(y_test, r_b['y_pred']))
            r2_b = r2_score(y_test, r_b['y_pred'])
            rmse_p = np.sqrt(mean_squared_error(y_test, r_p['y_pred']))
            r2_p = r2_score(y_test, r_p['y_pred'])
            info(f"Red Basica:  R²={r2_b:.4f}, RMSE={rmse_b:,.0f}")
            info(f"Red Profunda: R²={r2_p:.4f}, RMSE={rmse_p:,.0f}")
    pausa()


def opcion_8():
    if resultado_ml is None:
        error("Primero entrena modelos ML (opcion 5).")
        pausa(); return

    modelos = list(resultado_ml['predicciones'].keys())
    print("\n  Modelos disponibles:")
    for i, m in enumerate(modelos, 1):
        print(f"    {i}. {m}")
    try:
        idx = int(input("  Elige numero de modelo: ").strip()) - 1
    except ValueError:
        error("Numero invalido."); pausa(); return

    if idx < 0 or idx >= len(modelos):
        error("Numero fuera de rango."); pausa(); return

    nombre = modelos[idx]
    y_test = resultado_ml['y_test']
    y_pred = resultado_ml['predicciones'][nombre]

    if tarea_actual == 'clasificacion':
        reporte_clasificacion(y_test, y_pred, nombre)
    else:
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        info(f"Regresion ({nombre}):")
        info(f"  MAE:  {mean_absolute_error(y_test, y_pred):,.2f}")
        info(f"  MSE:  {mean_squared_error(y_test, y_pred):,.2f}")
        info(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
        info(f"  R²:   {r2_score(y_test, y_pred):.4f}")
    pausa()


def opcion_9():
    if resultado_ml is None and resultado_dl is None:
        error("Primero entrena modelos (opciones 5 y/o 6).")
        pausa(); return

    titulo("Generando graficas...")
    y_test_ml = resultado_ml['y_test'] if resultado_ml else None
    y_test_dl = resultado_dl['y_test'] if resultado_dl else None

    if resultado_ml is not None and tarea_actual == 'clasificacion':
        matrices_confusion(resultado_ml['y_test'], resultado_ml['predicciones'], 'parte6')
        df_clas = evaluar_clasificacion(resultado_ml['y_test'],
                                         resultado_ml['predicciones'])
        barras_metricas(df_clas.to_dict(orient='index'), 'parte6')

    if resultado_ml is not None and tarea_actual == 'regresion':
        df_reg = evaluar_regresion(resultado_ml['y_test'],
                                    resultado_ml['predicciones'])
        scatter_regresion(resultado_ml['y_test'], resultado_ml['predicciones'],
                          df_reg.to_dict(orient='index'), 'parte6')
        barras_errores_regresion(df_reg.to_dict(orient='index'), 'parte6')

    if resultado_dl is not None:
        r_b = resultado_dl['Red Basica']
        r_p = resultado_dl['Red Profunda']
        y_test = resultado_dl['y_test']

        if tarea_actual == 'clasificacion':
            from sklearn.metrics import accuracy_score, f1_score
            matrices_confusion(y_test, {'Red Basica': r_b['y_pred'],
                                        'Red Profunda': r_p['y_pred']}, 'parte7')
            curvas_entrenamiento_dl(r_b['historial'], r_p['historial'], 'clasificacion')
            acc_b = accuracy_score(y_test, r_b['y_pred'])
            f1_b = f1_score(y_test, r_b['y_pred'], average='weighted', zero_division=0)
            acc_p = accuracy_score(y_test, r_p['y_pred'])
            f1_p = f1_score(y_test, r_p['y_pred'], average='weighted', zero_division=0)
            comparacion_dl_clasificacion(acc_b, f1_b, r_b['tiempo'],
                                         acc_p, f1_p, r_p['tiempo'])
        else:
            from sklearn.metrics import mean_squared_error, r2_score
            curvas_entrenamiento_dl(r_b['historial'], r_p['historial'], 'regresion')
            rmse_b = np.sqrt(mean_squared_error(y_test, r_b['y_pred']))
            r2_b = r2_score(y_test, r_b['y_pred'])
            rmse_p = np.sqrt(mean_squared_error(y_test, r_p['y_pred']))
            r2_p = r2_score(y_test, r_p['y_pred'])
            scatter_regresion_dl(y_test, r_b['y_pred'], r_p['y_pred'], r2_b, r2_p)
            comparacion_dl_regresion(r2_b, rmse_b, r2_p, rmse_p)

    info("Todas las graficas se guardaron en output/")
    pausa()


def opcion_10():
    if resultado_ml is None:
        error("Primero entrena modelos ML (opcion 5).")
        pausa(); return

    modelos = list(resultado_ml['modelos'].keys())
    print("\n  Modelos disponibles:")
    for i, m in enumerate(modelos, 1):
        print(f"    {i}. {m}")
    try:
        idx = int(input("  Elige numero de modelo: ").strip()) - 1
    except ValueError:
        error("Numero invalido."); pausa(); return

    if idx < 0 or idx >= len(modelos):
        error("Numero fuera de rango."); pausa(); return

    nombre = modelos[idx]
    modelo = resultado_ml['modelos'][nombre]
    ruta = input("  Ruta del Excel con datos nuevos: ").strip()
    if not ruta:
        error("Debes proporcionar una ruta."); pausa(); return

    pred = predecir(ruta, modelo, tarea_actual)
    if pred is not None:
        info(f"Predicciones ({nombre}): {pred[:20]}..." if len(pred) > 20 else f"Predicciones ({nombre}): {pred}")
    pausa()


def main():
    while True:
        op = menu_principal()
        if op == '1': opcion_1()
        elif op == '2': opcion_2()
        elif op == '3': opcion_3()
        elif op == '4': opcion_4()
        elif op == '5': opcion_5()
        elif op == '6': opcion_6()
        elif op == '7': opcion_7()
        elif op == '8': opcion_8()
        elif op == '9': opcion_9()
        elif op == '10': opcion_10()
        elif op == '11':
            info("Hasta luego.")
            sys.exit(0)
        else:
            error("Opcion invalida. Intenta de nuevo.")
            pausa()


if __name__ == '__main__':
    main()
