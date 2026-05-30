# FoodVision AI

Sistema inteligente de consola para gestión y predicción en restaurantes. Usa Machine Learning y Deep Learning para predecir retrasos en entregas (clasificación binaria) y ventas diarias (regresión), comparando modelos tradicionales con redes neuronales.

## Qué hace

- Carga datos históricos de operaciones de un restaurante desde un archivo Excel.
- Limpia automáticamente: imputa nulos, corrige valores negativos/extremos, normaliza texto, elimina duplicados.
- Transforma variables: StandardScaler (Z-score), Label Encoding y Feature Engineering (`categoria_consumo`, `nivel_demora`, `cliente_premium`, `ventas_por_minuto`).
- Entrena 4 modelos de clasificación (Árbol de Decisión, Random Forest, Regresión Logística, KNN) y 3 de regresión (Árbol, Random Forest, KNN).
- Entrena 2 redes neuronales: Red Básica (1 capa, 64 neuronas) y Red Profunda (128→64→32 + Dropout) con EarlyStopping.
- Evalúa con métricas: Accuracy, Precision, Recall, F1 (clasificación); MAE, MSE, RMSE, R² (regresión).
- Genera y guarda 10 gráficas en `output/` (matrices de confusión, barras, scatter, curvas de entrenamiento, comparativas).
- Predice sobre datos nuevos usando un modelo entrenado.

## Requisitos

- Python 3.10+
- El archivo de datos `restaurante_seleccionado.xlsx` debe estar en el directorio padre del proyecto (`../`).

## Instalación

```bash
cd foodvision_app
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

Aparecerá un menú interactivo numerado con 11 opciones. El flujo recomendado es:

1. **Cargar dataset** — por defecto busca `../restaurante.xlsx`, o puedes indicar otra ruta.
2. **Mostrar dataset original** — primeras 10 filas y dimensiones.
3. **Limpiar datos** — imputación de nulos, corrección de negativos/valores extremos, normalización de texto, eliminación de duplicados.
4. **Transformar datos** — StandardScaler, Label Encoding y Feature Engineering. Muestra el dataset transformado.
5. **Entrenar modelos ML** — elige clasificación (`c`) o regresión (`r`).
6. **Entrenar modelos DL** — Red Básica y Red Profunda para la misma tarea.
7. **Comparar métricas** — tabla comparativa de ML y métricas de DL.
8. **Métricas detalladas** — classification report y matriz de confusión para un modelo específico.
9. **Generar gráficas** — guarda todas las gráficas en `output/`.
10. **Predecir con datos nuevos** — aplica un modelo entrenado a un Excel nuevo.
11. **Salir** — termina el programa.

Cada paso muestra resultados formateados en consola y las gráficas se guardan automáticamente sin preguntar.

## Estructura del proyecto

```
foodvision_app/
├── main.py                # Punto de entrada, menú interactivo
├── data_loader.py         # Carga de archivos Excel
├── data_cleaner.py        # Limpieza de datos
├── data_transformer.py    # Transformaciones y Feature Engineering
├── ml_models.py           # Modelos de Machine Learning
├── dl_models.py           # Modelos de Deep Learning (TensorFlow/Keras)
├── evaluator.py           # Cálculo de métricas
├── visualizer.py          # Generación de gráficas
├── predictor.py           # Predicción con datos nuevos
├── utils.py               # Utilidades de consola y rutas
├── output/                # Gráficas generadas (PNG)
├── requirements.txt       # Dependencias
├── README.md              # Este archivo
├── plan_partes_9_10.md    # Plan del proyecto (partes 9 y 10)
├── Practica Final.docx    # Documento técnico completo
└── Practica Final formato.docx  # Enunciado/formato del taller
```

## Notas

- El dataset esperado tiene 14 columnas incluyendo: `id_pedido`, `ciudad`, `tipo_comida`, `hora_pedido`, `cantidad_productos`, `valor_total`, `tiempo_preparacion`, `metodo_pago`, `cliente_frecuente`, `calificacion_cliente`, `clima`, `demora_entrega`, `satisfaccion_cliente`, `ventas_dia`.
- Las columnas originales de texto esperan el sufijo `_raw` (ej. `ciudad_raw`). Si no están presentes, el transformador intenta mapear desde los nombres sin sufijo.
- Los modelos usan `random_state=42` para reproducibilidad.
- Las redes neuronales usan EarlyStopping con `patience=10`, máximo 80 epochs y batch size de 32.
- Las predicciones de clasificación en DL usan umbral 0.5 sobre la salida sigmoide.
