"""
Limpieza de datos: imputa nulos, corrige valores negativos,
normaliza texto y elimina duplicados. También binariza variables
categóricas como demora_entrega y satisfaccion_cliente.
"""
import pandas as pd
from utils import info

# Columnas numéricas que requieren imputación y corrección
COLUMNAS_NUMERICAS = [
    'tiempo_preparacion', 'valor_total', 'hora_pedido',
    'cantidad_productos', 'calificacion_cliente', 'ventas_dia',
    'ventas_por_minuto',
]
COLUMNAS_TEXTO = ['ciudad_raw', 'tipo_comida_raw', 'metodo_pago_raw', 'clima_raw',
                  'cliente_frecuente_raw']


def normalizar_texto(serie: pd.Series) -> pd.Series:
    return serie.astype(str).str.strip().str.lower().str.normalize('NFKD') \
        .str.encode('ascii', errors='ignore').str.decode('utf-8')


def limpiar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    filas_inicial = len(df)
    resumen = {}

    for col in df.columns:
        if col in COLUMNAS_NUMERICAS and pd.api.types.is_numeric_dtype(df[col]):
            n_nulos = df[col].isna().sum()
            if n_nulos > 0:
                mediana = df[col].median()
                df[col] = df[col].fillna(mediana)
                resumen[f'nulos_{col}'] = n_nulos

            negativos = (df[col] < 0).sum()
            if negativos > 0:
                df.loc[df[col] < 0, col] = 0
                resumen[f'negativos_{col}'] = negativos

        elif col in COLUMNAS_TEXTO:
            n_nulos = df[col].isna().sum()
            if n_nulos > 0:
                df[col] = df[col].fillna('desconocido')
                resumen[f'nulos_texto_{col}'] = n_nulos
            df[col] = normalizar_texto(df[col])

    if 'demora_entrega' in df.columns and not pd.api.types.is_numeric_dtype(df['demora_entrega']):
        df['demora_entrega'] = df['demora_entrega'].astype(str).str.strip().str.lower()
        df['demora_entrega'] = df['demora_entrega'].map({'si': 1, 'sí': 1, 'no': 0}).fillna(0).astype(int)
        resumen['demora_entrega_binarizada'] = 1

    if 'satisfaccion_cliente' in df.columns and not pd.api.types.is_numeric_dtype(df['satisfaccion_cliente']):
        df['satisfaccion_cliente'] = df['satisfaccion_cliente'].astype(str).str.strip().str.lower()
        nivel_map = {'alta': 2, 'media': 1, 'baja': 0}
        df['satisfaccion_cliente'] = df['satisfaccion_cliente'].map(nivel_map).fillna(1).astype(int)
        resumen['satisfaccion_cliente_codificada'] = 1

    duplicados = df.duplicated().sum()
    if duplicados > 0:
        df = df.drop_duplicates()
        resumen['duplicados_eliminados'] = duplicados

    info(f"Limpieza completada: {filas_inicial} -> {len(df)} filas")
    if resumen:
        for k, v in resumen.items():
            info(f"  {k}: {v}")
    else:
        info("  No se encontraron problemas que limpiar.")
    return df
