"""
Transformación de datos: StandardScaler (Z-score) para numéricas,
Label Encoding para categóricas, y Feature Engineering
(categoria_consumo, nivel_demora, cliente_premium, ventas_por_minuto).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils import info

# Features que usan los modelos después de la transformación
FEATURES_FINALES = [
    'tiempo_preparacion_scaled', 'valor_total_scaled', 'hora_pedido_scaled',
    'cantidad_productos_scaled', 'calificacion_cliente_scaled', 'ventas_por_minuto_scaled',
    'ciudad_label', 'tipo_comida_label', 'metodo_pago_label', 'clima_label',
    'cliente_frecuente', 'cliente_premium', 'categoria_consumo', 'nivel_demora',
]
TARGETS = ['demora_entrega', 'ventas_dia', 'satisfaccion_cliente']

RANDOM_STATE = 42


def ya_transformado(df: pd.DataFrame) -> bool:
    return all(f in df.columns for f in FEATURES_FINALES[:3])


def transformar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if ya_transformado(df):
        info("Dataset ya transformado. Se conservan las transformaciones existentes.")
        return df

    info("Aplicando transformaciones...")
    df = df.copy()

    if 'id_pedido' not in df.columns:
        df['id_pedido'] = range(1, len(df) + 1)

    if 'ventas_por_minuto' not in df.columns or df['ventas_por_minuto'].isna().all():
        t = df.get('tiempo_preparacion', df.get('tiempo_preparacion_scaled', 1))
        v = df.get('valor_total', df.get('valor_total_scaled', 1))
        t = pd.to_numeric(t, errors='coerce').fillna(1).replace(0, 1)
        v = pd.to_numeric(v, errors='coerce').fillna(1)
        df['ventas_por_minuto'] = v / t
        df['ventas_por_minuto'] = df['ventas_por_minuto'].clip(
            0, df['ventas_por_minuto'].quantile(0.99))

    val_total = df.get('valor_total', df.get('valor_total_scaled', 0))
    val_total = pd.to_numeric(val_total, errors='coerce').fillna(0)
    df['cliente_premium'] = (val_total >= 45000).astype(int)

    if 'cliente_frecuente_raw' in df.columns:
        df['cliente_frecuente'] = (df['cliente_frecuente_raw'].str.lower() == 'si').astype(int)
    elif 'cliente_frecuente' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['cliente_frecuente']):
            df['cliente_frecuente'] = (df['cliente_frecuente'].astype(str).str.lower().str.strip() == 'sí').astype(int)
    else:
        df['cliente_frecuente'] = 0

    val = pd.to_numeric(val_total, errors='coerce').fillna(0)
    df['categoria_consumo'] = pd.cut(val, bins=[-1, 30000, 70000, float('inf')],
                                      labels=[0, 1, 2]).astype(int)

    tp = pd.to_numeric(df.get('tiempo_preparacion', df.get('tiempo_preparacion_scaled', 0)),
                       errors='coerce').fillna(0)
    df['nivel_demora'] = pd.cut(tp, bins=[-1, 15, 35, float('inf')],
                                 labels=[0, 1, 2]).astype(int)

    cols_cat = {
        'ciudad_raw': 'ciudad_label', 'tipo_comida_raw': 'tipo_comida_label',
        'metodo_pago_raw': 'metodo_pago_label', 'clima_raw': 'clima_label'
    }
    for raw, lbl in cols_cat.items():
        if raw in df.columns and lbl not in df.columns:
            df[lbl] = LabelEncoder().fit_transform(df[raw].astype(str))
        else:
            col_simple = raw.replace('_raw', '')
            if col_simple in df.columns and lbl not in df.columns:
                df[lbl] = LabelEncoder().fit_transform(df[col_simple].astype(str))

    cols_escalar = {
        'tiempo_preparacion': 'tiempo_preparacion_scaled',
        'valor_total': 'valor_total_scaled',
        'hora_pedido': 'hora_pedido_scaled',
        'cantidad_productos': 'cantidad_productos_scaled',
        'calificacion_cliente': 'calificacion_cliente_scaled',
        'ventas_por_minuto': 'ventas_por_minuto_scaled',
    }
    sz = StandardScaler()
    for raw, scl in cols_escalar.items():
        if raw in df.columns and scl not in df.columns:
            datos = pd.to_numeric(df[raw], errors='coerce').fillna(0).values.reshape(-1, 1)
            df[scl] = sz.fit_transform(datos).flatten()

    info(f"Transformación completada. Features finales: {len(FEATURES_FINALES)}")
    return df
