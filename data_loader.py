"""Carga del dataset desde archivo Excel."""
import pandas as pd
from utils import info, error


def cargar_dataset(ruta: str) -> pd.DataFrame | None:
    """Carga un archivo Excel y retorna un DataFrame."""
    if not ruta.endswith('.xlsx'):
        error("El archivo debe ser .xlsx")
        return None
    try:
        df = pd.read_excel(ruta)
        info(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except FileNotFoundError:
        error(f"No se encontró el archivo: {ruta}")
        return None
    except Exception as e:
        error(f"Error al cargar el archivo: {e}")
        return None
