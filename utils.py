"""
Utilidades de consola: formato de salida, manejo de rutas,
limpieza de pantalla y pausas.
"""
import os
import sys

# Carpeta donde se guardan las gráficas generadas
RUTA_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
# Dataset por defecto (busca en el mismo directorio del proyecto)
RUTA_DATASET = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'restaurante.xlsx')


def limpiar_pantalla():
    os.system('cls' if os.name == 'nt' else 'clear')


def titulo(texto: str):
    print()
    print("=" * 65)
    print(f"  {texto}")
    print("=" * 65)


def subtitulo(texto: str):
    print(f"\n--- {texto} ---")


def info(texto: str):
    print(f"  {texto}")


def error(texto: str):
    print(f"\n  ERROR: {texto}", file=sys.stderr)


def pausa():
    input("\nPresiona Enter para continuar...")


def asegurar_output():
    os.makedirs(RUTA_OUTPUT, exist_ok=True)


def ruta_output(nombre: str) -> str:
    asegurar_output()
    return os.path.join(RUTA_OUTPUT, nombre)
