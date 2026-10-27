"""Configuración central del proyecto: rutas, semilla y parámetros globales.

Todas las rutas se construyen a partir de la raíz del repositorio, así el
código funciona igual en Colab, en una computadora local o en la de otra
persona, sin depender de rutas absolutas.
"""

import random
from pathlib import Path

import numpy as np

# --- Rutas -------------------------------------------------------------------
# Raíz del repositorio: la carpeta que contiene a src/
RAIZ = Path(__file__).resolve().parents[1]

DIR_DATOS = RAIZ / "data"
DIR_CRUDOS = DIR_DATOS / "raw"            # Datos tal como vienen de Kaggle
DIR_PROCESADOS = DIR_DATOS / "processed"  # Particiones y datos transformados

ARCHIVO_KAGGLE = DIR_CRUDOS / "cs-training.csv"
ARCHIVO_TRAIN = DIR_PROCESADOS / "train.csv"
ARCHIVO_TEST = DIR_PROCESADOS / "test.csv"
ARCHIVO_TRAIN_LIMPIO = DIR_PROCESADOS / "train_limpio.csv"
ARCHIVO_TEST_LIMPIO = DIR_PROCESADOS / "test_limpio.csv"
ARCHIVO_TRAIN_WOE = DIR_PROCESADOS / "train_woe.csv"
ARCHIVO_TEST_WOE = DIR_PROCESADOS / "test_woe.csv"

# --- Parámetros globales -----------------------------------------------------
SEMILLA = 42               # Semilla única para todo el proyecto
PROPORCION_PRUEBA = 0.30   # 70 % entrenamiento / 30 % prueba
COLUMNA_OBJETIVO = "malo"  # 1 = 90+ días de atraso en los siguientes 2 años


def fijar_semilla(semilla: int = SEMILLA) -> None:
    """Fija las semillas de los generadores aleatorios de Python y NumPy.

    Nota: los modelos de scikit-learn, XGBoost y LightGBM además reciben
    `random_state=SEMILLA` de forma explícita; eso es lo que garantiza que
    sus resultados sean idénticos en cada corrida.
    """
    random.seed(semilla)
    np.random.seed(semilla)


def crear_carpetas() -> None:
    """Crea las carpetas de datos si no existen (no se suben a GitHub)."""
    for carpeta in (DIR_CRUDOS, DIR_PROCESADOS):
        carpeta.mkdir(parents=True, exist_ok=True)
