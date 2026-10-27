"""Reglas de limpieza: se aprenden con entrenamiento y se aplican igual a prueba.

Sigue la misma interfaz que scikit-learn (fit / transform) para que quede
explícito qué parámetros se calculan con la muestra de entrenamiento.
"""

import numpy as np
import pandas as pd

from src.config import (
    ARCHIVO_TEST_LIMPIO,
    ARCHIVO_TRAIN_LIMPIO,
)
from src.datos import obtener_particiones

# --- Reglas de negocio ---------------------------------------------------------
EDAD_MINIMA = 18                 # Edad mínima legal para contratar un crédito
ATRASOS_MAXIMO_POSIBLE = 24      # En 2 años caben a lo más 24 periodos mensuales
USO_MAXIMO_CONFIABLE = 10        # Uso de crédito > 10 veces el límite: error de captura
VARIABLES_ATRASO = ["atrasos_30_59", "atrasos_60_89", "atrasos_90"]
VARIABLES_A_ACOTAR = ["uso_credito_revolvente", "razon_deuda", "ingreso_mensual"]


class LimpiadorCredito:
    """Corrige valores imposibles y acota valores extremos.

    1. Edad menor a 18 -> faltante (valor imposible para un acreditado).
    2. Atrasos mayores a 24 -> faltante (son códigos especiales, no conteos).
    3. Razón de deuda sin ingreso (faltante o cero) -> faltante: sin
       denominador, el valor no puede ser una razón deuda/ingreso.
    4. Uso de crédito mayor a 10 -> faltante: en el análisis exploratorio
       esos clientes se comportan como el promedio, no como clientes
       sobregirados, así que el valor no es confiable.
    5. Acota uso de crédito, razón de deuda e ingreso en el percentil 99
       calculado SOLO con entrenamiento.

    Los faltantes no se imputan: en la etapa 3 forman su propio grupo de WoE,
    porque el hecho de que falte un dato puede estar relacionado con el riesgo.
    """

    def __init__(self, percentil_tope: float = 0.99):
        self.percentil_tope = percentil_tope
        self.topes_ = None  # Se llena con fit()

    @staticmethod
    def _corregir_invalidos(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Se pasan a decimal para poder guardar faltantes en columnas enteras
        enteras = ["edad"] + VARIABLES_ATRASO
        df[enteras] = df[enteras].astype(float)

        df.loc[df["edad"] < EDAD_MINIMA, "edad"] = np.nan
        for var in VARIABLES_ATRASO:
            df.loc[df[var] > ATRASOS_MAXIMO_POSIBLE, var] = np.nan

        sin_ingreso = df["ingreso_mensual"].isna() | (df["ingreso_mensual"] == 0)
        df.loc[sin_ingreso, "razon_deuda"] = np.nan

        df.loc[df["uso_credito_revolvente"] > USO_MAXIMO_CONFIABLE, "uso_credito_revolvente"] = np.nan
        return df

    def fit(self, df: pd.DataFrame) -> "LimpiadorCredito":
        """Calcula los topes (percentil 99) con la muestra de entrenamiento."""
        corregido = self._corregir_invalidos(df)
        self.topes_ = corregido[VARIABLES_A_ACOTAR].quantile(self.percentil_tope)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica las correcciones y los topes aprendidos en fit()."""
        if self.topes_ is None:
            raise RuntimeError("Primero ejecuta fit() con la muestra de entrenamiento.")
        df = self._corregir_invalidos(df)
        df[VARIABLES_A_ACOTAR] = df[VARIABLES_A_ACOTAR].clip(upper=self.topes_, axis=1)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def reporte(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cuántos registros modifica cada regla (sobre datos sin limpiar)."""
        if self.topes_ is None:
            raise RuntimeError("Primero ejecuta fit() con la muestra de entrenamiento.")
        sin_ingreso = df["ingreso_mensual"].isna() | (df["ingreso_mensual"] == 0)
        conteos = {"edad < 18 -> faltante": (df["edad"] < EDAD_MINIMA).sum()}
        for var in VARIABLES_ATRASO:
            conteos[f"{var} > 24 -> faltante"] = (df[var] > ATRASOS_MAXIMO_POSIBLE).sum()
        conteos["razon_deuda sin ingreso -> faltante"] = (sin_ingreso & df["razon_deuda"].notna()).sum()
        conteos["uso_credito_revolvente > 10 -> faltante"] = (df["uso_credito_revolvente"] > USO_MAXIMO_CONFIABLE).sum()
        corregido = self._corregir_invalidos(df)
        for var in VARIABLES_A_ACOTAR:
            conteos[f"{var} > {self.topes_[var]:,.2f} -> tope"] = (corregido[var] > self.topes_[var]).sum()

        tabla = pd.Series(conteos, name="registros").to_frame()
        tabla["pct"] = (tabla["registros"] / len(df) * 100).round(2)
        return tabla


def obtener_datos_limpios() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve entrenamiento y prueba limpios; si no existen, los genera.

    El limpiador se ajusta con entrenamiento y se aplica a ambas muestras,
    por lo que regenerar los archivos produce siempre el mismo resultado.
    """
    if ARCHIVO_TRAIN_LIMPIO.exists() and ARCHIVO_TEST_LIMPIO.exists():
        train = pd.read_csv(ARCHIVO_TRAIN_LIMPIO, index_col="id_cliente")
        test = pd.read_csv(ARCHIVO_TEST_LIMPIO, index_col="id_cliente")
        return train, test

    train, test = obtener_particiones()
    limpiador = LimpiadorCredito().fit(train)
    train_limpio, test_limpio = limpiador.transform(train), limpiador.transform(test)
    train_limpio.to_csv(ARCHIVO_TRAIN_LIMPIO)
    test_limpio.to_csv(ARCHIVO_TEST_LIMPIO)
    return train_limpio, test_limpio
