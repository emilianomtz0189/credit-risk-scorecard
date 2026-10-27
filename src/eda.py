"""Funciones de análisis exploratorio orientadas a riesgo de crédito."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import COLUMNA_OBJETIVO

COLOR_BARRAS = "#2a78d6"
COLOR_REFERENCIA = "#52514e"


def resumen_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    """Estadísticos descriptivos con percentiles extremos (1 %, 5 %, 95 %, 99 %).

    Los percentiles extremos muestran mejor que la media si hay valores atípicos.
    """
    return df.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T


def tasa_malos(df: pd.DataFrame, grupos) -> pd.DataFrame:
    """Clientes, malos y tasa de malos por grupo."""
    return (
        df.groupby(grupos, observed=True)[COLUMNA_OBJETIVO]
        .agg(clientes="size", malos="sum", tasa_malos="mean")
    )


def tasa_malos_por_faltante(df: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Compara la tasa de malos entre clientes con y sin dato en cada variable.

    Si las tasas difieren, el faltante es informativo y conviene conservarlo
    como grupo propio en lugar de imputarlo.
    """
    filas = []
    for col in columnas:
        falta = df[col].isna()
        filas.append({
            "variable": col,
            "clientes_sin_dato": int(falta.sum()),
            "tasa_malos_sin_dato": df.loc[falta, COLUMNA_OBJETIVO].mean(),
            "clientes_con_dato": int((~falta).sum()),
            "tasa_malos_con_dato": df.loc[~falta, COLUMNA_OBJETIVO].mean(),
        })
    return pd.DataFrame(filas).set_index("variable")


def _etiqueta_intervalo(intervalo: pd.Interval) -> str:
    """Formato legible para un intervalo de pd.qcut."""
    izquierda = max(intervalo.left, 0)
    def es_entero(x):
        return abs(x - round(x)) < 0.01
    # Enteros o montos grandes sin decimales; proporciones con 2 decimales
    if (es_entero(izquierda) and es_entero(intervalo.right)) or intervalo.right >= 100:
        decimales = 0
    else:
        decimales = 2
    return f"{izquierda:,.{decimales}f}–{intervalo.right:,.{decimales}f}"


def tasa_malos_por_tramos(df: pd.DataFrame, variable: str, n_tramos: int = 10,
                          tope: int | None = None) -> pd.DataFrame:
    """Tasa de malos por tramos de una variable, con los faltantes como grupo aparte.

    - Variables continuas (tope=None): tramos por cuantiles (deciles por omisión).
    - Variables de conteo (tope=k): un grupo por valor 0, 1, ..., k y "k+".
    """
    serie = df[variable]
    if tope is None:
        tramos = pd.qcut(serie, q=n_tramos, duplicates="drop")
        etiquetas = tramos.cat.categories.map(_etiqueta_intervalo)
        tramos = tramos.cat.rename_categories(list(etiquetas))
    else:
        valores = serie.clip(upper=tope)
        orden = [str(v) for v in range(tope)] + [f"{tope}+"]
        texto = valores.map(lambda v: np.nan if pd.isna(v) else (f"{tope}+" if v >= tope else str(int(v))))
        tramos = pd.Categorical(texto, categories=orden, ordered=True)
        tramos = pd.Series(tramos, index=serie.index)

    if serie.isna().any():
        tramos = tramos.cat.add_categories("Faltante").fillna("Faltante")
    return tasa_malos(df, tramos)


def graficar_tasa_malos(tabla: pd.DataFrame, titulo: str, tasa_global: float):
    """Barras de tasa de malos por tramo, con la tasa global como referencia."""
    fig, ax = plt.subplots(figsize=(8, 3.2))
    posiciones = np.arange(len(tabla))
    ax.bar(posiciones, tabla["tasa_malos"] * 100, color=COLOR_BARRAS, width=0.7)
    ax.axhline(tasa_global * 100, color=COLOR_REFERENCIA, linestyle="--", linewidth=1)
    ax.annotate(f"Tasa global {tasa_global:.1%}", xy=(1, tasa_global * 100),
                xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", ha="right", fontsize=8,
                color=COLOR_REFERENCIA)

    ax.set_xticks(posiciones)
    ax.set_xticklabels(tabla.index.astype(str), rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Tasa de malos (%)")
    ax.set_title(titulo, loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def pares_correlacionados(df: pd.DataFrame, umbral: float = 0.5) -> pd.DataFrame:
    """Pares de variables con correlación de Spearman en valor absoluto >= umbral.

    Se usa Spearman (basada en rangos) porque las variables tienen valores
    extremos y relaciones no lineales.
    """
    corr = df.corr(method="spearman")
    pares = []
    columnas = corr.columns
    for i, a in enumerate(columnas):
        for b in columnas[i + 1:]:
            if abs(corr.loc[a, b]) >= umbral:
                pares.append({"variable_1": a, "variable_2": b, "spearman": corr.loc[a, b]})
    if not pares:
        return pd.DataFrame(columns=["variable_1", "variable_2", "spearman"])
    return pd.DataFrame(pares).sort_values("spearman", key=abs, ascending=False, ignore_index=True)
